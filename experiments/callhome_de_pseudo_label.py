#!/usr/bin/env python
"""CallHome German(TalkBank, HF talkbank/callhome deu — CC BY-NC-SA 4.0) 전화 대화 → Phase 2 dialogues.jsonl (Stage 4 학습 재료, 2026-10-19).

HF 판은 화자 분리용으로 가공돼 오디오(mono 16 kHz, 통화의 라벨 구간만)와 화자 구간(시작·끝·화자)만 있고 전사가 없다.
nsc_pseudo_label.py 와 같은 두 교사 의사 전사를 쓴다(사용자 결정 2026-10-19: Whisper·Qwen3-ASR 둘이 WER 3 % 미만으로 맞는 발화만 학습).

  extract : CPU. parquet 5 개 → <out>/audio/<세션>.wav(원래 WAV 바이트 그대로) + <out>/callhome-de-segments.jsonl.
            세션 이름 = 전역 순번 3 자리 + 오디오 앞 1 MB sha1 8 자리(행 순서가 바뀌어도 오디오로 맞출 수 있게).
  asr     : GPU 샤드(세션 단위). 화자 구간마다 mono 를 그대로 잘라 Qwen3-ASR-1.7B(German)와 Whisper large-v3(de, 탐욕, 이전 문맥 없음)를 돌린다.
            구간은 최장 23.7 s 라 Whisper 30 s 창 하나에 들어간다. 상대 화자 소리가 섞인 구간은 두 교사가 어긋나 걸러진다.
  build   : 두 결과를 합쳐 <dest>/callhome-de.dialogues.jsonl. raw = Qwen 결과, text = asr-tn-de 학습 정규화.
            NSC 와 같은 채점 정규화 WER(Qwen 기준, Whisper 가설, 간투사 무시, Whisper 숫자는 수사로 편다)이 --max-wer 미만일 때만 감독에 쓰고,
            아니면(한쪽이 비어도) flags=asr_disagree — 시각은 유지하고 전사 감독에서 뺀다(학습 마스크 구간).
            채널: 화자별 채널이 없어 모든 화자의 ChannelRef 가 같은 mono 파일이다. mix_dialogue 는 같은 파형을 더하므로 결과는 원래 파형의 배율이고,
            p2_align 은 mono 의 발화 구간을 정렬한다(혼합 파형 정렬 — 겹침 구간은 이미 교사 불일치로 빠진다).

usage:
  python experiments/callhome_de_pseudo_label.py extract --src <…/data/callhome-deu> --out <work>
  python experiments/callhome_de_pseudo_label.py asr --out <work> --shard k/n            (hibiki env, CUDA_VISIBLE_DEVICES 로 GPU 지정)
  python experiments/callhome_de_pseudo_label.py build --out <work> --dest <phase2 dir> [--max-wer 0.03]
산출: <work>/audio, <work>/callhome-de-segments.jsonl, <work>/asr/shard-k-of-n.jsonl, <work>/callhome-de.agree.jsonl(발화별 WER — 전사 없음),
      <dest>/callhome-de.dialogues.jsonl(+ callhome-de.stats.json). 아무것도 지우지 않는다.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS = "callhome-de"
HF_REV = "talkbank/callhome@17c8a153215aa7c50b805078fd6284ba81c2fc47 (deu)"


def cmd_extract(a):
    import pyarrow.parquet as pq, soundfile as sf
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True); dest = os.path.join(a.out, f"{CORPUS}-segments.jsonl")
    assert not os.path.exists(dest), f"{dest} 가 이미 있다 — 새 --out 을 쓴다"
    n = 0; t0 = time.time()
    with open(dest + ".tmp", "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(os.path.join(a.src, "deu", "*.parquet"))):
            for r in pq.read_table(p).to_pylist():
                b = r["audio"]["bytes"]; key = f"{n:03d}_{hashlib.sha1(b[:1 << 20]).hexdigest()[:8]}"
                wav = os.path.join(a.out, "audio", key + ".wav")
                if not os.path.exists(wav):
                    with open(wav + ".tmp", "wb") as w: w.write(b)
                    os.replace(wav + ".tmp", wav)
                info = sf.info(io.BytesIO(b)); assert info.samplerate == 16000 and info.channels == 1, (key, info)
                segs = sorted((round(s, 3), round(e, 3), sp) for s, e, sp in zip(r["timestamps_start"], r["timestamps_end"], r["speakers"]) if e > s)
                f.write(json.dumps(dict(session=key, path=wav, duration_s=round(info.duration, 3), speakers=sorted({x[2] for x in segs}), segments=segs), ensure_ascii=False) + "\n")
                n += 1
    os.replace(dest + ".tmp", dest); print(f"extract: {n} sessions → {dest} ({time.time() - t0:.0f}s)")


def cmd_asr(a):
    import numpy as np, soundfile as sf, torch
    from qwen_asr import Qwen3ASRModel
    from simulstreaming.whisper.simul_whisper.whisper import load_model, log_mel_spectrogram, pad_or_trim, DecodingOptions, decode as wdecode
    rows = [json.loads(l) for l in open(os.path.join(a.out, f"{CORPUS}-segments.jsonl"), encoding="utf-8")]
    k, n = (int(v) for v in a.shard.split("/")); rows = rows[k::n]
    od = os.path.join(a.out, "asr"); os.makedirs(od, exist_ok=True); dest = os.path.join(od, f"shard-{k}-of-{n}.jsonl")
    done = {json.loads(l)["session"] for l in open(dest, encoding="utf-8")} if os.path.exists(dest) else set()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    qwen = Qwen3ASRModel.from_pretrained(a.qwen, dtype=torch.bfloat16, device_map="cuda")
    whisper = load_model(a.whisper, device="cuda")
    import simulstreaming.whisper.simul_whisper.whisper.decoding as _wd                  # nsc_pseudo_label 과 같은 우회: large-v3 의 언어 수(100)로 토크나이저를 만든다
    _gt = _wd.get_tokenizer
    _wd.get_tokenizer = lambda multilingual, **kw: _gt(multilingual, num_languages=whisper.num_languages, **kw)
    wopt = DecodingOptions(language="de", task="transcribe", temperature=0.0, fp16=True, without_timestamps=True)
    print(f"asr shard {k}/{n}: {len(rows)} sessions ({len(done)} done)", flush=True)
    with open(dest, "a", encoding="utf-8") as f:
        for r in rows:
            if r["session"] in done: continue
            t0 = time.time(); x, sr = sf.read(r["path"], dtype="float32"); assert sr == 16000
            cuts = [x[int(s * 16000): int(e * 16000)] for s, e, _ in r["segments"]]
            hyps, whyps = [], []
            for i in range(0, len(cuts), a.batch):
                res = qwen.transcribe(audio=[(c, 16000) for c in cuts[i:i + a.batch]], language=["German"] * len(cuts[i:i + a.batch]))
                hyps += [getattr(z, "text", str(z)) for z in res]
            for i in range(0, len(cuts), a.wbatch):
                mel = torch.stack([log_mel_spectrogram(pad_or_trim(torch.from_numpy(c)), n_mels=whisper.dims.n_mels, device="cuda") for c in cuts[i:i + a.wbatch]])
                whyps += [z.text.strip() for z in wdecode(whisper, mel, wopt)]
            out = [dict(i=j, speaker=sp, start=s, end=e, qwen=q.strip(), whisper=w) for j, ((s, e, sp), q, w) in enumerate(zip(r["segments"], hyps, whyps))]
            f.write(json.dumps(dict(session=r["session"], utts=out), ensure_ascii=False) + "\n"); f.flush()
            print(f"  {r['session']} {len(out)} utts {time.time() - t0:.0f}s", flush=True)


def cmd_build(a):
    from collections import Counter
    from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
    from vapasr.data.textnorm_de import target_de, target_flags_de, score_de, TEXTNORM_DE_ID
    from vapasr.data.dialogue_corpora import de_word_err
    from nsc_pseudo_label import expand_digits_de

    def wer(ref, hyp):                                                                  # nsc_pseudo_label.cmd_build 과 같다
        return de_word_err(score_de(ref), score_de(expand_digits_de(hyp)))
    seg = {json.loads(l)["session"]: json.loads(l) for l in open(os.path.join(a.out, f"{CORPUS}-segments.jsonl"), encoding="utf-8")}
    asr = {}
    for p in glob.glob(os.path.join(a.out, "asr", "shard-*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); asr[r["session"]] = r["utts"]
    st = Counter(); hours = kept_h = 0.0; dest = os.path.join(a.dest, f"{CORPUS}.dialogues.jsonl")
    assert not os.path.exists(dest), f"{dest} 가 이미 있다 — 지우지 않는다(새 --dest 또는 이름)"
    with open(dest + ".tmp", "w", encoding="utf-8") as f, open(os.path.join(a.out, f"{CORPUS}.agree.jsonl"), "w", encoding="utf-8") as fa:
        for s, r in sorted(seg.items()):
            if s not in asr: st["sessions_without_asr"] += 1; continue
            utts = []
            for u in sorted(asr[s], key=lambda u: (u["start"], u["speaker"], u["i"])):
                q, w = u["qwen"], u["whisper"]; fl = set(target_flags_de(q)) if q else {"empty"}
                e = wer(q, w) if q and w else 1.0
                if not q or not w or not e < a.max_wer: fl.add("asr_disagree")
                txt = "" if fl else target_de(q)
                if not fl and not txt.strip(): fl.add("empty_after_tn"); txt = ""
                uid = f"{u['speaker']}_{u['i']:05d}"
                utts.append(Utterance(speaker=u["speaker"], start=u["start"], end=u["end"], raw=q, text=txt, utt_id=uid, flags=sorted(fl) or None))
                fa.write(json.dumps(dict(conv=s, utt=uid, dur=round(u["end"] - u["start"], 3), wer=round(e, 4), kept=not fl)) + "\n")
                st["utts"] += 1; st["kept" if not fl else "quarantined"] += 1; kept_h += 0 if fl else (u["end"] - u["start"]) / 3600
                for x in fl: st[f"flag_{x}"] += 1
            spk = sorted({u.speaker for u in utts})
            d = Dialogue(conv_id=f"{CORPUS}:{s}", corpus=CORPUS, lang="German", split="train", duration_s=r["duration_s"], speakers=spk, utterances=utts,
                         channels={x: ChannelRef(path=r["path"], sr=16000) for x in spk},
                         meta=dict(source=HF_REV, license="CC BY-NC-SA 4.0", mono_only=True, n_speakers=len(spk), max_wer=a.max_wer,
                                   teachers=["Qwen3-ASR-1.7B", "whisper-large-v3"]))
            f.write(d.to_json() + "\n"); st["dialogues"] += 1; st[f"speakers={len(spk)}"] += 1; hours += r["duration_s"] / 3600
    os.replace(dest + ".tmp", dest)
    stats = dict(st, hours=round(hours, 2), kept_utt_hours=round(kept_h, 2), kept_share=round(st["kept"] / max(1, st["utts"]), 4), textnorm=TEXTNORM_DE_ID, max_wer=a.max_wer, rule="keep iff WER < max_wer")
    json.dump(stats, open(os.path.join(a.dest, f"{CORPUS}.stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("extract"); q.add_argument("--src", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("asr"); q.add_argument("--out", required=True); q.add_argument("--shard", default="0/1")
    q.add_argument("--qwen", default="/soundai/Model/Qwen3-ASR-1.7B")
    q.add_argument("--whisper", default="/soundai/users/tskim/VAPKT-data/vapkt-baselines/models/openai__whisper-large-v3/large-v3.pt")
    q.add_argument("--batch", type=int, default=32); q.add_argument("--wbatch", type=int, default=16); q.add_argument("--mem-frac", type=float, default=0.3)
    q = sub.add_parser("build"); q.add_argument("--out", required=True); q.add_argument("--dest", required=True); q.add_argument("--max-wer", type=float, default=0.03)
    a = ap.parse_args()
    {"extract": cmd_extract, "asr": cmd_asr, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
