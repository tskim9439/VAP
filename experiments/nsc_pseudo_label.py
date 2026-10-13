#!/usr/bin/env python
"""NSC(Nautilus Speaker Characterization, 독일어) 2 인 반자연 대화 → Phase 2 dialogues.jsonl (Stage 3, 2026-10-14).

쓰는 파일: 세션마다 `<spk>/<spk>_semispontaneousdialogs_{speaker,interlocutor}.wav`(48 kHz, 두 채널 길이 같음, Dialog 5–8 전체).
전사가 없어 교사 ASR 로 의사 전사를 만든다. 상대는 거의 한 사람(amsterdam)이고 일부 세션은 aden 이다(meta.interlocutor).

  segment : CPU. 50 fps 에너지로 채널마다 발화 구간을 자른다. 프레임이 그 채널 화자의 말이려면
            (1) 자기 채널이 바닥 잡음 + FLOOR_DB 이상이고 (2) 상대 채널보다 LEAK_DB 이상 작지 않다(상대 말이 새어 든 프레임 제외).
            채널 분리도(상대가 크게 말할 때 내 채널 레벨이 내가 말할 때보다 낮은 정도)는 24 세션 중앙값 26 dB, 하위 10 % 20 dB 였다
            (2026-10-14 실측) — LEAK_DB = 16 이면 누설 프레임은 빠지고 두 사람이 동시에 말하는 프레임은 남는다.
            0 으로 편집된 구간(−95 dB 미만)은 바닥 잡음 추정에서 뺀다. gap < 0.5 s 병합, 0.3 s 미만 버림, 20 s 넘으면 가장 조용한 프레임에서 자른다.
  asr     : GPU 샤드(세션 단위). 구간마다 Qwen3-ASR-1.7B(독일어)와 Whisper large-v3(독일어, 탐욕 디코드, 이전 문맥 없음, 배치 decode)를 돌린다.
            hibiki env(vapasr env + tiktoken)에서 돌리고 PYTHONPATH 에 SimulStreaming 코드(번들 Whisper)를 둔다.
  build   : 두 결과를 합쳐 `<out>/nsc.dialogues.jsonl`. raw = Qwen 결과, text = asr-tn-de 학습 정규화.
            asr-tn-de 채점 정규화 WER(Qwen 기준, Whisper 가설; 간투사 무시, Whisper 숫자는 수사로 편다) > --max-wer 이거나 한쪽이 비면
            flags=asr_disagree(시각은 유지, 전사 감독 제외).

usage:
  python experiments/nsc_pseudo_label.py segment --root <VAPKT-data/data/NSC> --out <dir>
  python experiments/nsc_pseudo_label.py asr --out <dir> --shard k/n            (CUDA_VISIBLE_DEVICES 로 GPU 지정)
  python experiments/nsc_pseudo_label.py build --root <…/NSC> --out <dir> [--dest <phase2 dir>] [--max-wer 0.2]
산출: <dir>/nsc-segments.jsonl, <dir>/nsc-asr/shard-k-of-n.jsonl, <dest>/nsc.dialogues.jsonl(+ nsc.stats.json). 아무것도 지우지 않는다.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FPS = 50
FLOOR_DB = 15.0
LEAK_DB = 16.0
MERGE_S = 0.5            # KEC 발화 경계(KEC_GAP_S)와 같게 — 0.3 이면 구간 중앙값이 1.2–1.7 s 로 짧아 교사 ASR 이 불리하다(2026-10-14)
MIN_S = 0.3
MAX_S = 20.0
ROLES = ("speaker", "interlocutor")


def frame_db(x, sr):
    import numpy as np
    hop = sr // FPS; n = len(x) // hop
    return 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, 1) + 1e-12)


def segments_from_energy(e_self, e_other):
    """프레임 에너지(dB, 50 fps) 두 줄 → 자기 채널 화자의 발화 구간 [(start_s, end_s)]."""
    import numpy as np
    valid = e_self > -95
    floor = np.percentile(e_self[valid], 10) if valid.any() else -95.0
    act = (e_self > floor + FLOOR_DB) & (e_self >= e_other - LEAK_DB)
    act = np.convolve(act.astype(float), np.ones(5) / 5, mode="same") > 0.5         # 100 ms 다수결로 튀는 프레임 정리
    segs, i, n = [], 0, len(act)
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]: j += 1
            segs.append([i, j]); i = j
        else:
            i += 1
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) / FPS < MERGE_S: merged[-1][1] = e
        else: merged.append([s, e])
    out = []
    stack = [tuple(x) for x in merged]
    while stack:
        s, e = stack.pop(0)
        if (e - s) / FPS < MIN_S: continue
        if (e - s) / FPS > MAX_S:                                                     # 가운데 60 % 안에서 가장 조용한 프레임에서 자른다
            a, b = s + (e - s) // 5, e - (e - s) // 5; cut = a + int(np.argmin(e_self[a:b]))
            stack[:0] = [(s, cut), (cut + 1, e)]; continue
        out.append((round(s / FPS, 2), round(e / FPS, 2)))
    return out


_ONES = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn",
         "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]


def num_de(n):
    """정수(0–999,999) → 독일어 기수 한 단어(1 = eins, 21 = einundzwanzig, 101 = hunderteins). 교사 일치 판정용."""
    def lt100(k): return _ONES[k] if k < 20 else (("ein" if k % 10 == 1 else _ONES[k % 10]) + "und" if k % 10 else "") + _TENS[k // 10]
    def lt1000(k): return ((("ein" if k // 100 == 1 else _ONES[k // 100]) + "hundert") if k >= 100 else "") + (lt100(k % 100) if k % 100 or k < 100 else "")
    if n < 1000: return lt1000(n)
    t, r = divmod(n, 1000)
    return ("ein" if t == 1 else lt1000(t)) + "tausend" + (lt1000(r) if r else "")


def expand_digits_de(text):
    """Whisper 가설의 숫자를 독일어 수사로 편다(Qwen3-ASR 는 수사로 적는다). 0 으로 시작하거나 7 자리 이상이면 한 자리씩 읽은 것으로 본다."""
    def one(m):
        d = m.group(0)
        if d.startswith("0") and len(d) > 1 or len(d) > 6: return " ".join(_ONES[int(c)] for c in d)
        return num_de(int(d))
    return re.sub(r"\d+", one, re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text))


def sessions(root):
    for p in sorted(glob.glob(os.path.join(root, "*", "*_semispontaneousdialogs_speaker.wav"))):
        spk = os.path.basename(os.path.dirname(p)); q = p.replace("_speaker.wav", "_interlocutor.wav")
        if os.path.exists(q): yield spk, {"speaker": p, "interlocutor": q}


def interlocutor_of(root, spk):
    try:
        t = open(os.path.join(root, spk, f"{spk}.cmdi.xml"), errors="ignore").read().lower()
    except OSError:
        return None
    names = [n for n in ("amsterdam", "aden") if n in t]
    return names[0] if len(names) == 1 else ("+".join(names) if names else None)


def cmd_segment(a):
    import soundfile as sf
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, "nsc-segments.jsonl"); n = 0; t0 = time.time()
    with open(dest + ".tmp", "w", encoding="utf-8") as f:
        for spk, paths in sessions(a.root):
            x = {r: sf.read(paths[r], dtype="float32") for r in ROLES}; sr = x["speaker"][1]
            e = {r: frame_db(x[r][0], sr) for r in ROLES}; m = min(len(e[r]) for r in ROLES)
            dur = min(len(x[r][0]) for r in ROLES) / sr
            segs = {r: segments_from_energy(e[r][:m], e[ROLES[1 - i]][:m]) for i, r in enumerate(ROLES)}
            zero = {r: float((e[r][:m] < -95).mean()) for r in ROLES}
            f.write(json.dumps(dict(session=spk, paths=paths, sr=sr, duration_s=round(dur, 3), interlocutor=interlocutor_of(a.root, spk), zero_frac=zero, segments=segs)) + "\n")
            n += 1
            if n % 25 == 0: print(f"  {n} sessions {time.time() - t0:.0f}s", flush=True)
    os.replace(dest + ".tmp", dest); print(f"segment: {n} sessions → {dest} ({time.time() - t0:.0f}s)")


def cmd_asr(a):
    import numpy as np, soundfile as sf, torch
    from scipy.signal import resample_poly
    from qwen_asr import Qwen3ASRModel
    from simulstreaming.whisper.simul_whisper.whisper import load_model, log_mel_spectrogram, pad_or_trim, DecodingOptions, decode as wdecode
    rows = [json.loads(l) for l in open(os.path.join(a.out, "nsc-segments.jsonl"), encoding="utf-8")]
    k, n = (int(v) for v in a.shard.split("/")); rows = rows[k::n]
    od = os.path.join(a.out, "nsc-asr"); os.makedirs(od, exist_ok=True); dest = os.path.join(od, f"shard-{k}-of-{n}.jsonl")
    done = {json.loads(l)["session"] for l in open(dest, encoding="utf-8")} if os.path.exists(dest) else set()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    qwen = Qwen3ASRModel.from_pretrained(a.qwen, dtype=torch.bfloat16, device_map="cuda")
    whisper = load_model(a.whisper, device="cuda")
    import simulstreaming.whisper.simul_whisper.whisper.decoding as _wd                  # 번들 DecodingTask 는 언어 수 99 로 토크나이저를 만든다 → large-v3(100)는
    _gt = _wd.get_tokenizer                                                              # 언어·과제 토큰 id 가 하나씩 밀려 영어 번역을 낸다(2026-10-14 스모크)
    _wd.get_tokenizer = lambda multilingual, **kw: _gt(multilingual, num_languages=whisper.num_languages, **kw)
    wopt = DecodingOptions(language="de", task="transcribe", temperature=0.0, fp16=True, without_timestamps=True)   # 번들 transcribe() 는 mel 80 고정이라 large-v3(128)에 못 쓴다
    print(f"asr shard {k}/{n}: {len(rows)} sessions ({len(done)} done)", flush=True)
    with open(dest, "a", encoding="utf-8") as f:
        for r in rows:
            if r["session"] in done: continue
            t0 = time.time(); out = []
            for role in ROLES:
                x, sr = sf.read(r["paths"][role], dtype="float32"); x = resample_poly(x, 1, sr // 16000).astype(np.float32) if sr != 16000 else x
                cuts = [x[int(s * 16000): int(e * 16000)] for s, e in r["segments"][role]]
                hyps = []
                for i in range(0, len(cuts), a.batch):
                    res = qwen.transcribe(audio=[(c, 16000) for c in cuts[i:i + a.batch]], language=["German"] * len(cuts[i:i + a.batch]))
                    hyps += [getattr(z, "text", str(z)) for z in res]
                whyps = []
                for i in range(0, len(cuts), a.wbatch):                       # 구간 ≤ 20 s 라 30 s 창 하나에 들어간다(이전 문맥 없음, 탐욕 디코드)
                    mel = torch.stack([log_mel_spectrogram(pad_or_trim(torch.from_numpy(c)), n_mels=whisper.dims.n_mels, device="cuda") for c in cuts[i:i + a.wbatch]])
                    whyps += [z.text.strip() for z in wdecode(whisper, mel, wopt)]
                for (s, e), q, w in zip(r["segments"][role], hyps, whyps):
                    out.append(dict(role=role, start=s, end=e, qwen=q.strip(), whisper=w))
            f.write(json.dumps(dict(session=r["session"], utts=out), ensure_ascii=False) + "\n"); f.flush()
            print(f"  {r['session']} {len(out)} utts {time.time() - t0:.0f}s", flush=True)


def cmd_build(a):
    from vapasr.data.dialogue import Dialogue, Utterance, ChannelRef
    from vapasr.data.textnorm_de import target_de, target_flags_de, score_de, TEXTNORM_DE_ID
    from collections import Counter

    from vapasr.data.dialogue_corpora import de_word_err

    def wer(ref, hyp):                                                                  # 간투사 무시, Whisper 숫자는 수사로 펴서 비교
        return de_word_err(score_de(ref), score_de(expand_digits_de(hyp)))
    seg = {json.loads(l)["session"]: json.loads(l) for l in open(os.path.join(a.out, "nsc-segments.jsonl"), encoding="utf-8")}
    asr = {}
    for p in glob.glob(os.path.join(a.out, "nsc-asr", "shard-*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); asr[r["session"]] = r["utts"]
    st = Counter(); hours = 0.0; dd = a.dest or a.out; dest = os.path.join(dd, "nsc.dialogues.jsonl")
    with open(dest + ".tmp", "w", encoding="utf-8") as f:
        for s, r in sorted(seg.items()):
            if s not in asr: st["sessions_without_asr"] += 1; continue
            utts = []
            for i, u in enumerate(sorted(asr[s], key=lambda u: (u["start"], u["role"]))):
                q, w = u["qwen"], u["whisper"]; fl = set(target_flags_de(q)) if q else {"empty"}
                e = wer(q, w) if q else 1.0
                if not q or not w or e > a.max_wer: fl.add("asr_disagree")
                txt = "" if fl else target_de(q)
                utts.append(Utterance(speaker=u["role"], start=u["start"], end=u["end"], raw=q, text=txt, utt_id=f"{u['role']}_{i:05d}", flags=sorted(fl) or None))
                st["utts"] += 1; st["kept" if not fl else "quarantined"] += 1
                for x in fl: st[f"flag_{x}"] += 1
            d = Dialogue(conv_id=f"nsc:{s}", corpus="NSC", lang="German", split="train", duration_s=r["duration_s"], speakers=list(ROLES), utterances=utts,
                         channels={role: ChannelRef(path=r["paths"][role], sr=r["sr"]) for role in ROLES},
                         meta=dict(speaker_id=s, interlocutor=r.get("interlocutor"), zero_frac=r.get("zero_frac"), max_wer=a.max_wer))
            f.write(d.to_json() + "\n"); st["dialogues"] += 1; hours += r["duration_s"] / 3600; st[f"interlocutor={r.get('interlocutor')}"] += 1
    os.replace(dest + ".tmp", dest)
    stats = dict(st, hours=round(hours, 2), textnorm=TEXTNORM_DE_ID, max_wer=a.max_wer)
    json.dump(stats, open(os.path.join(dd, "nsc.stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("segment"); q.add_argument("--root", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("asr"); q.add_argument("--out", required=True); q.add_argument("--shard", default="0/1")
    q.add_argument("--qwen", default="/soundai/Model/Qwen3-ASR-1.7B")
    q.add_argument("--whisper", default="/soundai/users/tskim/VAPKT-data/vapkt-baselines/models/openai__whisper-large-v3/large-v3.pt")
    q.add_argument("--batch", type=int, default=32); q.add_argument("--wbatch", type=int, default=16); q.add_argument("--mem-frac", type=float, default=0.3)
    q = sub.add_parser("build"); q.add_argument("--root", required=True); q.add_argument("--out", required=True); q.add_argument("--max-wer", type=float, default=0.2)
    q.add_argument("--dest", help="nsc.dialogues.jsonl 을 쓸 폴더(기본 --out). 정렬·정제가 코퍼스 이름 nsc 로 읽도록 phase2 폴더를 준다")
    a = ap.parse_args()
    {"segment": cmd_segment, "asr": cmd_asr, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
