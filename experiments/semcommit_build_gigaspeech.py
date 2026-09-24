#!/usr/bin/env python3
"""GigaSpeech test(podcast·YouTube 자발 발화)에서 EN 스트림을 골라 semcommit words.jsonl 을 만든다 — 골드셋의 대화체 EN 몫(E2·semcommit 학습에 안 쓴 코퍼스).

선택: text_tn 의 태그가 구두점(<COMMA> <PERIOD> <QUESTIONMARK> <EXCLAMATIONPOINT>)뿐(<OTHER>·<MUSIC>·<NOISE>·<SIL> 이 있으면 제외), 길이 [min_s, max_s],
  단어 [min_w, max_w], 문장 끝 태그가 끝 말고도 1 개 이상(스트림 중간 확정 자리), 같은 원본 오디오(aid)에서 최대 --per-aid 개, podcast·youtube 반반(sha1(seed:sid) 순).
오디오: test_chunks_*.tar.gz 에서 <sid>.wav 만 out-dir/wav 로 꺼낸다(16 kHz mono 확인).
텍스트: 태그를 뺀 단어열 → textnorm.target_en = 학습 타깃(lexical), 태그를 앞 단어의 구두점으로 바꾼 전사 = pnc_text·EN 구두점 태그(LibriSpeech-PC 와 같은 쓰임).
정렬: Qwen3-ForcedAligner + s1_align 과 같은 BPE 토큰 종료 시각 로직 → aligned 행 → vapasr.data.semcommit_words.build_stream(roots=None, pnc_lookup={sid: pnc}).
출력: out-dir/words-gs-test.jsonl (+ gs-test.aligned.jsonl, gs-test.streams.jsonl, gs-test.stats.json). 이미 꺼낸 wav 는 다시 쓰지 않는다.

  python experiments/semcommit_build_gigaspeech.py --giga /data5/GigaSpeech/gigaspeech/data --aligner <Qwen3-ForcedAligner-0.6B dir> --qwen <Qwen3-ASR-0.6B dir> \\
      --n 80 --out-dir /data4/tskim/semcommit/data/gold --gpu 0
"""
import argparse, csv, glob, hashlib, io, json, os, re, sys, tarfile, time
from collections import Counter, defaultdict

if __name__ == "__main__" and "--gpu" in sys.argv[:-1]:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PUNCT = {"<COMMA>": ",", "<PERIOD>": ".", "<QUESTIONMARK>": "?", "<EXCLAMATIONPOINT>": "!"}
FINAL = {"<PERIOD>", "<QUESTIONMARK>", "<EXCLAMATIONPOINT>"}
SR = 16000


def parse_text(text_tn: str):
    """GigaSpeech text_tn → (단어열 소문자, 구두점 전사, 문장 끝 태그 수, 끝 말고 문장 끝 수, 허용 안 된 태그 목록)."""
    toks = text_tn.split(); words, pnc, bad, finals, mid = [], [], [], 0, 0
    for j, t in enumerate(toks):
        if t.startswith("<") and t.endswith(">"):
            if t not in PUNCT: bad.append(t); continue
            if pnc: pnc[-1] += PUNCT[t]
            if t in FINAL:
                finals += 1; mid += int(any(not (x.startswith("<") and x.endswith(">")) for x in toks[j + 1:]))
            continue
        words.append(t.lower()); pnc.append(t.lower())
    return words, " ".join(pnc), finals, mid, bad


def select(rows, a):
    """조건을 만족하는 세그먼트 → podcast·youtube 반반, aid 당 최대 per_aid, sha1(seed:sid) 순."""
    pools = defaultdict(list); why = Counter()
    for r in rows:
        src = r["source"]; dur = float(r["end_time"]) - float(r["begin_time"]); words, pnc, finals, mid, bad = parse_text(r["text_tn"])
        if src not in ("podcast", "youtube"): why["source"] += 1; continue
        if bad: why["nonspeech_tag"] += 1; continue
        if not (a.min_s <= dur <= a.max_s): why["duration"] += 1; continue
        if not (a.min_w <= len(words) <= a.max_w): why["n_words"] += 1; continue
        if mid < 1: why["no_mid_sentence_end"] += 1; continue
        if any(ch.isdigit() for w in words for ch in w): why["digit"] += 1; continue
        why["ok"] += 1; pools[src].append(dict(r, dur=dur, words=words, pnc=pnc, finals=finals))
    out, per_aid = [], Counter(); want = {"podcast": a.n // 2, "youtube": a.n - a.n // 2}
    for src, pool in pools.items():
        pool.sort(key=lambda r: hashlib.sha1(f"{a.seed}:{r['sid']}".encode()).hexdigest()); k = 0
        for r in pool:
            if k >= want[src]: break
            if per_aid[r["aid"]] >= a.per_aid: continue
            per_aid[r["aid"]] += 1; out.append(r); k += 1
    return out, why


def extract(giga, chosen, wav_dir):
    """chunk 별 tar.gz 를 한 번씩 훑어 고른 sid 의 wav 만 쓴다(있으면 건너뜀)."""
    import soundfile as sf
    os.makedirs(wav_dir, exist_ok=True); by_chunk = defaultdict(set)
    for r in chosen: by_chunk[r["chunk"]].add(r["sid"])
    for chunk, sids in sorted(by_chunk.items()):
        todo = {s for s in sids if not os.path.exists(os.path.join(wav_dir, f"{s}.wav"))}
        if not todo: continue
        with tarfile.open(os.path.join(giga, "audio", "test_files", f"{chunk}.tar.gz"), "r|gz") as tf:
            for m in tf:
                sid = os.path.splitext(os.path.basename(m.name))[0]
                if sid not in todo: continue
                data = tf.extractfile(m).read(); x, sr = sf.read(io.BytesIO(data), dtype="float32")
                assert sr == SR and x.ndim == 1, f"{sid}: sr {sr} shape {x.shape}"
                sf.write(os.path.join(wav_dir, f"{sid}.wav"), x, SR); todo.discard(sid)
                if not todo: break
        assert not todo, f"{chunk}: tar 에 없는 sid {sorted(todo)[:3]}"


def tokens_from_items(tok, items, text):
    """aligner 항목(단어 시각) → BPE 토큰 종료 시각 — experiments/s1_align.py tokens_from_items 와 같은 로직."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans, cur = [], 0
    for it in items:
        s0 = text.find(it.text, cur)
        if s0 < 0: s0 = cur
        spans.append((s0, s0 + len(it.text), float(it.end_time))); cur = s0 + len(it.text)
    out = []
    for tid, (o0, o1) in zip(ids, offs):
        cover = [sp for sp in spans if sp[0] < o1 and sp[1] > o0]
        t_end = max(c[2] for c in cover) if cover else (out[-1][1] if out else 0.0)
        out.append([int(tid), round(t_end, 3)])
    for j in range(1, len(out)): out[j][1] = max(out[j][1], out[j - 1][1])                    # 단조(드문 역전 보정)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--giga", required=True, help="GigaSpeech data 디렉터리(audio/test_files, metadata/test_metadata)")
    ap.add_argument("--aligner", required=True); ap.add_argument("--qwen", required=True, help="Qwen3-ASR tokenizer 디렉터리(vocab.json)")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--n", type=int, default=80); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-s", type=float, default=8.0); ap.add_argument("--max-s", type=float, default=25.0)
    ap.add_argument("--min-w", type=int, default=20); ap.add_argument("--max-w", type=int, default=80); ap.add_argument("--per-aid", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--gpu", default=None)
    a = ap.parse_args()
    t0 = time.time(); rows = []
    for p in sorted(glob.glob(os.path.join(a.giga, "metadata", "test_metadata", "test_chunks_*_metadata.csv"))):
        chunk = os.path.basename(p).replace("_metadata.csv", "")
        with open(p, newline="", encoding="utf-8") as f: rows += [dict(r, chunk=chunk) for r in csv.DictReader(f)]
    chosen, why = select(rows, a)
    print(f"metadata {len(rows)} 세그먼트 → 조건 {dict(why)} → 선택 {len(chosen)} ({Counter(r['source'] for r in chosen)})", flush=True)
    wav_dir = os.path.join(a.out_dir, "wav"); extract(a.giga, chosen, wav_dir); print(f"wav 추출 {time.time() - t0:.0f}s", flush=True)

    import numpy as np, soundfile as sf, torch
    from qwen_asr import Qwen3ForcedAligner
    from transformers import AutoTokenizer
    from vapasr.data.textnorm import target_en
    from vapasr.data.semcommit_words import PieceVocab, build_stream
    aligner = Qwen3ForcedAligner.from_pretrained(a.aligner, dtype=torch.bfloat16, device_map="cuda"); tok = AutoTokenizer.from_pretrained(a.qwen); pv = PieceVocab(a.qwen)
    aligned, streams, words_rows, drops = [], [], [], Counter(); pnc = {}
    items = []
    for r in chosen:
        lex = target_en(" ".join(r["words"]))
        x, sr = sf.read(os.path.join(wav_dir, f"{r['sid']}.wav"), dtype="float32"); assert sr == SR
        items.append(dict(r=r, lex=lex, audio=x)); pnc[r["sid"]] = r["pnc"]
    for b in range(0, len(items), a.batch):
        batch = items[b:b + a.batch]
        res = aligner.align(audio=[(it["audio"], SR) for it in batch], text=[it["lex"] for it in batch], language=["English"] * len(batch))
        for it, rr in zip(batch, res):
            r = it["r"]; its = rr.items if hasattr(rr, "items") else rr; dur = round(len(it["audio"]) / SR, 3)
            sid = f"gs-test-{r['source']}-{r['sid']}"; path = os.path.join(os.path.abspath(wav_dir), f"{r['sid']}.wav")
            seg = dict(path=path, offset_s=0.0, silence_before_s=0.0, dur_s=dur)
            arow = dict(name="gs-test", id=sid, K=int(round(dur / 0.08)), npy=None, lang="English", subset=r["source"], mode="utt",
                        tokens=tokens_from_items(tok, its, it["lex"]), text=it["lex"], duration_s=dur, segments=[seg])
            srow = dict(id=sid, corpus="gigaspeech", split="test", subset=r["source"], mode="utt", lang="English", duration_s=dur, n_utts=1,
                        segments=[dict(seg, utt_id=r["sid"], text=it["lex"], lexical_text=it["lex"], raw_text=r["text_tn"])], title=r.get("title"), aid=r["aid"])
            info = {}; out = build_stream(arow, srow, pv, roots=None, pnc_lookup=pnc, info=info, set_name="gs-test")
            aligned.append(arow); streams.append(srow)
            if isinstance(out, tuple): drops[out[1]] += 1; continue
            words_rows.append(out)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows_ in (("gs-test.aligned.jsonl", aligned), ("gs-test.streams.jsonl", streams), ("words-gs-test.jsonl", words_rows)):
        with open(os.path.join(a.out_dir, name), "w", encoding="utf-8") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows_)
    nw = sum(len(r["words"]) for r in words_rows); hours = sum(r["duration_s"] for r in words_rows) / 3600
    stats = dict(selection=vars(a), population=dict(metadata=len(rows), **{f"why_{k}": v for k, v in why.items()}), chosen=len(chosen), kept=len(words_rows), dropped=dict(drops),
                 words=nw, hours=round(hours, 3), with_pnc=sum(1 for r in words_rows if r.get("pnc_text")), sources=dict(Counter(r["id"].split("-")[2] for r in words_rows)),
                 seconds=round(time.time() - t0, 1))
    json.dump(stats, open(os.path.join(a.out_dir, "gs-test.stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False)[:900], flush=True)


if __name__ == "__main__":
    main()
