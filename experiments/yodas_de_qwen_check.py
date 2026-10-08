#!/usr/bin/env python3
"""YODAS-Granary German: compare the Granary transcripts with Qwen3-ASR-1.7B on a stratified sample.

Granary's German transcripts are Whisper-large-v3 pseudo-labels with LLM-restored punctuation (Koluguri et al.,
Interspeech 2025). Before deciding whether an extra ASR-based filter is needed, we measure how often an independent
model (Qwen3-ASR-1.7B, language given) agrees with them.

  python experiments/yodas_de_qwen_check.py --root <yodas_granary/data> --out <dir> [--per-stratum 250]

Sampling: for every subset (de000, de100, ...) x task (asr_only, ast), --files random parquet files, one random row
group each, --per-file random rows from it (seeded). Outputs in --out:
  results.jsonl   utt_id, subset, task, duration, ref (Granary text), hyp (Qwen), normalised forms, wer, flags
  summary.json    per stratum and overall: WER quantiles and the share of utterances within 3 / 5 / 10 / 30 % WER
  review.tsv      disagreements (WER > 3 %) sorted by WER, for reading
Comparison only: both texts go through `norm_de` (lower case, punctuation removed, digits spelled out with
num2words, ss for sharp s, hyphens split, German fillers dropped); stored texts are never changed.
"""
import argparse
import io
import json
import random
import re
import statistics as st
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

FILLERS = {"äh", "ähm", "öh", "öhm", "hm", "hmm", "mhm", "mm", "eh", "ehm", "em", "uh", "um"}
# Phrases that Whisper is known to hallucinate on German YouTube audio (silence, music, credits).
HALLUCINATION = re.compile(r"untertitel|amara\.org|vielen dank für(s| das) zuschauen|copyright|swr \d{4}|"
                           r"ard text|zdf|abonnier|bis zum nächsten mal", re.I)
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def norm_de(text):
    s = unicodedata.normalize("NFC", text or "")
    try:
        from num2words import num2words

        def spell(m):
            x = m.group(0)
            try:
                if "," in x or "." in x:
                    a, b = re.split(r"[.,]", x, maxsplit=1)
                    if len(b) == 3 and "." in x:          # 1.000 = thousand separator
                        return " " + num2words(int(a + b), lang="de") + " "
                    return " " + num2words(float(a + "." + b), lang="de") + " "
                return " " + num2words(int(x), lang="de") + " "
            except (ValueError, OverflowError, NotImplementedError):
                return " " + x + " "
        s = _NUM.sub(spell, s)
    except ImportError:
        pass
    s = s.lower().replace("ß", "ss")
    s = re.sub(r"[-‐–—/]", " ", s)
    s = re.sub(r"[^\w\s']|_", " ", s)
    s = s.replace("'", " ")
    return [w for w in s.split() if w not in FILLERS]


def edit_ops(ref, hyp):
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j - 1] + (r != h), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def repeated_ngram(words, n=3, k=3):
    seen = defaultdict(int)
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        seen[g] += 1
        if seen[g] >= k:
            return True
    return False


def sample_rows(root, per_file, files, seed):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    out = []
    for sub in sorted(p.name for p in Path(root).iterdir() if p.is_dir()):
        for task in ("asr_only", "ast"):
            paths = sorted((Path(root) / sub / task).glob("*.parquet"))
            for p in rng.sample(paths, min(files, len(paths))):
                f = pq.ParquetFile(p)
                g = rng.randrange(f.metadata.num_row_groups)
                t = f.read_row_group(g, columns=["utt_id", "audio", "duration", "text"])
                idx = rng.sample(range(t.num_rows), min(per_file, t.num_rows))
                for i in idx:
                    out.append(dict(subset=sub, task=task, file=p.name, utt_id=t.column("utt_id")[i].as_py(),
                                    duration=t.column("duration")[i].as_py(), ref=t.column("text")[i].as_py(),
                                    audio=t.column("audio")[i].as_py()["bytes"]))
            print(f"sampled {sub}/{task}: {sum(1 for r in out if r['subset'] == sub and r['task'] == task)}", flush=True)
    return out


def quantiles(xs):
    xs = sorted(xs)
    q = lambda p: round(xs[min(len(xs) - 1, int(p * len(xs)))], 4)
    return {"n": len(xs), "mean": round(st.fmean(xs), 4), "p50": q(0.5), "p75": q(0.75), "p90": q(0.9),
            **{f"le_{int(t * 100)}pct": round(sum(x <= t for x in xs) / len(xs), 4) for t in (0.0, 0.03, 0.05, 0.1)},
            "gt_30pct": round(sum(x > 0.3 for x in xs) / len(xs), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", type=int, default=10, help="parquet files per stratum")
    ap.add_argument("--per-file", type=int, default=25, help="rows per file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="/soundai/Model/Qwen3-ASR-1.7B")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    rows = sample_rows(a.root, a.per_file, a.files, a.seed)
    print(f"sampled {len(rows)} rows in {time.monotonic() - t0:.0f} s", flush=True)

    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=a.batch, max_new_tokens=448)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
    t1, audio_s = time.monotonic(), 0.0
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        wavs = []
        for i in idx:
            x, sr = sf.read(io.BytesIO(rows[i]["audio"]), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            wavs.append((x, sr))
            audio_s += len(x) / sr
        with torch.inference_mode():
            res = model.transcribe(audio=wavs, language=["German"] * len(idx))
        for i, r in zip(idx, res):
            rows[i]["hyp"] = r.text
    infer_s = time.monotonic() - t1
    print(f"transcribed {len(rows)} rows ({audio_s / 3600:.2f} h audio) in {infer_s:.0f} s", flush=True)

    by = defaultdict(list)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            rn, hn = norm_de(r["ref"]), norm_de(r["hyp"])
            wer = edit_ops(rn, hn) / max(1, len(rn))
            flags = [k for k, ok in (("ref_hallucination_phrase", bool(HALLUCINATION.search(r["ref"] or ""))),
                                      ("ref_repeated_ngram", repeated_ngram(rn)),
                                      ("ref_empty", not rn), ("hyp_empty", not hn),
                                      ("over_30s", r["duration"] > 30.0)) if ok]
            rec = {k: r[k] for k in ("subset", "task", "file", "utt_id", "duration", "ref", "hyp")}
            rec.update(ref_norm=" ".join(rn), hyp_norm=" ".join(hn), ref_words=len(rn), wer=round(wer, 4), flags=flags)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            by[(r["subset"], r["task"])].append(rec)
            by[("all", r["task"])].append(rec)
            by[("all", "all")].append(rec)
    summary = {"model": a.model, "seed": a.seed, "rows": len(rows), "audio_hours": round(audio_s / 3600, 3),
               "infer_seconds": round(infer_s, 1), "rtf": round(infer_s / max(audio_s, 1e-9), 5), "strata": {}}
    for (sub, task), recs in sorted(by.items()):
        q = quantiles([x["wer"] for x in recs])
        q["flags"] = {fl: sum(fl in x["flags"] for x in recs) for fl in
                      ("ref_hallucination_phrase", "ref_repeated_ngram", "ref_empty", "hyp_empty", "over_30s")}
        short = [x["wer"] for x in recs if x["ref_words"] < 10]
        q["short_lt10_words"] = quantiles(short) if short else None
        summary["strata"][f"{sub}/{task}"] = q
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    bad = sorted((x for x in by[("all", "all")] if x["wer"] > 0.03), key=lambda x: -x["wer"])
    with open(out / "review.tsv", "w", encoding="utf-8") as f:
        f.write("wer\tduration\tsubset/task\tutt_id\tflags\tgranary\tqwen\n")
        for x in bad:
            f.write(f"{x['wer']}\t{x['duration']}\t{x['subset']}/{x['task']}\t{x['utt_id']}\t{','.join(x['flags'])}\t"
                    f"{x['ref']}\t{x['hyp']}\n")
    print(json.dumps(summary["strata"].get("all/all"), ensure_ascii=False), flush=True)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
