#!/usr/bin/env python3
"""YODAS-Granary German preprocessing, CPU stages (stage 0 of the German ASR data pipeline).

  inventory  read every parquet's metadata columns (no audio) into one parquet per subset/task
             -> <out>/inventory/<subset>.<task>.parquet
             columns: subset, task, file, row_group, row, utt_id, duration, text, video_id, offset_s
  prefilter  flag rows before any GPU work and choose the held-out dev videos -> <out>/prefilter/
             prefilter.parquet  (inventory key + flags + split), summary.json
  python experiments/yodas_de_prepare.py inventory --root <yodas_granary/data> --out <dir> [--workers 16]
  python experiments/yodas_de_prepare.py prefilter --out <dir> --decon <eval text files ...>

Flags (a row is a candidate only if it has none):
  dur_short / dur_long       duration < --min-dur (1.0 s) or > --max-dur (30.0 s)
  text_empty                 no words after normalisation
  hallucination_phrase       phrases Whisper hallucinates on German YouTube audio (credits, "Danke fürs Zuschauen")
  repeated_ngram             a word 3-gram occurs 3 or more times
  rate_outlier               characters per second outside [--min-cps, --max-cps]
  non_latin                  more than 10 % of letters outside Latin script
  duplicate                  same video and an overlapping time span as an earlier row (asr_only and ast together),
                             or the same normalised text and duration (rounded to 10 ms) as an earlier row
  eval_overlap               shares a word 8-gram with an evaluation text (decision-eval-train-decontamination)
Split: rows of --dev-videos videos (chosen by a stable hash of the video id) go to `dev`; the rest to `train`.
Decisions: wiki/outputs/output-yodas-granary-de-transcript-check-20261008.md (Qwen-only selection at WER <= 3 %,
Qwen text as target) and wiki/decisions/decision-eval-train-decontamination.md.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yodas_de_qwen_check import HALLUCINATION, norm_de, repeated_ngram  # noqa: E402

COLS = ["subset", "task", "file", "row_group", "row", "utt_id", "duration", "text", "video_id", "offset_s"]


def _scan(path):
    import pyarrow.parquet as pq
    p = Path(path)
    f = pq.ParquetFile(p)
    out = {c: [] for c in COLS}
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["utt_id", "duration", "text", "original_audio_id", "original_audio_offset"])
        n = t.num_rows
        out["subset"] += [p.parent.parent.name] * n
        out["task"] += [p.parent.name] * n
        out["file"] += [p.name] * n
        out["row_group"] += [g] * n
        out["row"] += list(range(n))
        out["utt_id"] += t.column("utt_id").to_pylist()
        out["duration"] += t.column("duration").to_pylist()
        out["text"] += t.column("text").to_pylist()
        out["video_id"] += t.column("original_audio_id").to_pylist()
        out["offset_s"] += t.column("original_audio_offset").to_pylist()
    return out


def cmd_inventory(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root, out = Path(a.root), Path(a.out) / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    for sub in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for task in ("asr_only", "ast"):
            dst = out / f"{sub}.{task}.parquet"
            if dst.exists():
                print("exists", dst, flush=True)
                continue
            paths = sorted(str(p) for p in (root / sub / task).glob("*.parquet"))
            t0 = time.monotonic()
            merged = {c: [] for c in COLS}
            with ProcessPoolExecutor(a.workers) as ex:
                for part in ex.map(_scan, paths, chunksize=4):
                    for c in COLS:
                        merged[c] += part[c]
            tmp = dst.with_suffix(".tmp")
            pq.write_table(pa.table(merged), tmp, compression="zstd")
            tmp.replace(dst)
            print(f"{sub}/{task}: {len(paths)} files, {len(merged['utt_id'])} rows, {time.monotonic() - t0:.0f} s",
                  flush=True)


def _ngrams(words, n=8):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _latin_share(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters)


def _video_hash(video_id):
    """Platform-independent order of videos for the held-out dev choice."""
    return int(hashlib.sha1(video_id.encode("utf-8")).hexdigest()[:8], 16)


def cmd_prefilter(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    inv = sorted((Path(a.out) / "inventory").glob("*.parquet"))
    t = pa.concat_tables([pq.read_table(p) for p in inv])
    rows = t.to_pydict()
    n = t.num_rows
    print(f"inventory rows {n}", flush=True)

    eval_grams = set()
    for path in a.decon or []:
        for line in open(path, encoding="utf-8"):
            eval_grams |= _ngrams(norm_de(line), 8)
    print(f"evaluation 8-grams {len(eval_grams)} from {len(a.decon or [])} files", flush=True)

    videos = sorted(set(rows["video_id"]))
    ranked = sorted(videos, key=_video_hash)
    dev_videos = set(ranked[:a.dev_videos])

    order = sorted(range(n), key=lambda i: (rows["video_id"][i], rows["offset_s"][i], rows["task"][i] != "ast"))
    flags = [[] for _ in range(n)]
    last_end = {}
    seen_text = set()
    for i in order:
        d, text = rows["duration"][i], rows["text"][i] or ""
        f = flags[i]
        if d < a.min_dur:
            f.append("dur_short")
        if d > a.max_dur:
            f.append("dur_long")
        w = norm_de(text)
        if not w:
            f.append("text_empty")
        if HALLUCINATION.search(text):
            f.append("hallucination_phrase")
        if repeated_ngram(w):
            f.append("repeated_ngram")
        cps = len("".join(w)) / max(d, 1e-3)
        if w and not (a.min_cps <= cps <= a.max_cps):
            f.append("rate_outlier")
        if _latin_share(text) < 0.9:
            f.append("non_latin")
        v, start = rows["video_id"][i], rows["offset_s"][i]
        if start < last_end.get(v, -1.0) - 0.05:
            f.append("duplicate")
        last_end[v] = max(last_end.get(v, -1.0), start + d)
        key = (" ".join(w), round(d, 2))
        if w and key in seen_text:
            if "duplicate" not in f:
                f.append("duplicate")
        seen_text.add(key)
        if eval_grams and _ngrams(w, 8) & eval_grams:
            f.append("eval_overlap")

    split = ["dev" if rows["video_id"][i] in dev_videos else "train" for i in range(n)]
    out = Path(a.out) / "prefilter"
    out.mkdir(parents=True, exist_ok=True)
    tab = pa.table({"subset": rows["subset"], "task": rows["task"], "file": rows["file"], "row_group": rows["row_group"],
                    "row": rows["row"], "utt_id": rows["utt_id"], "duration": rows["duration"],
                    "video_id": rows["video_id"], "flags": [",".join(f) for f in flags], "split": split})
    tmp = out / "prefilter.parquet.tmp"
    pq.write_table(tab, tmp, compression="zstd")
    tmp.replace(out / "prefilter.parquet")

    fc, fh = Counter(), Counter()
    cand = {"train": [0, 0.0], "dev": [0, 0.0]}
    for i in range(n):
        for x in flags[i]:
            fc[x] += 1
            fh[x] += rows["duration"][i] / 3600
        if not flags[i]:
            cand[split[i]][0] += 1
            cand[split[i]][1] += rows["duration"][i] / 3600
    summary = {"rows": n, "hours": round(sum(rows["duration"]) / 3600, 2), "videos": len(videos),
               "dev_videos": len(dev_videos), "params": {k: getattr(a, k) for k in
                                                          ("min_dur", "max_dur", "min_cps", "max_cps", "dev_videos")},
               "decon_files": a.decon or [], "eval_8grams": len(eval_grams),
               "flag_rows": dict(fc.most_common()), "flag_hours": {k: round(v, 2) for k, v in fh.most_common()},
               "candidates": {k: {"rows": v[0], "hours": round(v[1], 2)} for k, v in cand.items()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("inventory")
    q.add_argument("--root", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=16)
    q.set_defaults(fn=cmd_inventory)
    q = sub.add_parser("prefilter")
    q.add_argument("--out", required=True)
    q.add_argument("--decon", nargs="*", help="evaluation text files, one sentence per line")
    q.add_argument("--min-dur", type=float, default=1.0)
    q.add_argument("--max-dur", type=float, default=30.0)
    q.add_argument("--min-cps", type=float, default=4.0)
    q.add_argument("--max-cps", type=float, default=30.0)
    q.add_argument("--dev-videos", type=int, default=200)
    q.set_defaults(fn=cmd_prefilter)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
