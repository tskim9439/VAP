#!/usr/bin/env python3
"""Finish one labeled speechlm part: split words/labels into the training pools and write DONE.json (the completion marker).

Pools follow semcommit_train.py: main = 8–30 s streams (--train-words/--train-labels), short = < 8 s (--short-train-words/-labels).
Only streams with a labels row are written, so every pool pair is exactly aligned. DONE.json is written last (open 'x'); a
part without it is not finished and semcommit_collect_done.py ignores it.

  python experiments/semcommit_part_finalize.py --part part-000123 --words W --labels L --stats S --pool-dir <attempt dir> --done <part dir>/DONE.json
"""
import argparse
import hashlib
import json
from pathlib import Path

SHORT_S = 8.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize(part, words, labels, stats, pool_dir, done, extra=None):
    pool_dir, done = Path(pool_dir), Path(done)
    if done.exists():
        raise FileExistsError(done)
    pool_dir.mkdir(parents=True, exist_ok=False)
    lab = {}
    if labels and Path(labels).exists():
        for line in open(labels, encoding="utf-8"):
            if line.strip():
                r = json.loads(line); lab[r["id"]] = line if line.endswith("\n") else line + "\n"
    pools, counts = {"main": ([], []), "short": ([], [])}, dict(words=0, labeled=0, main=0, short=0, over_30s=0)
    for line in open(words, encoding="utf-8"):
        if not line.strip():
            continue
        w = json.loads(line); counts["words"] += 1
        if w["id"] not in lab:
            continue
        if w["duration_s"] > 30:
            counts["over_30s"] += 1; continue
        pool = "short" if w["duration_s"] < SHORT_S else "main"
        pools[pool][0].append(line if line.endswith("\n") else line + "\n"); pools[pool][1].append(lab[w["id"]])
        counts["labeled"] += 1; counts[pool] += 1
    files = {}
    for pool, (ws, ls) in pools.items():
        if not ws:
            continue
        wp, lp = pool_dir / f"words-{pool}.jsonl", pool_dir / f"labels-{pool}.jsonl"
        wp.open("x", encoding="utf-8").writelines(ws); lp.open("x", encoding="utf-8").writelines(ls)
        files[pool] = dict(words=str(wp), labels=str(lp), words_sha256=sha256(wp), labels_sha256=sha256(lp), streams=len(ws))
    st = json.loads(Path(stats).read_text()) if stats and Path(stats).exists() else {}
    grades = {k: v for lang in (st.get("by_lang") or {}).values() for k, v in lang.items() if k in ("A", "B", "N")}
    rec = dict(schema="speechlm-semcommit-part-done-v1", part=part, counts=counts, files=files, grades=grades,
               recipe=st.get("recipe"), thresholds=st.get("thresholds"), neg_rules=st.get("neg_rules"), mask_rules=st.get("mask_rules"),
               source_words=str(words), source_words_sha256=sha256(words), **(extra or {}))
    done.open("x").write(json.dumps(rec, ensure_ascii=False))
    print(json.dumps(dict(part=part, counts=counts)))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", required=True); p.add_argument("--words", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None); p.add_argument("--stats", type=Path, default=None)
    p.add_argument("--pool-dir", type=Path, required=True, help="new directory for words-/labels-{main,short}.jsonl")
    p.add_argument("--done", type=Path, required=True, help="completion marker path (<part dir>/DONE.json)")
    p.add_argument("--meta", default="{}", help="extra JSON fields for DONE.json (job id, gate dir …)")
    a = p.parse_args()
    finalize(a.part, a.words, a.labels, a.stats, a.pool_dir, a.done, json.loads(a.meta))


if __name__ == "__main__":
    main()
