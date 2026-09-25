#!/usr/bin/env python3
"""Snapshot of finished speechlm semcommit parts → training input lists (usable while labeling is still running).

Reads <labels-root>/parts/*/DONE.json (only parts with the marker count), optionally re-verifies file digests, and writes list
files for semcommit_train.py's @file arguments:
  main-words.list / main-labels.list    →  --train-words @main-words.list --train-labels @main-labels.list
  short-words.list / short-labels.list  →  --short-train-words @short-words.list --short-train-labels @short-labels.list
plus summary.json (parts, streams, grades per source DB). Never overwrites: --out-dir must not exist.

  python experiments/semcommit_collect_done.py --labels-root <OUT> --out-dir <OUT>/snapshots/$(date +%Y%m%d-%H%M) [--verify]
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(labels_root, out_dir, verify=False, order=None):
    root, out_dir = Path(labels_root) / "parts", Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    source = {}
    if order:
        for line in Path(order).read_text().splitlines():
            if line.strip():
                p, s = line.split("\t")[:2]; source[p] = s
    lists, per_source, bad = defaultdict(list), defaultdict(Counter), []
    for part in sorted(os.listdir(root)):
        done = root / part / "DONE.json"
        if not done.exists():
            continue
        d = json.loads(done.read_text())
        for pool, f in d["files"].items():
            if verify and (sha256(f["words"]) != f["words_sha256"] or sha256(f["labels"]) != f["labels_sha256"]):
                bad.append(part); continue
            lists[f"{pool}-words"].append(f["words"]); lists[f"{pool}-labels"].append(f["labels"])
            per_source[source.get(part, "?")][f"{pool}_streams"] += f["streams"]
        s = per_source[source.get(part, "?")]; s["parts"] += 1
        for k, v in (d.get("grades") or {}).items():
            s[k] += v
    out_dir.mkdir(parents=True)
    for name in ("main-words", "main-labels", "short-words", "short-labels"):
        (out_dir / f"{name}.list").open("x").write("".join(p + "\n" for p in lists[name]))
    summary = dict(schema="speechlm-semcommit-done-snapshot-v1", labels_root=str(labels_root), verified=verify, bad_parts=bad,
                   parts=sum(s["parts"] for s in per_source.values()), main_files=len(lists["main-words"]), short_files=len(lists["short-words"]),
                   per_source={k: dict(v) for k, v in sorted(per_source.items())})
    (out_dir / "summary.json").open("x").write(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ("parts", "main_files", "short_files", "bad_parts")}))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--order", type=Path, default=None, help="parts-order.tsv (part → source DB for the summary)")
    p.add_argument("--verify", action="store_true", help="re-hash every listed file")
    a = p.parse_args()
    collect(a.labels_root, a.out_dir, a.verify, a.order)


if __name__ == "__main__":
    main()
