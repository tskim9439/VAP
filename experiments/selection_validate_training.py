#!/usr/bin/env python3
"""Validate an approved mono view and check exact audio leakage to held-out manifests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest


def rows(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def locator(seg):
    return seg["path"], seg.get("src_offset_s"), round(float(seg["dur_s"]), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--heldout-root", type=Path, required=True)
    ap.add_argument("--heldout", default="librispeech-dev,librispeech-test,kspon-dev,kspon-eval")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.view / "summary.json").read_text())
    if not summary.get("complete") or not summary.get("training_eligible"):
        raise ValueError("training view is not complete and eligible")
    held_paths, held_locators = set(), set()
    held_counts = Counter()
    for name in filter(None, a.heldout.split(",")):
        path = a.heldout_root / name / "streams.jsonl"
        if not path.exists(): raise FileNotFoundError(path)
        for row in rows(path):
            for seg in row["segments"]:
                held_paths.add(seg["path"]); held_locators.add(locator(seg)); held_counts[name] += 1
    ids, keys = set(), set(); counts = Counter(); errors = []; path_overlap = set(); locator_overlap = set()
    for name, expected in summary["datasets"].items():
        manifest = a.view / "manifests" / name / "streams.jsonl"
        for row in rows(manifest):
            counts[name] += 1
            if row["id"] in ids and len(errors) < 20: errors.append("duplicate_stream:" + row["id"])
            ids.add(row["id"])
            if row.get("split") != "train" and len(errors) < 20: errors.append("non_train_split:" + row["id"])
            for seg in row["segments"]:
                key = seg.get("source_key")
                if not key or key in keys:
                    if len(errors) < 20: errors.append("missing_or_duplicate_source_key:" + str(key))
                keys.add(key)
                if not seg.get("lexical_text") and len(errors) < 20: errors.append("empty_target:" + str(key))
                if seg["path"] in held_paths: path_overlap.add(seg["path"])
                loc = locator(seg)
                if loc in held_locators: locator_overlap.add(loc)
        if counts[name] != expected["streams"]: errors.append(f"stream_count:{name}:{counts[name]}!={expected['streams']}")
        if file_digest(manifest) != expected["manifest_sha256"]: errors.append("manifest_sha256:" + name)
    if len(keys) != summary["trainable_rows"]: errors.append(f"trainable_rows:{len(keys)}!={summary['trainable_rows']}")
    if path_overlap: errors.append(f"heldout_path_overlap:{len(path_overlap)}")
    if locator_overlap: errors.append(f"heldout_locator_overlap:{len(locator_overlap)}")
    result = dict(complete=not errors, training_eligible=not errors,
                  view_fingerprint=summary["fingerprint"], approval_fingerprint=summary["approval_fingerprint"],
                  streams=dict(counts), source_keys=len(keys), heldout_segments=dict(held_counts),
                  heldout_path_overlap=len(path_overlap), heldout_locator_overlap=len(locator_overlap), errors=errors)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
