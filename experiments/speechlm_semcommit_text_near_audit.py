#!/usr/bin/env python3
"""Screen all aligned speechlm text for held-out textual near duplicates.

This is a conservative *review screen*, never proof of acoustic identity.  A
shared transcript and similar duration is flagged, not silently removed.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SETS = ("librispeech-dev", "librispeech-test", "kspon-dev", "kspon-eval", "aihub71631-dev")
_INDEX = None


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalized(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def anchors(value):
    if len(value) < 20:
        return ()
    width = 12
    middle = max(0, (len(value) - width) // 2)
    return tuple(dict.fromkeys((value[:width], value[middle:middle + width], value[-width:])))


def heldout_text_index(manifests, gold):
    entries = {}
    for name in SETS:
        path = manifests / name / "streams.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for stream in rows(path):
            lang = stream["lang"]
            for seg in stream["segments"]:
                value = normalized(seg.get("lexical_text") or seg.get("text") or "")
                if not anchors(value):
                    continue
                key = (lang, value, round(float(seg["dur_s"]), 1))
                entries.setdefault(key, name)
    gold_path = gold / "words-gs-test.jsonl"
    for stream in rows(gold_path):
        for seg in stream["segments"]:
            value = normalized(stream.get("text") or seg.get("raw_text") or "")
            if not anchors(value):
                continue
            key = (stream["lang"], value, round(float(seg["dur_s"]), 1))
            entries.setdefault(key, "gold-gs-test")
    index = defaultdict(list)
    for (lang, value, duration), name in entries.items():
        for anchor in anchors(value):
            index[(lang, anchor)].append((value, duration, name))
    # Very common anchors are useless for near-duplicate screening and can
    # otherwise make a handful of generic phrases dominate runtime.
    return {key: value for key, value in index.items() if len(value) <= 50}, len(entries)


def match(text, lang, duration, index):
    value = normalized(text)
    if not anchors(value):
        return None
    options = {}
    for anchor in anchors(value):
        for ref in index.get((lang, anchor), ()):
            options[(ref[0], ref[1], ref[2])] = ref
    best = None
    for ref_text, ref_duration, ref_set in options.values():
        if abs(float(duration) - ref_duration) > max(0.75, 0.25 * max(float(duration), ref_duration)):
            continue
        score = 1.0 if value == ref_text else SequenceMatcher(None, value, ref_text, autojunk=False).ratio()
        if score >= 0.9 and (best is None or score > best["text_similarity"]):
            best = dict(heldout_set=ref_set, text_similarity=round(score, 4),
                        candidate_norm=value, reference_norm=ref_text,
                        reference_duration_s=ref_duration)
    return best


def _set_index(index):
    global _INDEX
    _INDEX = index


def _scan_part(job):
    root, item = job
    path = Path(root) / item["part"] / "aligned.jsonl"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["outputs"]["aligned.jsonl"]:
        raise ValueError(f"alignment source changed: {path}")
    hits = []
    count = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        count += 1
        found = match(row["target_text"], row["lang"], row["duration_s"], _INDEX)
        if found:
            hits.append(dict(key=row["key"], source=row["source"], source_shard=item["part"],
                             duration_s=row["duration_s"], **found))
    if count != item["aligned"]:
        raise ValueError(f"aligned row count changed: {path}")
    return count, hits


def build(qc, manifests, gold, out, workers=8, max_parts=0):
    if out.exists():
        raise FileExistsError(out)
    q = json.loads((qc / "summary.json").read_text())
    if not q.get("complete") or sha256(qc / "snapshot.json") != q["snapshot_sha256"]:
        raise ValueError("QC is incomplete or snapshot changed")
    snapshot = json.loads((qc / "snapshot.json").read_text())
    parts = snapshot["parts"][:max_parts or None]
    index, protected_texts = heldout_text_index(manifests, gold)
    out.mkdir(parents=True)
    counts = Counter()
    with (out / "near-candidates.jsonl").open("x", encoding="utf-8") as handle, \
         ProcessPoolExecutor(max_workers=workers, initializer=_set_index, initargs=(index,)) as pool:
        jobs = [(snapshot["results"], item) for item in parts]
        for i, (read_count, hits) in enumerate(pool.map(_scan_part, jobs, chunksize=1), 1):
            counts["aligned_rows"] += read_count
            counts["near_hits"] += len(hits)
            for hit in hits:
                counts[f"source:{hit['source']}"] += 1
                counts[f"heldout:{hit['heldout_set']}"] += 1
                handle.write(json.dumps(hit, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"TEXT_NEAR {i}/{len(parts)} aligned={counts['aligned_rows']} hits={counts['near_hits']}", flush=True)
    report = dict(schema="speechlm-semcommit-text-near-audit-v1", complete=not max_parts,
                  training_eligible=False, qc_summary_sha256=sha256(qc / "summary.json"),
                  protected_texts=protected_texts, anchor_keys=len(index), parts=len(parts), counts=dict(counts),
                  near_candidates_sha256=sha256(out / "near-candidates.jsonl"),
                  interpretation="transcript/duration review screen; not acoustic near-duplicate proof")
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-parts", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.max_parts < 0:
        parser.error("workers must be 1..32 and max-parts nonnegative")
    build(args.qc, args.manifest_root, args.gold_root, args.out, args.workers, args.max_parts)


if __name__ == "__main__":
    main()
