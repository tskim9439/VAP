#!/usr/bin/env python3
"""Build a read-only held-out audio identity index for speechlm approval.

The protected manifests are authoritative.  Hashing uses the same deterministic
16 kHz float32 decoder as the speechlm ASR selection pipeline.  A path-only
set is reported explicitly and never presented as a completed PCM audit.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection_audio import decode


DEFAULT_HASH_SETS = ("librispeech-dev", "librispeech-test", "kspon-dev", "kspon-eval")
DEFAULT_PATH_SETS = ("aihub71631-dev",)


def file_sha256(path):
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


def segment_key(path, offset_s, duration_s):
    return (str(path), round(float(offset_s or 0), 3), round(float(duration_s), 3))


def collect(manifest_root, gold_root, hash_sets, path_sets):
    unique = {}
    source_files = {}
    for name in (*hash_sets, *path_sets):
        path = manifest_root / name / "streams.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files[str(path)] = file_sha256(path)
        for stream in rows(path):
            for seg in stream["segments"]:
                key = segment_key(seg["path"], seg.get("src_offset_s", 0), seg["dur_s"])
                entry = unique.setdefault(key, dict(path=key[0], offset_s=key[1], duration_s=key[2],
                                                    sets=[], hash_required=False))
                if name not in entry["sets"]:
                    entry["sets"].append(name)
                entry["hash_required"] |= name in hash_sets
    gold = gold_root / "words-gs-test.jsonl"
    if not gold.is_file():
        raise FileNotFoundError(gold)
    source_files[str(gold)] = file_sha256(gold)
    for stream in rows(gold):
        for seg in stream["segments"]:
            # Gold words were authored on rack4.  The same WAV files are copied
            # into this immutable mxc bundle; map by basename, not old prefix.
            wav = gold_root / "wav" / Path(seg["path"]).name
            if not wav.is_file():
                raise FileNotFoundError(wav)
            key = segment_key(wav, seg.get("src_offset_s", 0), seg["dur_s"])
            entry = unique.setdefault(key, dict(path=key[0], offset_s=key[1], duration_s=key[2],
                                                sets=[], hash_required=False))
            if "gold-gs-test" not in entry["sets"]:
                entry["sets"].append("gold-gs-test")
            entry["hash_required"] = True
    return [unique[k] for k in sorted(unique)], source_files


def fingerprint(entry):
    if not entry["hash_required"]:
        return dict(entry, waveform_sha256=None, hash_status="path_only")
    try:
        _, info = decode(dict(audio=dict(path=entry["path"], offset_s=entry["offset_s"],
                                         duration_s=entry["duration_s"]), duration_s=entry["duration_s"]))
        return dict(entry, waveform_sha256=info["sha256"], hash_status="ok",
                    decoded_duration_s=info["duration_s"])
    except Exception as exc:
        return dict(entry, waveform_sha256=None, hash_status="error",
                    error=f"{type(exc).__name__}: {str(exc)[:180]}")


def build(manifest_root, gold_root, out, workers=4, hash_sets=DEFAULT_HASH_SETS,
          path_sets=DEFAULT_PATH_SETS, max_items=0):
    if out.exists():
        raise FileExistsError(out)
    entries, source_files = collect(manifest_root, gold_root, hash_sets, path_sets)
    if max_items:
        entries = entries[:max_items]
    out.mkdir(parents=True)
    counts = Counter()
    with (out / "identities.jsonl").open("x", encoding="utf-8") as handle, \
         ProcessPoolExecutor(max_workers=workers) as pool:
        for i, item in enumerate(pool.map(fingerprint, entries, chunksize=16), 1):
            counts[item["hash_status"]] += 1
            for name in item["sets"]:
                counts[f"set:{name}"] += 1
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print(f"HELDOUT_INDEX {i}/{len(entries)} hashed={counts['ok']} errors={counts['error']}", flush=True)
    report = dict(schema="speechlm-semcommit-heldout-index-v1", complete=counts["error"] == 0,
                  pcm_scope=list(hash_sets) + ["gold-gs-test"], path_only_scope=list(path_sets),
                  source_files=source_files, items=len(entries), counts=dict(counts),
                  identities_sha256=file_sha256(out / "identities.jsonl"), max_items=max_items)
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0, help="smoke only; 0 means all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.max_items < 0:
        parser.error("workers must be 1..32 and max-items nonnegative")
    result = build(args.manifest_root, args.gold_root, args.out, args.workers,
                   max_items=args.max_items)
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
