#!/usr/bin/env python3
"""Full metadata census and deterministic train-only stratified pilot.

Creates a NEW output directory; never edits upstream manifests. Audio decoding
and teacher inference are separate. A metadata pass is NOT a quality pass.
"""
import argparse
from collections import Counter
import gzip
import heapq
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import (VERSION, file_digest, digest, stream_rows,
                                   dialogue_rows, stratum, audio_identity)

P1 = ["librispeech-960", "kspon-full", "swbd-train", "mnsc-1000", "nikl-1000",
      "voxpopuli-train", "yodas-en129", "aihub-bc-train"]
P2 = ["aihub71631", "aihub134-1", "aihub134-2", "otoSpeech", "ami", "notsofar", "icsi"]


def jsonline(f, row):
    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def census(root, out, per_stratum=16):
    if per_stratum < 1:
        raise ValueError("per-stratum must be positive")
    split_path = root / "phase2/splits/v1.json"
    splits = json.loads(split_path.read_text())["corpora"]  # fail closed
    heldout = {x for v in splits.values() for x in v["heldout"]}
    # 71631 and adult crops can be two recordings/exports of the same session.
    aliases = {x.split(":", 1)[1] for x in heldout
               if x.startswith(("71631:", "aihub134-1:"))}
    stereo_stems = set()
    stereo_file = root / "phase2/aihub71631.dialogues.jsonl"
    with stereo_file.open() as f:
        for line in f:
            stereo_stems.add(json.loads(line)["conv_id"].split(":", 1)[1])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, split_sha256=file_digest(split_path),
                  sources={}, excluded_catalog=["aihub71631-train (use Phase 2 original)",
                    "librispeech-100 / kspon-100 (subsets)",
                    "all dev/test/eval, nikl2020, chime6, turnbench"],
                  scope="existing unrefined manifests, not entire original DB",
                  training_eligible=0, per_stratum=per_stratum)
    with (out / "pilot.jsonl").open("w") as pilot:
        for name, kind in [(n, "stream") for n in P1] + [(n, "dialogue") for n in P2]:
            path = root / (f"manifests/{name}/streams.jsonl" if kind == "stream"
                           else f"phase2/{name}.dialogues.jsonl")
            if not path.exists():
                report["sources"][name] = {"status": "missing_source", "path": str(path)}
                continue
            if kind == "dialogue" and name not in splits:
                raise ValueError(f"No reviewed split for {name}")
            src_hash = file_digest(path)
            stats, why, buckets, hours = Counter(), Counter(), {}, Counter()
            seen_keys, seen_audio = set(), set()
            with path.open() as f, gzip.open(out / f"{name}.candidates.jsonl.gz", "wt", encoding="utf-8") as dest:
                for line in f:
                    d = json.loads(line)
                    rows = stream_rows(d, name) if kind == "stream" else dialogue_rows(d, name)
                    for r in rows:
                        r["manifest_sha256"] = src_hash
                        duration = r["duration_s"]
                        if not math.isfinite(duration) or duration <= 0:
                            r["reasons"].append("invalid_duration")
                            r["duration_s"] = None
                        if not r["text"].strip():
                            r["reasons"].append("empty_target")
                        if r["source_flags"]:
                            r["reasons"].append("upstream_flags")
                        parent = r["parent_id"]
                        excluded = r["split"] != "train" or parent in heldout
                        if kind == "dialogue" and name in ("aihub71631", "aihub134-1"):
                            excluded |= parent.split(":", 1)[1] in aliases
                        duplicate_session = (name == "aihub134-1" and parent.split(":", 1)[1] in stereo_stems)
                        duplicate = r["key"] in seen_keys or (r["audio"] is not None and audio_identity(r) in seen_audio)
                        seen_keys.add(r["key"])
                        if r["audio"] is not None:
                            seen_audio.add(audio_identity(r))
                        if excluded:
                            state = "HELDOUT"
                        elif duplicate or duplicate_session:
                            state = "DUPLICATE_LOCATOR_OR_SESSION"
                        elif r["reasons"]:
                            state = "QUARANTINE_METADATA"
                        else:
                            state = "PENDING_AUDIO_TEACHERS"
                        r["selection_state"] = state
                        stats[state] += 1
                        if math.isfinite(duration) and duration > 0:
                            hours[state] += duration / 3600
                        why.update(r["reasons"])
                        if state == "PENDING_AUDIO_TEACHERS":
                            bucket = stratum(r)
                            stats["stratum_" + bucket] += 1
                            r["stratum"] = bucket
                            # bottom-k hash reservoir; independent of file ordering
                            h = -int(digest(["selection-seed-20260918", r["key"]])[:16], 16)
                            heap = buckets.setdefault(bucket, [])
                            item = (h, r["key"], r)
                            if len(heap) < per_stratum:
                                heapq.heappush(heap, item)
                            elif item[:2] > heap[0][:2]:
                                heapq.heapreplace(heap, item)
                        jsonline(dest, r)
            samples = [t[2] for heap in buckets.values() for t in sorted(heap, reverse=True)]
            for r in samples:
                r["sample_stratum_population"] = stats["stratum_" + r["stratum"]]
                r["sample_stratum_n"] = len(buckets[r["stratum"]])
                jsonline(pilot, r)
            pilot.flush()
            report["sources"][name] = dict(path=str(path), sha256=src_hash, counts=dict(stats),
                hours={k: round(v, 3) for k, v in hours.items()}, reasons=dict(why), pilot_n=len(samples))
            print(json.dumps({"source": name, **report["sources"][name]}, ensure_ascii=False), flush=True)
            with (out / "census.json").open("w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
    report["complete"] = all(s.get("status") != "missing_source" for s in report["sources"].values())
    report["pilot_sha256"] = file_digest(out / "pilot.jsonl")
    report["implementation_sha256"] = file_digest(Path(__file__).resolve())
    with (out / "census.json").open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-stratum", type=int, default=16)
    a = p.parse_args()
    census(a.data_root, a.out, a.per_stratum)
