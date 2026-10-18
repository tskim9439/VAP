#!/usr/bin/env python3
"""Audit speechlm QC candidates against protected audio and source policy.

This prepares evidence for human approval.  It never flips training_eligible.
An exact PCM/path comparison is not an acoustic near-duplicate audit.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path


# Existing training lineages, held-out FLEURS, and evaluation-only Earnings22.
# YODAS requires a sublanguage path check; hold all of 000091 pending that audit.
DENIED_CATALOGS = frozenset({
    "000019", "000020", "000032", "000080", "000085", "000092",
    "000062", "000079", "000087", "000091",
})


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def catalog(source):
    return source.rsplit("-", 1)[-1]


def _sample_add(heaps, cell, row, limit=20):
    rank = int(hashlib.sha256(f"{cell}:{row['key']}".encode()).hexdigest(), 16)
    heap = heaps[cell]
    item = (-rank, row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def audit(qc, heldout, out, max_rows=0):
    if out.exists():
        raise FileExistsError(out)
    qc_summary_path = qc / "summary.json"
    held_summary_path = heldout / "summary.json"
    q = json.loads(qc_summary_path.read_text())
    h = json.loads(held_summary_path.read_text())
    if not q.get("complete") or q.get("training_eligible") is not False:
        raise ValueError("QC is not a complete candidate-only snapshot")
    if not h.get("complete") or h.get("max_items") != 0:
        raise ValueError("held-out index is incomplete or smoke-only")
    if sha256(qc / "decisions.jsonl") != q["decisions_sha256"]:
        raise ValueError("QC decisions checksum mismatch")
    if sha256(heldout / "identities.jsonl") != h["identities_sha256"]:
        raise ValueError("held-out index checksum mismatch")
    waves, paths = set(), set()
    for r in rows(heldout / "identities.jsonl"):
        paths.add(r["path"])
        if r["hash_status"] == "ok":
            waves.add(r["waveform_sha256"])
        elif r["hash_status"] != "path_only":
            raise ValueError(f"held-out index has failed item: {r['path']}")
    out.mkdir(parents=True)
    states, preliminary, by_source, hours = Counter(), Counter(), defaultdict(Counter), defaultdict(float)
    samples = defaultdict(list)
    with (out / "heldout-collisions.jsonl").open("x", encoding="utf-8") as collisions:
        for i, r in enumerate(rows(qc / "decisions.jsonl"), 1):
            state = r["state"]
            states[state] += 1
            source = r["source"]
            by_source[source][state] += 1
            if state == "QC_PASS_CANDIDATE":
                duration = float(r["duration_s"])
                if not 0 < duration <= 30:
                    raise ValueError(f"invalid candidate duration: {r['key']}")
                pool = "short" if duration < 8 else "long"
                if catalog(source) in DENIED_CATALOGS:
                    preliminary["HOLD_SOURCE_POLICY"] += 1
                elif r["waveform_sha256"] in waves or r["audio_locator"] in paths:
                    preliminary["HOLD_HELDOUT_EXACT"] += 1
                    collisions.write(json.dumps(r, ensure_ascii=False) + "\n")
                else:
                    preliminary[f"CANDIDATE_{pool.upper()}"] += 1
                    by_source[source][f"candidate_{pool}"] += 1
                    hours[source] += duration / 3600
                    _sample_add(samples, (source, pool), r)
            elif state == "REVIEW_ZERO_DURATION":
                preliminary["HOLD_ZERO_DURATION"] += 1
                _sample_add(samples, (source, "zero_duration"), r)
            if i % 500000 == 0:
                print(f"APPROVAL_AUDIT {i} rows candidate={preliminary['CANDIDATE_LONG']+preliminary['CANDIDATE_SHORT']}", flush=True)
            if max_rows and i >= max_rows:
                break
    if not max_rows and dict(states) != q["counts"]:
        raise ValueError("QC decision counts do not match its summary")
    with (out / "review-sample.jsonl").open("x", encoding="utf-8") as handle:
        for cell in sorted(samples):
            for _, row in sorted(samples[cell], reverse=True):
                handle.write(json.dumps(dict(row, review_cell=list(cell)), ensure_ascii=False) + "\n")
    report = dict(schema="speechlm-semcommit-approval-audit-v1", complete=not max_rows,
                  training_eligible=False, approval_status="pending_human_review",
                  qc_summary_sha256=sha256(qc_summary_path), heldout_summary_sha256=sha256(held_summary_path),
                  heldout_index_scope=dict(pcm=h["pcm_scope"], path_only=h["path_only_scope"]),
                  protected_pcm_hashes=len(waves), protected_paths=len(paths),
                  qc_states=dict(states), preliminary=dict(preliminary),
                  by_source={source: dict(counts) for source, counts in by_source.items()},
                  candidate_hours_by_source={source: round(value, 3) for source, value in hours.items()},
                  review_sample_sha256=sha256(out / "review-sample.jsonl"),
                  heldout_collisions_sha256=sha256(out / "heldout-collisions.jsonl"),
                  max_rows=max_rows,
                  remaining_gates=["text_and_acoustic_near_duplicate_review", "acoustic_sample_review",
                                   "source_balance_policy", "human_approval"])
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--heldout-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="smoke only; 0 means all")
    args = parser.parse_args()
    if args.max_rows < 0:
        parser.error("max-rows must be nonnegative")
    audit(args.qc, args.heldout_index, args.out, args.max_rows)


if __name__ == "__main__":
    main()
