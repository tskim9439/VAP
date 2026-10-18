#!/usr/bin/env python3
"""Build a *candidate-only* speechlm approval proposal.

No training flag is set.  The capped P0 alternative uses deterministic sampling;
the all-pass alternative references the QC decision snapshot without copying
millions of rows.  Both require signed human acoustic review before training.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path


P0 = ("000004", "000005", "000011", "000015", "000016", "000027", "000029")
SEED = "speechlm-semcommit-p0-proposal-v1"


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


def validated_inputs(qc, audit, near):
    q, a, n = (json.loads((root / "summary.json").read_text()) for root in (qc, audit, near))
    qc_digest = sha256(qc / "summary.json")
    if not q.get("complete") or q.get("training_eligible") is not False:
        raise ValueError("QC is not complete candidate-only input")
    if (not a.get("complete") or a.get("qc_summary_sha256") != qc_digest
            or a.get("preliminary", {}).get("HOLD_HELDOUT_EXACT", 0)
            or a.get("preliminary", {}).get("HOLD_SOURCE_POLICY", 0)):
        raise ValueError("held-out audit is incomplete or has unresolved collisions")
    if (not n.get("complete") or n.get("qc_summary_sha256") != qc_digest
            or n.get("counts", {}).get("near_hits", 0)):
        raise ValueError("near-text screen is incomplete or has unresolved hits")
    if sha256(qc / "decisions.jsonl") != q["decisions_sha256"]:
        raise ValueError("QC decisions changed")
    if (a.get("preliminary", {}).get("CANDIDATE_SHORT", 0)
            + a.get("preliminary", {}).get("CANDIDATE_LONG", 0)
            != q.get("counts", {}).get("QC_PASS_CANDIDATE", 0)):
        raise ValueError("QC/audit pass accounting mismatch")
    return q, a, n, qc_digest


def all_pass(qc, audit, near, out):
    """Freeze the user's all-19-source decision without duplicating 8.8M keys."""
    if out.exists():
        raise FileExistsError(out)
    q, a, n, qc_digest = validated_inputs(qc, audit, near)
    sources = sorted(q.get("by_source", {}))
    if len(sources) != 19 or set(sources) != set(a.get("candidate_hours_by_source", {})):
        raise ValueError("expected exactly the 19 audited speechlm sources")
    if any(q["by_source"][source].get("QC_PASS_CANDIDATE", 0) <= 0 for source in sources):
        raise ValueError("source without QC pass candidates")
    policy = dict(schema="speechlm-semcommit-proposal-policy-v1", mode="all_qc_pass",
                  sources=sources, decision_filter="state == QC_PASS_CANDIDATE",
                  no_source_or_duration_cap=True, short_pool_boundary_s=8.0,
                  planned_train_short_step_fraction=0.10,
                  key_materialization="deferred_until_human_review")
    report = dict(schema="speechlm-semcommit-approval-proposal-v1", complete=True,
                  training_eligible=False, approval_status="scope_chosen_human_review_pending",
                  policy=policy, qc_summary_sha256=qc_digest,
                  audit_summary_sha256=sha256(audit / "summary.json"),
                  near_summary_sha256=sha256(near / "summary.json"),
                  decision_file=str(qc / "decisions.jsonl"), decision_file_sha256=q["decisions_sha256"],
                  selected_rows=q["counts"]["QC_PASS_CANDIDATE"],
                  selected_short_rows=a["preliminary"]["CANDIDATE_SHORT"],
                  selected_long_rows=a["preliminary"]["CANDIDATE_LONG"],
                  selected_hours=round(sum(a["candidate_hours_by_source"].values()), 3),
                  selected_hours_by_source=a["candidate_hours_by_source"])
    out.mkdir(parents=True)
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def sample(qc, audit, near, out, long_hours=160, short_hours=40, source_ids=P0):
    if out.exists():
        raise FileExistsError(out)
    q, a, n, qc_digest = validated_inputs(qc, audit, near)
    if long_hours <= 0 or short_hours < 0:
        raise ValueError("invalid pool caps")
    sources = {f"SLM-SPEECH-{x}" for x in source_ids}
    caps = {"long": long_hours * 3600, "short": short_hours * 3600}
    heaps = defaultdict(list)
    seconds = defaultdict(float)
    counts = Counter()
    for i, row in enumerate(rows(qc / "decisions.jsonl"), 1):
        if row["state"] != "QC_PASS_CANDIDATE" or row["source"] not in sources:
            continue
        duration = float(row["duration_s"])
        pool = "short" if duration < 8 else "long"
        cap = caps[pool]
        if cap == 0:
            continue
        rank = int(hashlib.sha256(f"{SEED}:{row['key']}".encode()).hexdigest(), 16)
        cell = (row["source"], pool)
        item = (-rank, row["key"], row["source_shard"], duration)
        heapq.heappush(heaps[cell], item)
        seconds[cell] += duration
        while seconds[cell] > cap:
            removed = heapq.heappop(heaps[cell])
            seconds[cell] -= removed[3]
        counts[f"seen_{pool}"] += 1
        if i % 1000000 == 0:
            print(f"PROPOSAL_SCAN {i} rows", flush=True)
    selected = []
    for cell, heap in heaps.items():
        source, pool = cell
        for rank, key, shard, duration in heap:
            selected.append(dict(key=key, source=source, source_shard=shard, pool=pool,
                                 duration_s=duration, selection_rank_sha256=f"{-rank:064x}",
                                 candidate_only=True, training_eligible=False))
    selected.sort(key=lambda r: (r["source_shard"], r["key"]))
    if len({r["key"] for r in selected}) != len(selected):
        raise ValueError("duplicate selected key")
    out.mkdir(parents=True)
    with (out / "selected-keys.jsonl").open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_cell = {}
    for (source, pool), duration in sorted(seconds.items()):
        by_cell[f"{source}:{pool}"] = dict(rows=len(heaps[(source, pool)]), hours=round(duration / 3600, 3))
    policy = dict(schema="speechlm-semcommit-proposal-policy-v1", sources=list(source_ids),
                  long_hours_per_source=long_hours, short_hours_per_source=short_hours,
                  selection_seed=SEED, selection_method="smallest_sha256_rank_with_duration_cap")
    report = dict(schema="speechlm-semcommit-approval-proposal-v1", complete=True,
                  training_eligible=False, approval_status="proposal_only", policy=policy,
                  qc_summary_sha256=qc_digest, audit_summary_sha256=sha256(audit / "summary.json"),
                  near_summary_sha256=sha256(near / "summary.json"), selected_rows=len(selected),
                  selected_hours=round(sum(seconds.values()) / 3600, 3), by_cell=by_cell,
                  selected_keys_sha256=sha256(out / "selected-keys.jsonl"))
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--near", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--long-hours", type=float, default=160)
    parser.add_argument("--short-hours", type=float, default=40)
    parser.add_argument("--sources", default=",".join(P0), help="comma-separated catalog IDs")
    parser.add_argument("--all-qc-pass", action="store_true", help="all 19 audited sources, no time cap; still candidate-only")
    args = parser.parse_args()
    if args.all_qc_pass:
        all_pass(args.qc, args.audit, args.near, args.out)
    else:
        sample(args.qc, args.audit, args.near, args.out, args.long_hours, args.short_hours,
               tuple(x for x in args.sources.split(",") if x))


if __name__ == "__main__":
    main()
