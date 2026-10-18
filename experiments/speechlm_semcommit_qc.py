#!/usr/bin/env python3
"""Freeze completed speechlm shards and audit exact duplicates/alignment.

This is a *candidate QC*, not automatic training approval. Only the fixed
snapshot is read; running or incomplete alignment shards are ignored.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection_alignment_qc import validate_original


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def approved_hashes(view):
    """Exact waveform/locator exclusion against the existing approved set."""
    waves, locators = set(), set()
    index = view / "supervision-inputs.jsonl"
    if not index.is_file():
        raise FileNotFoundError(index)
    for i, item in enumerate(jsonl(index), 1):
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"approved source changed: {path}")
        for row in jsonl(path):
            if row.get("waveform_sha256"):
                waves.add(row["waveform_sha256"])
            p = (row.get("audio") or {}).get("path")
            if p:
                locators.add(p)
        if i % 100 == 0:
            print(f"APPROVED_INDEX {i} files waveforms={len(waves)}", flush=True)
    return waves, locators


def snapshot(results, out, max_parts=0, verify_outputs=True):
    parts = []
    # Listing directory names avoids a metadata RPC for every summary before
    # max_parts can take effect (especially costly on the remote NFS mount).
    names = sorted(n for n in os.listdir(results) if n.startswith("part-"))
    print(f"SNAPSHOT_BEGIN listed={len(names)}", flush=True)
    for name in names:
        p = results / name / "summary.json"
        try:
            s = json.loads(p.read_text())
        except FileNotFoundError:
            continue
        if s.get("complete") is not True:
            continue
        for name in ("aligned.jsonl", "records.jsonl"):
            path = p.parent / name
            # Parallel workers verify existence and content hashes while reading
            # each shard. Avoid thousands of redundant metadata RPCs here.
            if verify_outputs and (not path.is_file() or digest(path) != s.get("outputs", {}).get(name)):
                raise ValueError(f"completed source changed: {path}")
        parts.append(dict(part=p.parent.name, summary_sha256=digest(p), outputs={
            k: s["outputs"][k] for k in ("aligned.jsonl", "records.jsonl")},
            aligned=s["aligned"], rows=s["rows"], fingerprint=s["fingerprint"]))
        if len(parts) % 100 == 0:
            print(f"SNAPSHOT {len(parts)} completed parts", flush=True)
        if max_parts and len(parts) >= max_parts:
            break
    if not parts:
        raise ValueError("no completed shards")
    with (out / "snapshot.json").open("x", encoding="utf-8") as f:
        json.dump(dict(schema="speechlm-semcommit-snapshot-v1", results=str(results), parts=parts), f, ensure_ascii=False)
    return parts


def _read_hashed_jsonl(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"source checksum mismatch: {path}")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _parallel_part(job):
    """Independent shard verification/QC; global dedup happens in ordered parent."""
    results, item = job
    part = Path(results) / item["part"]
    if digest(part / "summary.json") != item["summary_sha256"]:
        raise ValueError(f"source summary changed: {part}")
    records_rows = _read_hashed_jsonl(part / "records.jsonl", item["outputs"]["records.jsonl"])
    records = {r["key"]: r for r in records_rows}
    if len(records_rows) != item["rows"] or len(records) != item["rows"]:
        raise ValueError(f"record row/key accounting changed: {part}")
    aligned_rows = _read_hashed_jsonl(part / "aligned.jsonl", item["outputs"]["aligned.jsonl"])
    if len(aligned_rows) != item["aligned"]:
        raise ValueError(f"aligned row accounting changed: {part}")
    decisions = []
    for row in aligned_rows:
        rec = records.get(row.get("key"))
        source = row.get("source", "unknown")
        state, why = "QC_PASS_CANDIDATE", ""
        meta = (rec or {}).get("source_manifest") or {}
        if not rec or rec.get("selection_state") != "KEEP_ASR_SILVER":
            state, why = "REJECT_JOIN", "missing or non-silver source record"
        elif rec.get("split") != "train" or meta.get("split") != "train":
            state, why = "REJECT_SPLIT", "not training split"
        elif meta.get("num_channels") != 1:
            state, why = "REJECT_CHANNEL", "not mono"
        elif any(x in str(meta.get("corpus", "")).lower() for x in ("earnings", "turnbench")):
            state, why = "REJECT_HELDOUT", "evaluation-only corpus"
        elif not row.get("waveform_sha256") or not (row.get("audio") or {}).get("path"):
            state, why = "REJECT_HASH", "missing audio identity"
        else:
            try:
                validate_original(dict(row, schema="qwen-verbatim-align-v1"))
                if any(float(w["end_s"]) == float(w["start_s"]) for w in row["words"]):
                    state, why = "REVIEW_ZERO_DURATION", "at least one zero-duration aligned word"
            except Exception as exc:
                state, why = "REJECT_ALIGNMENT", f"{type(exc).__name__}: {str(exc)[:180]}"
        decisions.append(dict(key=row["key"], source=source, source_shard=item["part"], state=state,
                              reason=why, waveform_sha256=row.get("waveform_sha256"),
                              audio_locator=(row.get("audio") or {}).get("path"),
                              duration_s=row.get("duration_s"), training_eligible=False))
    print(f"PART_OK {item['part']} aligned={len(decisions)}", flush=True)
    return decisions


def audit_parallel(results, parts, out, approved, workers):
    old_waves, old_locators = approved_hashes(approved)
    print(f"APPROVED_INDEX_DONE waveforms={len(old_waves)} locators={len(old_locators)}", flush=True)
    new_waves, new_locators = set(), set()
    counts, by_source = Counter(), defaultdict(Counter)
    sample = defaultdict(list)
    jobs = [(str(results), item) for item in parts]
    with (out / "decisions.jsonl").open("x", encoding="utf-8") as decisions, \
         (out / "review-sample.jsonl").open("x", encoding="utf-8") as review, \
         ProcessPoolExecutor(max_workers=workers) as pool:
        for part_no, batch in enumerate(pool.map(_parallel_part, jobs, chunksize=1), 1):
            for row in batch:
                state, wave, locator = row["state"], row["waveform_sha256"], row["audio_locator"]
                if state in ("QC_PASS_CANDIDATE", "REVIEW_ZERO_DURATION"):
                    if wave in old_waves or locator in old_locators:
                        row.update(state="REJECT_OLD_DUP", reason="exact duplicate of approved training set")
                    elif wave in new_waves or locator in new_locators:
                        row.update(state="REJECT_INTRA_DUP", reason="exact duplicate in speechlm snapshot")
                    else:
                        new_waves.add(wave); new_locators.add(locator)
                state = row["state"]
                counts[state] += 1; by_source[row["source"]][state] += 1
                decisions.write(json.dumps(row, ensure_ascii=False) + "\n")
                if state in ("QC_PASS_CANDIDATE", "REVIEW_ZERO_DURATION"):
                    rank = hashlib.sha256(f"{row['source']}:{state}:{row['key']}".encode()).hexdigest()
                    cell = (row["source"], state)
                    sample[cell].append((rank, row))
                    sample[cell].sort(key=lambda x: x[0]); del sample[cell][20:]
            if part_no % 100 == 0:
                print(f"AUDIT {part_no}/{len(parts)} parts aligned={sum(counts.values())} pass={counts['QC_PASS_CANDIDATE']} review={counts['REVIEW_ZERO_DURATION']}", flush=True)
        for cell in sorted(sample):
            for _, row in sample[cell]:
                review.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(schema="speechlm-semcommit-qc-v1", complete=True, training_eligible=False,
                  snapshot_sha256=digest(out / "snapshot.json"), approved_view=str(approved),
                  approved_existing_waveforms=len(old_waves), parts=len(parts), counts=dict(counts),
                  by_source={k: dict(v) for k, v in by_source.items()},
                  decisions_sha256=digest(out / "decisions.jsonl"),
                  review_sample_sha256=digest(out / "review-sample.jsonl"),
                  remaining_gates=["heldout_exact_and_near_duplicate_audit", "acoustic_sample_review", "source_balance", "human_approval"])
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def audit(results, parts, out, approved):
    old_waves, old_locators = approved_hashes(approved)
    new_waves, new_locators = set(), set()
    counts, by_source = Counter(), defaultdict(Counter)
    with (out / "decisions.jsonl").open("x", encoding="utf-8") as decisions, \
         (out / "review-sample.jsonl").open("x", encoding="utf-8") as review:
        sample = defaultdict(list)
        for part_no, item in enumerate(parts, 1):
            part = results / item["part"]
            records = {r["key"]: r for r in jsonl(part / "records.jsonl")}
            if len(records) != item["rows"]:
                raise ValueError(f"record row/key accounting changed: {item['part']}")
            seen_aligned = 0
            for row in jsonl(part / "aligned.jsonl"):
                seen_aligned += 1
                rec = records.get(row.get("key"))
                source = row.get("source", "unknown")
                state, why = "QC_PASS_CANDIDATE", ""
                meta = (rec or {}).get("source_manifest") or {}
                if not rec or rec.get("selection_state") != "KEEP_ASR_SILVER":
                    state, why = "REJECT_JOIN", "missing or non-silver source record"
                elif rec.get("split") != "train" or meta.get("split") != "train":
                    state, why = "REJECT_SPLIT", "not training split"
                elif meta.get("num_channels") != 1:
                    state, why = "REJECT_CHANNEL", "not mono"
                elif any(x in str(meta.get("corpus", "")).lower() for x in ("earnings", "turnbench")):
                    state, why = "REJECT_HELDOUT", "evaluation-only corpus"
                elif not row.get("waveform_sha256") or not (row.get("audio") or {}).get("path"):
                    state, why = "REJECT_HASH", "missing audio identity"
                elif row["waveform_sha256"] in old_waves or row["audio"]["path"] in old_locators:
                    state, why = "REJECT_OLD_DUP", "exact duplicate of approved training set"
                elif row["waveform_sha256"] in new_waves or row["audio"]["path"] in new_locators:
                    state, why = "REJECT_INTRA_DUP", "exact duplicate in speechlm snapshot"
                else:
                    try:
                        probe = dict(row, schema="qwen-verbatim-align-v1")
                        validate_original(probe)
                        if any(float(w["end_s"]) == float(w["start_s"]) for w in row["words"]):
                            state, why = "REVIEW_ZERO_DURATION", "at least one zero-duration aligned word"
                    except Exception as exc:
                        state, why = "REJECT_ALIGNMENT", f"{type(exc).__name__}: {str(exc)[:180]}"
                if state in ("QC_PASS_CANDIDATE", "REVIEW_ZERO_DURATION"):
                    new_waves.add(row["waveform_sha256"])
                    new_locators.add(row["audio"]["path"])
                counts[state] += 1
                by_source[source][state] += 1
                compact = dict(key=row["key"], source=source, source_shard=item["part"], state=state,
                               reason=why, waveform_sha256=row.get("waveform_sha256"),
                               duration_s=row.get("duration_s"), training_eligible=False)
                decisions.write(json.dumps(compact, ensure_ascii=False) + "\n")
                if state in ("QC_PASS_CANDIDATE", "REVIEW_ZERO_DURATION"):
                    rank = hashlib.sha256(f"{source}:{state}:{row['key']}".encode()).hexdigest()
                    cell = (source, state)
                    sample[cell].append((rank, compact))
                    sample[cell].sort(key=lambda x: x[0])
                    del sample[cell][20:]
            if seen_aligned != item["aligned"]:
                raise ValueError(f"aligned row accounting changed: {item['part']}")
            if part_no % 100 == 0:
                print(f"AUDIT {part_no}/{len(parts)} parts aligned={sum(counts.values())} pass={counts['QC_PASS_CANDIDATE']} review={counts['REVIEW_ZERO_DURATION']}", flush=True)
        for cell in sorted(sample):
            for _, row in sample[cell]:
                review.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = dict(schema="speechlm-semcommit-qc-v1", complete=True, training_eligible=False,
                  snapshot_sha256=digest(out / "snapshot.json"), approved_view=str(approved),
                  approved_existing_waveforms=len(old_waves), parts=len(parts), counts=dict(counts),
                  by_source={k: dict(v) for k, v in by_source.items()},
                  decisions_sha256=digest(out / "decisions.jsonl"),
                  review_sample_sha256=digest(out / "review-sample.jsonl"),
                  remaining_gates=["heldout_exact_and_near_duplicate_audit", "acoustic_sample_review", "source_balance", "human_approval"])
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--approved-view", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-parts", type=int, default=0, help="smoke only; 0 means every completed part")
    p.add_argument("--workers", type=int, default=1, help="independent checksum/QC workers; 1 uses the original serial implementation")
    a = p.parse_args()
    if a.max_parts < 0:
        p.error("--max-parts must be nonnegative")
    if not 1 <= a.workers <= 32:
        p.error("--workers must be 1..32")
    a.out.mkdir(parents=True, exist_ok=False)
    parts = snapshot(a.results, a.out, a.max_parts, verify_outputs=a.workers == 1)
    report = (audit(a.results, parts, a.out, a.approved_view) if a.workers == 1
              else audit_parallel(a.results, parts, a.out, a.approved_view, a.workers))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
