#!/usr/bin/env python3
"""Join completed QC to candidates; stratified human-review sample, no promotion."""
import argparse
from collections import Counter, defaultdict
import gzip
import heapq
import itertools
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest


def joined(candidates, results):
    eligible = (r for r in candidates if r["selection_state"] == "PENDING_AUDIO_TEACHERS")
    for row, qc in itertools.zip_longest(eligible, results):
        if row is None or qc is None or (row["key"], row["source"]) != (qc["key"], qc["source"]):
            raise ValueError("QC/candidate sequence mismatch")
        if qc.get("training_eligible") is not False:
            raise ValueError("QC must not approve training")
        yield row, qc


def subgroup(row):
    if row["source"] == "nikl-1000":
        m = re.search(r"20[12][0-9]", row["audio"]["path"])
        return m.group(0) if m else "unknown-year"
    return "all"


def bucket(row, qc):
    flags = qc.get("audio", {}).get("review", [])
    error = qc.get("error", "").split(":", 1)[0]
    return (subgroup(row), qc["status"], "+".join(sorted(flags)) or error or "none", row["stratum"])


def keep_sample(heap, row, qc, cap):
    rank = int(digest(["review-calibration-v1", row["key"]]), 16)
    item = (-rank, row["key"], row, qc)
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--qc", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-cell", type=int, default=8)
    a = ap.parse_args()
    if a.per_cell < 1:
        raise ValueError("per-cell must be positive")
    census = json.loads((a.selection / "census.json").read_text())
    if not census.get("complete"):
        raise ValueError("Incomplete census")
    fp = json.loads((a.qc / "fingerprint.json").read_text())
    if fp["census_sha256"] != file_digest(a.selection / "census.json"):
        raise ValueError("QC census fingerprint mismatch")
    algorithm_fp = {k: fp[k] for k in ("census_sha256", "mnsc_audit_sha256", "code_sha256")}
    a.out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, training_eligible=0, per_cell=a.per_cell,
        census_sha256=file_digest(a.selection / "census.json"),
        implementation_sha256=file_digest(Path(__file__).resolve()), sources={})
    selected = []
    for source, meta in census["sources"].items():
        marker = a.qc / f"{source}.done.json"
        data = json.loads(marker.read_text())
        if data["source"] != source or data["fingerprint"] != digest(algorithm_fp):
            raise ValueError("QC provenance mismatch")
        if data.get("status") == "SOURCE_HOLD":
            report["sources"][source] = dict(status="SOURCE_HOLD")
            continue
        result = Path(data["output"])
        if file_digest(result) != data["output_sha256"]:
            raise ValueError(f"QC output checksum mismatch: {source}")
        candidate_path = a.selection / f"{source}.candidates.jsonl.gz"
        counts, cells, groups = Counter(), defaultdict(list), {}
        candidates_sha = file_digest(candidate_path)
        with gzip.open(candidate_path, "rt") as cf, gzip.open(result, "rt") as qf:
            for row, qc in joined(map(json.loads, cf), map(json.loads, qf)):
                if row["manifest_sha256"] != meta["sha256"]:
                    raise ValueError("Manifest provenance mismatch")
                counts[qc["status"]] += 1
                group = groups.setdefault(subgroup(row), dict(counts=Counter(), delta_ms=Counter(),
                    ratio=Counter(), format=Counter(), reasons=Counter(), errors=Counter()))
                group["counts"][qc["status"]] += 1
                if "audio" in qc:
                    info = qc["audio"]
                    group["delta_ms"][str(round((info["duration_s"]-row["duration_s"])*1000))] += 1
                    group["ratio"][str(round(row["duration_s"]/info["duration_s"], 3))] += 1
                    group["format"][f'{info["original_sample_rate"]}Hz/{info["original_channels"]}ch'] += 1
                    group["reasons"].update(info["review"])
                else:
                    group["errors"][qc.get("error", "unknown").split(": ", 1)[0]] += 1
                keep_sample(cells[bucket(row, qc)], row, qc, a.per_cell)
        if counts != Counter(data["counts"]):
            raise ValueError("QC count mismatch")
        for cell, heap in sorted(cells.items()):
            for _, _, row, qc in sorted(heap, key=lambda x: x[1]):
                row = dict(row, review_cell=list(cell), audio_qc=qc, training_eligible=False)
                selected.append(row)
        report["sources"][source] = dict(counts=dict(counts), groups=groups,
            marker=str(marker), output=str(result), output_sha256=data["output_sha256"],
            candidates_sha256=candidates_sha, review_n=sum(map(len, cells.values())))
        print(json.dumps(dict(source=source, counts=dict(counts), review_n=sum(map(len, cells.values())))), flush=True)
    selected.sort(key=lambda r: (r["source"], r["key"]))
    with (a.out / "review.jsonl").open("x") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    report.update(complete=True, review_n=len(selected),
                  review_sha256=file_digest(a.out / "review.jsonl"))
    (a.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("COMPLETE review_n="+str(len(selected)), flush=True)


if __name__ == "__main__":
    main()
