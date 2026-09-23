#!/usr/bin/env python3
"""Resume the unchanged audio QC with more I/O workers in a NEW output tree.

Checks the original decoder/worker fingerprint; reuses verified complete DBs and
the readable prefix of an interrupted gzip. Never edits the previous run.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
import itertools
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.selection_audio_census import check
from vapasr.data.selection import digest, file_digest


def readable_prefix(path):
    """Only tolerate EOF from an interrupted gzip, not CRC/JSON corruption."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        while True:
            try:
                line = f.readline()
            except EOFError:
                return
            if not line or not line.endswith("\n"):
                return
            yield json.loads(line)


def validate_previous(selection, previous):
    root = Path(__file__).resolve().parents[1]
    fp = json.loads((previous / "fingerprint.json").read_text())
    if fp["census_sha256"] != file_digest(selection / "census.json"):
        raise ValueError("Metadata census changed")
    if fp["mnsc_audit_sha256"] != file_digest(selection / "mnsc-source-audit.json"):
        raise ValueError("MNSC source hold changed")
    for name, expected in fp["code_sha256"].items():
        if file_digest(root / name) != expected:
            raise ValueError(f"Original QC code changed: {name}")
    return fp


def recover_prefix(path, rows, source, emit):
    n = 0
    for old in readable_prefix(path):
        reference = next(rows, None)
        if reference is None or old["key"] != reference["key"] or old["source"] != source:
            raise ValueError("Partial results do not match the candidate sequence")
        if old.get("training_eligible") is not False or old["status"] not in (
                "AUDIO_OK_TEXT_PENDING", "AUDIO_REVIEW", "AUDIO_ERROR"):
            raise ValueError("Invalid prior QC row")
        emit(old)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", required=True, type=Path)
    ap.add_argument("--resume-from", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    if not 1 <= a.workers <= 64:
        raise ValueError("workers must be 1..64")
    if a.out.resolve() == a.resume_from.resolve():
        raise ValueError("Use a new output tree to preserve previous evidence")
    old_fp = validate_previous(a.selection, a.resume_from)
    census = json.loads((a.selection / "census.json").read_text())
    if not census.get("complete"):
        raise ValueError("Incomplete metadata census")
    if a.validate_only:
        print("VALIDATED original worker, decoder, manifests and source hold", flush=True)
        return
    a.out.mkdir(parents=True, exist_ok=False)
    lock = (a.out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fp = dict(old_fp, orchestration_sha256=file_digest(Path(__file__).resolve()),
              workers=a.workers, resume_from=str(a.resume_from), pid=os.getpid())
    (a.out / "fingerprint.json").write_text(json.dumps(fp, indent=2))
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    order = ["aihub-bc-train", "nikl-1000", "voxpopuli-train"]
    order += [s for s in census["sources"] if s not in order]
    print(json.dumps(dict(event="START", workers=a.workers, pid=os.getpid(),
                          resume_from=str(a.resume_from))), flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for source in order:
            if stopping:
                return
            old_marker = a.resume_from / f"{source}.done.json"
            marker = a.out / f"{source}.done.json"
            if old_marker.exists():
                old = json.loads(old_marker.read_text())
                if old["source"] != source or old["fingerprint"] != digest(old_fp):
                    raise ValueError("Completed DB marker provenance mismatch")
                if old.get("status") != "SOURCE_HOLD":
                    if file_digest(old["output"]) != old["output_sha256"]:
                        raise ValueError("Completed DB result checksum mismatch")
                    if sum(old["counts"].values()) != census["sources"][source]["counts"].get("PENDING_AUDIO_TEACHERS", 0):
                        raise ValueError("Completed DB result count mismatch")
                old["reused_from_marker"] = str(old_marker)
                marker.write_text(json.dumps(old, indent=2))
                print(json.dumps(dict(event="REUSED_COMPLETE", source=source)), flush=True)
                continue
            if source == "mnsc-1000":
                raise ValueError("Expected prior MNSC source hold marker")
            src = a.selection / f"{source}.candidates.jsonl.gz"
            partial = a.resume_from / f"{source}.audio.jsonl.gz"
            dst = a.out / f"{source}.audio.jsonl.gz"
            counts, reasons, hours, done = Counter(), Counter(), 0.0, 0
            started = time.monotonic()
            with gzip.open(src, "rt") as f, gzip.open(dst, "wt", compresslevel=3) as out:
                rows = (r for r in map(json.loads, f) if r["selection_state"] == "PENDING_AUDIO_TEACHERS")

                def emit(r):
                    nonlocal hours, done
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counts[r["status"]] += 1
                    if "audio" in r:
                        hours += r["audio"]["duration_s"] / 3600
                        reasons.update(r["audio"]["review"])
                    done += 1

                recovered = recover_prefix(partial, rows, source, emit) if partial.exists() else 0
                print(json.dumps(dict(event="RECOVERED_PREFIX", source=source,
                                      rows=recovered)), flush=True)
                while not stopping:
                    batch = list(itertools.islice(rows, 1024))
                    if not batch:
                        break
                    for result in pool.map(check, batch):
                        emit(result)
                    print(json.dumps(dict(source=source, rows=done, reused_rows=recovered,
                        new_rows=done-recovered, counts=dict(counts), workers=a.workers,
                        elapsed_s=round(time.monotonic()-started, 1))), flush=True)
            if stopping:
                print("STOPPED after batch; gzip closed; no completion marker", flush=True)
                return
            expected = census["sources"][source]["counts"]["PENDING_AUDIO_TEACHERS"]
            if done != expected:
                raise ValueError(f"QC count mismatch: {done} != {expected}")
            marker.write_text(json.dumps(dict(source=source, counts=dict(counts),
                reasons=dict(reasons), audio_hours=hours, output=str(dst),
                output_sha256=file_digest(dst), elapsed_s=time.monotonic()-started,
                fingerprint=digest(old_fp), orchestration_fingerprint=digest(fp),
                reused_rows=recovered, workers=a.workers, training_eligible=False), indent=2))
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
