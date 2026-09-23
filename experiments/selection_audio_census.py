#!/usr/bin/env python3
"""CPU-only full audio QC. Read originals; write new per-utterance sidecars.

Bounded batches, explicit error rows, per-source completion markers, immutable
input/code fingerprints. No dataset is automatically promoted to training.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import itertools
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest, digest
from vapasr.data.selection_audio import decode


def check(r):
    try:
        _, info = decode(r)
        state = "AUDIO_REVIEW" if info["review"] else "AUDIO_OK_TEXT_PENDING"
        return dict(key=r["key"], source=r["source"], status=state, audio=info,
                    declared_duration_s=r["duration_s"], training_eligible=False)
    except Exception as e:
        return dict(key=r["key"], source=r["source"], status="AUDIO_ERROR",
                    error=f"{type(e).__name__}: {e}", training_eligible=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not 1 <= a.workers <= 16:
        raise ValueError("workers must be 1..16")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    census = json.loads((a.selection / "census.json").read_text())
    if not census.get("complete"):
        raise ValueError("Incomplete metadata census")
    hold = json.loads((a.selection / "mnsc-source-audit.json").read_text())
    if not hold["decision"].startswith("HOLD_"):
        raise ValueError("Expected explicit MNSC audit disposition")
    root = Path(__file__).resolve().parents[1]
    fp = dict(census_sha256=file_digest(a.selection / "census.json"),
              mnsc_audit_sha256=file_digest(a.selection / "mnsc-source-audit.json"),
              code_sha256={str(p.relative_to(root)): file_digest(p) for p in
                   [Path(__file__).resolve(), *sorted((root / "vapasr/data").glob("*.py"))]})
    a.out.mkdir(parents=True, exist_ok=True)
    fp_path = a.out / "fingerprint.json"
    if fp_path.exists() and json.loads(fp_path.read_text()) != fp:
        raise ValueError("Audio census fingerprint changed; use a new output directory")
    if not fp_path.exists():
        fp_path.write_text(json.dumps(fp, indent=2))
    # Target the newly discovered timebase risks first. No additional GPUs.
    order = ["aihub-bc-train", "nikl-1000", "voxpopuli-train"]
    order += [s for s in census["sources"] if s not in order]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for source in order:
            marker = a.out / f"{source}.done.json"
            if marker.exists():
                print(f"SKIP completed {source}", flush=True)
                continue
            if source == "mnsc-1000":
                marker.write_text(json.dumps(dict(source=source,
                    status="SOURCE_HOLD", audit="../mnsc-source-audit.json",
                    fingerprint=digest(fp))))
                continue
            src = a.selection / f"{source}.candidates.jsonl.gz"
            dst = a.out / f"{source}.audio.jsonl.gz"
            # Preserve interrupted work. A distinct attempt file can be resumed
            # at source granularity without clobbering partial evidence.
            attempt = 0
            while dst.exists():
                attempt += 1
                dst = a.out / f"{source}.attempt-{attempt}.audio.jsonl.gz"
            counts, reasons, hours, done = Counter(), Counter(), 0.0, 0
            started = time.monotonic()
            with gzip.open(src, "rt") as f, gzip.open(dst, "wt", compresslevel=3) as out:
                rows = (r for r in map(json.loads, f) if r["selection_state"] == "PENDING_AUDIO_TEACHERS")
                while True:
                    batch = list(itertools.islice(rows, 1024))
                    if not batch:
                        break
                    for r in pool.map(check, batch):
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        counts[r["status"]] += 1
                        if "audio" in r:
                            hours += r["audio"]["duration_s"] / 3600
                            reasons.update(r["audio"]["review"])
                        done += 1
                    print(json.dumps(dict(source=source, rows=done, counts=dict(counts),
                                          elapsed_s=round(time.monotonic()-started, 1))), flush=True)
            marker.write_text(json.dumps(dict(source=source, counts=dict(counts), reasons=dict(reasons),
                audio_hours=hours, output=str(dst), output_sha256=file_digest(dst),
                elapsed_s=time.monotonic()-started, fingerprint=digest(fp),
                training_eligible=False), indent=2))


if __name__ == "__main__":
    main()
