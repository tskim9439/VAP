#!/usr/bin/env python3
"""Read-only MNSC path/label audit after pilot missing-audio failures.

Counts conflicts, never chooses the first of conflicting source transcripts as
truth. No audio files or upstream labels are edited.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    available = {p.name for p in a.audio_dir.iterdir() if p.name.endswith(".wav")}
    manifest_counts, manifest_hours = Counter(), Counter()
    with a.manifest.open() as f:
        for line in f:
            for s in json.loads(line)["segments"]:
                path = Path(s["path"])
                state = ("listed_on_disk" if path.parent == a.audio_dir and path.name in available
                         else "missing_at_declared_path")
                manifest_counts[state] += 1
                manifest_hours[state] += s["dur_s"] / 3600
    first, conflicting_ids, conflicting_paths = {}, set(), set()
    counts, examples = Counter(), []
    with a.csv.open(encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("|")
        if header != ["audio_path", "script", "sampling_rate", "duration", "script_tn"]:
            raise ValueError(f"Unexpected header {header}")
        for line in f:
            v = line.rstrip("\n").split("|")
            if len(v) != 5:
                counts["ambiguous_delimiter_row"] += 1
                continue
            path, raw, sr, duration, target = v
            if "ASR-PART1" not in path:
                continue
            counts["part1_rows"] += 1
            uid = Path(path).stem
            th = hashlib.sha256(target.encode()).hexdigest()
            if uid not in first:
                first[uid] = (path, th, target)
                continue
            counts["repeated_id_rows"] += 1
            prev_path, prev_th, prev_text = first[uid]
            if path != prev_path:
                conflicting_paths.add(uid)
            if th != prev_th:
                counts["target_differs_from_first_rows"] += 1
                conflicting_ids.add(uid)
                if len(examples) < 8:
                    examples.append(dict(uid=uid, first_path=prev_path, next_path=path,
                                         first_target=prev_text, next_target=target))
    report = dict(csv=str(a.csv), csv_sha256=file_digest(a.csv),
        manifest=str(a.manifest), manifest_sha256=file_digest(a.manifest),
        directory=str(a.audio_dir), listed_wav_files=len(available),
        scope="filename availability and label conflicts only; not full audio decoding",
        manifest_counts=dict(manifest_counts), manifest_hours=dict(manifest_hours),
        csv_counts=dict(counts), unique_part1_ids=len(first),
        ids_with_conflicting_targets=len(conflicting_ids),
        ids_with_different_paths=len(conflicting_paths), examples=examples,
        decision="HOLD_MNSC_UNTIL_SOURCE_MAPPING_VALIDATED")
    with a.out.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != "examples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
