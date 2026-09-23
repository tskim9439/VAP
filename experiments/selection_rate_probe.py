#!/usr/bin/env python3
"""Diagnostic 48 kHz hypothesis for NIKL 2022; NEVER changes canonical inputs."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest
from vapasr.data.selection_audio import decode


def main():
    import numpy as np
    import soundfile as sf
    import soxr
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = list(map(json.loads, a.input.open()))
    chosen = [r for r in rows if r["source"] == "nikl-1000"
              and r["review_cell"][0] == "2022" and "audio" in r["audio_qc"]
              and 2.9 < r["audio_qc"]["audio"]["duration_s"] / r["duration_s"] < 3.1]
    if not chosen:
        raise ValueError("No diagnostic candidates")
    (a.out / "audio").mkdir(parents=True, exist_ok=False)
    with (a.out / "probe.jsonl").open("x") as f:
        for original in chosen:
            path = Path(original["audio"]["path"])
            data = path.read_bytes()
            if len(data) % 2 or data[:4] in (b"RIFF", b"fLaC"):
                raise ValueError("Not assumed headerless int16 PCM")
            wave = np.frombuffer(data, dtype="<i2").astype(np.float32)/32768
            wave = soxr.resample(wave, 48000, 16000)
            row = copy.deepcopy(original)
            row["key"] = digest([original["key"], "diagnostic-only-48000Hz-v1"])
            target = a.out / "audio" / (row["key"]+".wav")
            sf.write(target, wave, 16000, subtype="FLOAT")
            row.update(audio=dict(path=str(target), offset_s=None, duration_s=None),
                       duration_s=len(wave)/16000, training_eligible=False,
                       diagnostic_only=True, source_key=original["key"],
                       rate_hypothesis=48000, original_audio=original["audio"],
                       original_audio_sha256=file_digest(path),
                       original_declared_duration_s=original["duration_s"],
                       review_cell=["2022", "DIAGNOSTIC_ONLY", "48000Hz-hypothesis", original["stratum"]])
            _, info = decode(row)
            row["audio_qc"] = dict(key=row["key"], source=row["source"], status="AUDIO_REVIEW",
                                   audio=info, training_eligible=False)
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    (a.out / "fingerprint.json").write_text(json.dumps(dict(
        input_sha256=file_digest(a.input), script_sha256=file_digest(Path(__file__)),
        n=len(chosen), diagnostic_only=True, training_eligible=False), indent=2))
    print("COMPLETE diagnostic_rows="+str(len(chosen)), flush=True)


if __name__ == "__main__":
    main()
