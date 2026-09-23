#!/usr/bin/env python
"""공통 manifest와 모델 독립 segments를 MeetEval로 채점. 누락 세션도 보고한다."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.baseline_eval import read_manifest, score_session, score_text, seglst, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    a = p.parse_args()
    rows = read_manifest(a.manifest)
    run_path = a.results / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["manifest_sha256"] != sha256(a.manifest):
        raise ValueError("Run manifest fingerprint mismatch")
    scored, missing, groups = [], [], {}
    for i, r in enumerate(rows):
        path = a.results / f"session-{i:04d}.json"
        if not path.exists():
            missing.append(r["session_id"])
            continue
        h = json.loads(path.read_text())
        if h["session_id"] != r["session_id"] or h["audio_sha256"] != r["audio_sha256"]:
            raise ValueError("Result manifest mismatch")
        m = score_session(r["session_id"], r["reference"], h["segments"], r["lang"])
        scored.append(dict(session_id=r["session_id"], corpus=r["corpus"], metrics=m,
                           cap_hits=sum(c.get("cap_hit", False) for c in h.get("chunks", [])),
                           unassigned_units=sum(len(score_text(s["text"], r["lang"]).split())
                               for s in h["segments"] if s["speaker"] == "__unassigned__")))
        # 시각을 발명하지 않는 SegLST: 참조 발화순·가설 출력순으로만 채점.
        for name, segments in (("ref", r["reference"]), ("hyp", h["segments"])):
            (a.results / f"session-{i:04d}.{name}.seglst.json").write_text(
                json.dumps(seglst(r["session_id"], segments, r["lang"]), ensure_ascii=False, indent=2))
        group = groups.setdefault(r["corpus"], dict(unit="CER-nospace" if r["lang"] in ("ko", "Korean") else "WER",
                                                  sessions=0, cp={}, orc={}))
        group["sessions"] += 1
        for metric in ("cp", "orc"):
            for key in ("errors", "length", "substitutions", "deletions", "insertions"):
                group[metric][key] = group[metric].get(key, 0) + m[metric][key]
    for g in groups.values():
        for metric in ("cp", "orc"):
            n = g[metric]["length"]
            g[metric]["rate"] = g[metric]["errors"]/n if n else None
    report = dict(complete=not missing, manifest_sha256=sha256(a.manifest),
                  meeteval_version=importlib.metadata.version("meeteval"), expected=len(rows),
                  missing=missing, sessions=scored, by_corpus=groups)
    (a.results / "scores.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["by_corpus"], ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(f"Incomplete evaluation: {len(missing)} sessions missing")


if __name__ == "__main__":
    main()
