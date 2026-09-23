#!/usr/bin/env python3
"""Join matching two-teacher evidence into PROVISIONAL decisions, never gold.

Stratified pilot counts are not whole-corpus quality estimates. Outputs include
every input row, including missing or failed inference. Never drop a timeline.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import agreement, adjudicate, file_digest, digest


def load(path):
    result = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["key"] in result:
                raise ValueError(f"Duplicate teacher result: {r['key']}")
            result[r["key"]] = r
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--qwen", type=Path, required=True)
    ap.add_argument("--whisper", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--human-review", type=Path)
    ap.add_argument("--review-manifest", type=Path)
    ap.add_argument("--source-hold", action="append", default=[], metavar="SOURCE=AUDIT_JSON",
                    help="Quarantine an entire source after a provenance audit")
    a = ap.parse_args()
    if bool(a.human_review) != bool(a.review_manifest):
        ap.error('--human-review and --review-manifest must be supplied together')
    from vapasr.data.selection_review import load_reviews, review_flags
    human = load_reviews(a.human_review, a.review_manifest) if a.human_review else {}
    applied = set()
    from vapasr.data.textnorm import score_en, score_ko, fingerprint
    holds = {}
    for item in a.source_hold:
        source, path = item.split("=", 1)
        audit = json.loads(Path(path).read_text())
        if not audit.get("decision", "").startswith("HOLD_"):
            raise ValueError("Source hold requires an explicit audit HOLD decision")
        holds[source] = dict(audit_path=path, audit_sha256=file_digest(path),
                             manifest_sha256=audit["manifest_sha256"])
    q, w = load(a.qwen), load(a.whisper)
    for teacher, path, records in (("qwen", a.qwen, q), ("whisper", a.whisper, w)):
        fp = json.loads(path.with_suffix(".fingerprint.json").read_text())
        if fp["teacher"] != teacher:
            raise ValueError("Cannot use one teacher twice")
        if fp["input_sha256"] != file_digest(a.input):
            raise ValueError("Teacher input mismatch")
        if fp["tn"] != fingerprint():
            raise ValueError("Teacher TN mismatch")
        if any(r.get("teacher") != teacher or r.get("fingerprint") != digest(fp)
               for r in records.values()):
            raise ValueError("Teacher row provenance mismatch")
    a.out.mkdir(parents=True, exist_ok=False)
    summary, reasons = defaultdict(Counter), defaultdict(Counter)
    with a.input.open() as src, (a.out / "decisions.jsonl").open("w") as dst:
        for line in src:
            r = json.loads(line)
            qr, wr = q.get(r["key"]), w.get(r["key"])
            hard, review, metrics = list(r["reasons"]), [], None
            if r['key'] in human:
                if r['key'] in applied:
                    raise ValueError('Duplicate application of human review')
                evidence = human[r['key']]
                hf, hr = review_flags(r, evidence, (qr, wr))
                hard.extend(hf); review.extend(hr)
                r['human_review'] = evidence
                applied.add(r['key'])
            if r["source"] in holds:
                if holds[r["source"]]["manifest_sha256"] != r["manifest_sha256"]:
                    raise ValueError("Source hold audit is for a different manifest")
                hard.append("source_integrity_hold")
            ok = all(t is not None and t["status"] == "ok" for t in (qr, wr))
            if r["selection_state"] != "PENDING_AUDIO_TEACHERS":
                hard.append("input_not_training_candidate")
            for t in (qr, wr):
                if t and t["status"] == "audio_error":
                    hard.append("audio_decode_failed")
                if t and t["status"] == "ok":
                    review.extend(t["audio"]["review"])
                    review.extend("tn:" + f for f in t["tn_flags"])
                    if t.get("warning"):
                        review.append(t["warning"])
            if ok:
                if qr["audio"]["sha256"] != wr["audio"]["sha256"]:
                    hard.append("teacher_audio_hash_mismatch")
                normalize = (lambda x: list(score_ko(x, False))) if r["lang"] == "Korean" else (lambda x: score_en(x).split())
                metrics = agreement(normalize(r["text"]), normalize(qr["hyp"]), normalize(wr["hyp"]))
            state, why = adjudicate(metrics, hard=hard, review=review, teachers_ok=ok)
            r.update(selection_state=state, reasons=why, metrics=metrics,
                     qwen=qr, whisper=wr, training_eligible=False,
                     text_quality="provisional_dual_teacher" if ok else "pending",
                     timing_quality="unverified",
                     speaker_quality="unverified", turn_quality="unverified")
            summary[r["source"]][state] += 1
            reasons[r["source"]].update(why)
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
    if applied != set(human):
        raise ValueError('Some human review keys were not applied; output is incomplete')
    with (a.out / "summary.json").open("w") as f:
        json.dump(dict(scope="stratified pilot only, not population pass rate",
            training_eligible=0, input_sha256=file_digest(a.input),
            implementation_sha256=file_digest(Path(__file__).resolve()),
            source_holds=holds,
            human_review_sha256=file_digest(a.human_review) if a.human_review else None,
            human_review_applied=len(applied),
            sources={k: dict(v) for k, v in summary.items()},
            reasons={k: dict(v) for k, v in reasons.items()}), f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
