#!/usr/bin/env python3
"""Freeze source-level human approval for QC-pass speechlm rows.

This view authorizes semcommit labeling input only.  Rows with failed QC stay
excluded, and training_eligible remains false until teacher labels pass gates.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from speechlm_semcommit_approval_proposal import rows, sha256, validated_inputs


CELLS = ("short", "long", "zero_duration")


def audio_manifest_sha256(bundle, cards):
    names = sorted(card["audio_file"].split("/", 1)[1] for card in cards)
    if len(set(names)) != len(names):
        raise ValueError("duplicate audio filename")
    lines = []
    for name in names:
        if "/" in name or name.startswith("."):
            raise ValueError("unsafe audio filename")
        audio = bundle / "audio" / name
        if not audio.is_file():
            raise ValueError(f"missing reviewed audio: {name}")
        lines.append(f"{sha256(audio)}  {name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def finalize(qc, audit, near, scope, bundle, attestation, out):
    if out.exists():
        raise FileExistsError(out)
    q, a, _, qc_digest = validated_inputs(qc, audit, near)
    s = json.loads((scope / "summary.json").read_text())
    if (s.get("complete") is not True or s.get("training_eligible") is not False
            or s.get("approval_status") != "scope_chosen_human_review_pending"
            or s.get("policy", {}).get("mode") != "all_qc_pass"
            or s.get("policy", {}).get("decision_filter") != "state == QC_PASS_CANDIDATE"
            or s.get("qc_summary_sha256") != qc_digest
            or s.get("audit_summary_sha256") != sha256(audit / "summary.json")
            or s.get("near_summary_sha256") != sha256(near / "summary.json")
            or s.get("decision_file_sha256") != q["decisions_sha256"]
            or s.get("selected_rows") != q["counts"]["QC_PASS_CANDIDATE"]):
        raise ValueError("selected scope is not the audited all-QC-pass snapshot")
    sources = set(s["policy"]["sources"])
    if len(sources) != 19 or sources != set(q["by_source"]):
        raise ValueError("expected the same 19 QC sources")
    b = json.loads((bundle / "summary.json").read_text())
    bundle_digest = sha256(bundle / "summary.json")
    items = bundle / "review-items.jsonl"
    if (b.get("complete") is not True or b.get("sample_rows") != 57
            or b.get("approval_audit_sha256") != sha256(audit / "summary.json")
            or b.get("review_items_sha256") != sha256(items)
            or b.get("html_sha256") != sha256(bundle / "index.html")):
        raise ValueError("review bundle checksum or audit mismatch")
    t = json.loads(attestation.read_text())
    if (t.get("schema") != "speechlm-human-review-attestation-v1"
            or t.get("reviewer_id") != "tskim" or t.get("verdict") != "pass_all"
            or t.get("statement") != "57개 모두 통과"
            or t.get("reviewed_items") != 57 or t.get("exception_keys") != []
            or t.get("bundle_summary_sha256") != bundle_digest
            or t.get("bundle_review_items_sha256") != sha256(items)):
        raise ValueError("human review attestation does not match bundle")
    cards = list(rows(items))
    if len(cards) != 57 or len({card["key"] for card in cards}) != 57:
        raise ValueError("review cards missing or duplicated")
    cells = Counter((card["source"], card["review_cell"][1]) for card in cards)
    expected_cells = {(source, cell) for source in sources for cell in CELLS}
    if set(cells) != expected_cells or any(count != 1 for count in cells.values()):
        raise ValueError("review coverage is not 19 sources x 3 cells")
    expected = {}
    for row in rows(audit / "review-sample.jsonl"):
        cell = tuple(row["review_cell"])
        expected.setdefault(cell, row)
    if sha256(audit / "review-sample.jsonl") != a["review_sample_sha256"]:
        raise ValueError("review sample changed")
    for card in cards:
        cell = (card["source"], card["review_cell"][1])
        source_row = expected.get(cell)
        state = "REVIEW_ZERO_DURATION" if cell[1] == "zero_duration" else "QC_PASS_CANDIDATE"
        if (source_row is None or card["key"] != source_row["key"]
                or card["waveform_sha256"] != source_row["waveform_sha256"]
                or card["qc_state"] != state or card["acoustic_flags"]):
            raise ValueError(f"review item mismatch: {cell}")
    audio_digest = audio_manifest_sha256(bundle, cards)
    if audio_digest != t.get("bundle_audio_manifest_sha256"):
        raise ValueError("reviewed audio files changed")
    report = dict(schema="speechlm-semcommit-source-approval-v1", complete=True,
                  approval_status="source_approved_for_semcommit_labeling",
                  data_approved=True, labeling_eligible=True, training_eligible=False,
                  reviewer_id=t["reviewer_id"], review_date=t["review_date"],
                  human_reviewed_items=57, human_reviewed_sources=19,
                  reviewed_zero_duration_examples=19,
                  zero_duration_population_excluded=q["counts"]["REVIEW_ZERO_DURATION"],
                  approved_filter="state == QC_PASS_CANDIDATE",
                  approved_rows=s["selected_rows"], approved_short_rows=s["selected_short_rows"],
                  approved_long_rows=s["selected_long_rows"], approved_hours=s["selected_hours"],
                  approved_sources=sorted(sources),
                  decision_file=s["decision_file"], decision_file_sha256=s["decision_file_sha256"],
                  qc_summary_sha256=qc_digest, audit_summary_sha256=sha256(audit / "summary.json"),
                  near_summary_sha256=sha256(near / "summary.json"),
                  scope_summary_sha256=sha256(scope / "summary.json"),
                  review_bundle_summary_sha256=bundle_digest,
                  review_audio_manifest_sha256=audio_digest,
                  attestation_sha256=sha256(attestation),
                  remaining_gates=["words_serializer_and_punctuation_qc", "teacher_gold_gate",
                                   "teacher_label_qc", "training_mixture_and_step_sampler"],
                  limitations=["57 sampled items do not establish per-row transcription correctness",
                               "285-item expanded bundle was not human-reviewed",
                               "AI Hub dev heldout check was path-only",
                               "text-near screen is not acoustic-near duplicate proof"])
    out.mkdir(parents=True)
    with (out / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("qc", "audit", "near", "scope", "bundle", "attestation", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    finalize(args.qc, args.audit, args.near, args.scope, args.bundle, args.attestation, args.out)


if __name__ == "__main__":
    main()
