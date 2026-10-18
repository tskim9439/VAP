import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_approval_finalize as mod
import speechlm_semcommit_approval_proposal as proposal


def fixture(tmp_path):
    qc, audit, near, bundle = (tmp_path / name for name in ("qc", "audit", "near", "bundle"))
    for root in (qc, audit, near, bundle, bundle / "audio"):
        root.mkdir()
    sources = [f"SLM-SPEECH-{i:06d}" for i in range(1, 20)]
    decisions, samples, cards = [], [], []
    for source in sources:
        for cell in mod.CELLS:
            key = f"{source}-{cell}"
            state = "REVIEW_ZERO_DURATION" if cell == "zero_duration" else "QC_PASS_CANDIDATE"
            decisions.append(dict(key=key, state=state, source=source, source_shard="part-0",
                                  duration_s=5 if cell == "short" else 10))
            sample = dict(key=key, source=source, review_cell=[source, cell],
                          waveform_sha256=key)
            samples.append(sample)
            audio_file = f"audio/{len(cards):05d}.wav"
            (bundle / audio_file).write_bytes(key.encode())
            cards.append(dict(**sample, qc_state=state, acoustic_flags=[], audio_file=audio_file))
    (qc / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
    (qc / "summary.json").write_text(json.dumps(dict(complete=True, training_eligible=False,
        decisions_sha256=mod.sha256(qc / "decisions.jsonl"),
        counts=dict(QC_PASS_CANDIDATE=38, REVIEW_ZERO_DURATION=19),
        by_source={source: dict(QC_PASS_CANDIDATE=2, REVIEW_ZERO_DURATION=1) for source in sources})))
    qc_digest = mod.sha256(qc / "summary.json")
    (audit / "review-sample.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
    (audit / "summary.json").write_text(json.dumps(dict(complete=True, qc_summary_sha256=qc_digest,
        preliminary=dict(CANDIDATE_SHORT=19, CANDIDATE_LONG=19),
        candidate_hours_by_source={source: 0.004 for source in sources},
        review_sample_sha256=mod.sha256(audit / "review-sample.jsonl"))))
    (near / "summary.json").write_text(json.dumps(dict(complete=True, qc_summary_sha256=qc_digest,
        counts=dict(near_hits=0))))
    scope = tmp_path / "scope"
    proposal.all_pass(qc, audit, near, scope)
    items = bundle / "review-items.jsonl"
    items.write_text("".join(json.dumps(r) + "\n" for r in cards))
    (bundle / "index.html").write_text("review")
    (bundle / "summary.json").write_text(json.dumps(dict(complete=True, sample_rows=57,
        approval_audit_sha256=mod.sha256(audit / "summary.json"),
        review_items_sha256=mod.sha256(items), html_sha256=mod.sha256(bundle / "index.html"))))
    attestation = tmp_path / "human-review.json"
    attestation.write_text(json.dumps(dict(schema="speechlm-human-review-attestation-v1",
        reviewer_id="tskim", review_date="2026-09-25", statement="57개 모두 통과",
        verdict="pass_all", reviewed_items=57, exception_keys=[],
        bundle_summary_sha256=mod.sha256(bundle / "summary.json"),
        bundle_review_items_sha256=mod.sha256(items),
        bundle_audio_manifest_sha256=mod.audio_manifest_sha256(bundle, cards))))
    return qc, audit, near, scope, bundle, attestation


def test_human_review_approves_qc_pass_for_labeling_only(tmp_path):
    inputs = fixture(tmp_path)
    report = mod.finalize(*inputs, tmp_path / "approved")
    assert report["approved_rows"] == 38
    assert report["zero_duration_population_excluded"] == 19
    assert report["labeling_eligible"] is True
    assert report["training_eligible"] is False
    assert report["approved_filter"] == "state == QC_PASS_CANDIDATE"


def test_human_review_rejects_changed_audio(tmp_path):
    inputs = fixture(tmp_path)
    (inputs[4] / "audio" / "00000.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio files changed"):
        mod.finalize(*inputs, tmp_path / "approved")


def test_human_review_rejects_missing_cell(tmp_path):
    inputs = fixture(tmp_path)
    items = inputs[4] / "review-items.jsonl"
    cards = list(mod.rows(items))
    cards[-1]["review_cell"][1] = "short"
    items.write_text("".join(json.dumps(r) + "\n" for r in cards))
    bundle_summary = json.loads((inputs[4] / "summary.json").read_text())
    bundle_summary["review_items_sha256"] = mod.sha256(items)
    (inputs[4] / "summary.json").write_text(json.dumps(bundle_summary))
    att = json.loads(inputs[5].read_text())
    att["bundle_review_items_sha256"] = mod.sha256(items)
    att["bundle_summary_sha256"] = mod.sha256(inputs[4] / "summary.json")
    inputs[5].write_text(json.dumps(att))
    with pytest.raises(ValueError, match="review coverage"):
        mod.finalize(*inputs, tmp_path / "approved")
