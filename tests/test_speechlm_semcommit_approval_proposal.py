import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_approval_proposal as mod


def test_proposal_is_capped_and_never_training_eligible(tmp_path):
    qc, audit, near = (tmp_path / x for x in ("qc", "audit", "near"))
    for path in (qc, audit, near):
        path.mkdir()
    source = "SLM-SPEECH-000004"
    decisions = [dict(key=str(i), state="QC_PASS_CANDIDATE", source=source,
                      source_shard="part-0", duration_s=2.0) for i in range(10)]
    (qc / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
    (qc / "summary.json").write_text(json.dumps(dict(complete=True, training_eligible=False,
        decisions_sha256=mod.sha256(qc / "decisions.jsonl"), counts=dict(QC_PASS_CANDIDATE=10))))
    qc_digest = mod.sha256(qc / "summary.json")
    (audit / "summary.json").write_text(json.dumps(dict(complete=True, qc_summary_sha256=qc_digest,
        preliminary=dict(CANDIDATE_SHORT=10, CANDIDATE_LONG=0))))
    (near / "summary.json").write_text(json.dumps(dict(complete=True, qc_summary_sha256=qc_digest,
        counts=dict(near_hits=0))))
    report = mod.sample(qc, audit, near, tmp_path / "out", long_hours=1,
                        short_hours=6/3600, source_ids=("000004",))
    assert report["selected_rows"] == 3
    assert report["training_eligible"] is False
    assert all(r["candidate_only"] and not r["training_eligible"]
               for r in mod.rows(tmp_path / "out" / "selected-keys.jsonl"))


def test_all_pass_scope_references_full_qc_without_approving(tmp_path):
    qc, audit, near = (tmp_path / x for x in ("qc", "audit", "near"))
    for path in (qc, audit, near):
        path.mkdir()
    sources = [f"SLM-SPEECH-{i:06d}" for i in range(1, 20)]
    decisions = [dict(key=str(i), state="QC_PASS_CANDIDATE", source=s,
                      source_shard="part-0", duration_s=5) for i, s in enumerate(sources)]
    (qc / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in decisions))
    (qc / "summary.json").write_text(json.dumps(dict(complete=True, training_eligible=False,
        decisions_sha256=mod.sha256(qc / "decisions.jsonl"),
        counts=dict(QC_PASS_CANDIDATE=19),
        by_source={s: dict(QC_PASS_CANDIDATE=1) for s in sources})))
    qc_digest = mod.sha256(qc / "summary.json")
    (audit / "summary.json").write_text(json.dumps(dict(complete=True, qc_summary_sha256=qc_digest,
        preliminary=dict(CANDIDATE_SHORT=19, CANDIDATE_LONG=0),
        candidate_hours_by_source={s: 0.001 for s in sources})))
    (near / "summary.json").write_text(json.dumps(dict(complete=True, qc_summary_sha256=qc_digest,
        counts=dict(near_hits=0))))
    result = mod.all_pass(qc, audit, near, tmp_path / "out")
    assert result["selected_rows"] == 19
    assert result["selected_short_rows"] == 19
    assert result["training_eligible"] is False
    assert not (tmp_path / "out" / "selected-keys.jsonl").exists()
