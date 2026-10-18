import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_approval_audit as mod


def _jsonl(path, items):
    path.write_text("".join(json.dumps(x) + "\n" for x in items))


def test_preapproval_holds_lineage_heldout_and_zero_duration(tmp_path):
    qc = tmp_path / "qc"
    qc.mkdir()
    source = "SLM-SPEECH-000004"
    decisions = [
        dict(key="a", source=source, state="QC_PASS_CANDIDATE", duration_s=5,
             waveform_sha256="different", audio_locator="/train/a"),
        dict(key="b", source=source, state="QC_PASS_CANDIDATE", duration_s=10,
             waveform_sha256="protected", audio_locator="/train/b"),
        dict(key="c", source="SLM-SPEECH-000032", state="QC_PASS_CANDIDATE", duration_s=10,
             waveform_sha256="novel", audio_locator="/train/c"),
        dict(key="d", source=source, state="REVIEW_ZERO_DURATION", duration_s=10,
             waveform_sha256="another", audio_locator="/train/d"),
    ]
    _jsonl(qc / "decisions.jsonl", decisions)
    (qc / "summary.json").write_text(json.dumps(dict(complete=True, training_eligible=False,
        decisions_sha256=mod.sha256(qc / "decisions.jsonl"),
        counts={"QC_PASS_CANDIDATE": 3, "REVIEW_ZERO_DURATION": 1})))
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    _jsonl(heldout / "identities.jsonl", [dict(path="/test/z", waveform_sha256="protected",
                                                hash_status="ok")])
    (heldout / "summary.json").write_text(json.dumps(dict(complete=True, max_items=0,
        identities_sha256=mod.sha256(heldout / "identities.jsonl"),
        pcm_scope=["test"], path_only_scope=[])))
    result = mod.audit(qc, heldout, tmp_path / "out")
    assert result["preliminary"] == dict(CANDIDATE_SHORT=1, HOLD_HELDOUT_EXACT=1,
                                         HOLD_SOURCE_POLICY=1, HOLD_ZERO_DURATION=1)
    assert result["training_eligible"] is False
    assert len(list(mod.rows(tmp_path / "out" / "heldout-collisions.jsonl"))) == 1
