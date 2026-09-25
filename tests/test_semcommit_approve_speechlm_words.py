import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "semcommit_approve_speechlm_words", Path(__file__).resolve().parents[1] / "experiments/semcommit_approve_speechlm_words.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _make(tmp_path):
    keys = [f"{i:064x}" for i in range(4)]
    rows = [dict(id=f"slm-SRC-{k}", set="speechlm-partial", lang="Korean", duration_s=d, source_key=k, candidate_only=True,
                 training_eligible=False, words=[dict(i=0, text="네", a=0, b=1, end_time=0.3, seg=0, tags=["punct_final"])])
            for k, d in zip(keys, (3.0, 12.0, 9.0, 2.0))]
    cand = tmp_path / "part-000001.words.jsonl"
    cand.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (tmp_path / "part-000001.stats.json").write_text(json.dumps(dict(candidate_only=True, words_sha256=_sha(cand))))
    dec = tmp_path / "decisions.jsonl"
    states = ["QC_PASS_CANDIDATE", "QC_PASS_CANDIDATE", "REVIEW_ZERO_DURATION", "QC_PASS_CANDIDATE"]
    dec.write_text("".join(json.dumps(dict(key=k, source="SRC", source_shard="part-000001", state=s)) + "\n" for k, s in zip(keys, states)))
    te = tmp_path / "te.json"
    te.write_text(json.dumps(dict(complete=True, training_eligible=True, labeling_eligible=True, approval_status="gold_gate_passed",
                                  approved_filter="state == QC_PASS_CANDIDATE", approval_fingerprint="fp1",
                                  decision_file=str(dec), decision_file_sha256=_sha(dec))))
    return te, cand, dec


def test_approve_filters_stamps_and_pools(tmp_path):
    te, cand, _ = _make(tmp_path)
    out = tmp_path / "approved"
    s = mod.approve(te, [cand], out, shard_rows=1)
    assert s["counts"] == {"kept_short": 2, "kept_long": 1, "drop_REVIEW_ZERO_DURATION": 1}
    idx = [l.split("\t") for l in (out / "index.tsv").read_text().split("\n") if l]
    assert sorted(p for p, _ in idx) == ["long", "short", "short"]
    for pool, path in idx:
        for r in map(json.loads, open(path)):
            assert r["training_eligible"] is True and r["candidate_only"] is False and r["approval_fingerprint"] == "fp1"
            assert (r["duration_s"] < 8) == (pool == "short")      # the worker's pool rule
    assert s["punct"]["stream_end_rate"] == 1.0
    with pytest.raises(FileExistsError):
        mod.approve(te, [cand], out)


def test_approve_refuses_tampered_inputs(tmp_path):
    te, cand, dec = _make(tmp_path)
    dec.write_text(dec.read_text().replace("REVIEW_ZERO_DURATION", "QC_PASS_CANDIDATE"))
    with pytest.raises(ValueError, match="digest mismatch"):
        mod.approve(te, [cand], tmp_path / "a1")
    (tmp_path / "x").mkdir(); te2, cand2, _ = _make(tmp_path / "x")
    cand2.write_text(cand2.read_text() + "\n")
    with pytest.raises(ValueError, match="builder stats"):
        mod.approve(te2, [cand2], tmp_path / "a2")
    t = json.loads(te2.read_text()); t["training_eligible"] = False; te2.write_text(json.dumps(t))
    with pytest.raises(ValueError, match="gate-passed"):
        mod.approve(te2, [cand2], tmp_path / "a3")
