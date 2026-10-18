import hashlib
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import speechlm_semcommit_qc as mod


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_snapshot_skips_incomplete_and_qc_never_approves(tmp_path):
    results = tmp_path / "results"
    part = results / "part-000000"
    part.mkdir(parents=True)
    aligned = dict(key="k", source="catalog", lang="English", audio=dict(path="/x.tar::a.wav"),
                   target_text="Hi.", target_sha256=hashlib.sha256(b"Hi.").hexdigest(),
                   waveform_sha256="w1", duration_s=2.0, alignment_ok=True,
                   training_eligible=False, timing_scope="decoded_clip_relative",
                   aligner_items=[dict(text="Hi", start_time=0.1, end_time=0.5)],
                   words=[dict(text="Hi", char_start=0, char_end=2, start_s=0.1, end_s=0.5)],
                   tokens=[dict(id=1, char_start=0, char_end=2, end_s=0.5,
                                timing_kind="word_end_projection", lexical_timing_mask=True),
                           dict(id=2, char_start=2, char_end=3, end_s=0.5,
                                timing_kind="nonlexical_anchor", lexical_timing_mask=False)])
    rec = dict(key="k", split="train", selection_state="KEEP_ASR_SILVER",
               source_manifest=dict(split="train", num_channels=1, corpus="example"))
    write_jsonl(part / "aligned.jsonl", [aligned])
    write_jsonl(part / "records.jsonl", [rec])
    summary = dict(complete=True, aligned=1, rows=1, fingerprint="f",
                   outputs={name: mod.digest(part / name) for name in ("aligned.jsonl", "records.jsonl")})
    (part / "summary.json").write_text(json.dumps(summary))
    (results / "part-000001").mkdir()
    (results / "part-000001" / "summary.json").write_text(json.dumps(dict(complete=False)))
    old = tmp_path / "approved"
    old.mkdir()
    write_jsonl(old / "supervision.jsonl", [])
    write_jsonl(old / "supervision-inputs.jsonl", [dict(path=str(old / "supervision.jsonl"),
                                                     sha256=mod.digest(old / "supervision.jsonl"))])
    out = tmp_path / "out"
    out.mkdir()
    parts = mod.snapshot(results, out)
    assert len(parts) == 1
    report = mod.audit(results, parts, out, old)
    assert report["training_eligible"] is False
    assert report["counts"] == {"QC_PASS_CANDIDATE": 1}
    assert next(mod.jsonl(out / "decisions.jsonl"))["training_eligible"] is False
    parallel = tmp_path / "parallel"
    parallel.mkdir()
    parts2 = mod.snapshot(results, parallel, verify_outputs=False)
    report2 = mod.audit_parallel(results, parts2, parallel, old, workers=2)
    assert report2["counts"] == report["counts"]
    assert next(mod.jsonl(parallel / "decisions.jsonl"))["training_eligible"] is False
