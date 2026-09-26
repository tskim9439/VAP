import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"experiments/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


split_mod, approve_mod, fin_mod, col_mod = (load(n) for n in (
    "semcommit_split_qc_pass", "semcommit_approve_speechlm_words", "semcommit_part_finalize", "semcommit_collect_done"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_interleave_round_robin():
    order = split_mod.interleave({"B": ["p3", "p4", "p5"], "A": ["p1", "p2"]})
    assert order == [("p1", "A"), ("p3", "B"), ("p2", "A"), ("p4", "B"), ("p5", "B")]   # any prefix covers every DB


def test_split_approve_finalize_collect(tmp_path):
    keys = [f"{i:064x}" for i in range(4)]
    dec = tmp_path / "decisions.jsonl"
    states = ["QC_PASS_CANDIDATE", "QC_PASS_CANDIDATE", "REVIEW_ZERO_DURATION", "QC_PASS_CANDIDATE"]
    dec.write_text("".join(json.dumps(dict(key=k, source="SRC", source_shard="part-000001", state=s)) + "\n" for k, s in zip(keys, states)))
    te = tmp_path / "te.json"
    te.write_text(json.dumps(dict(complete=True, training_eligible=True, labeling_eligible=True, approval_status="gold_gate_passed",
                                  approved_filter="state == QC_PASS_CANDIDATE", approval_fingerprint="fp1",
                                  decision_file=str(dec), decision_file_sha256=sha(dec))))
    split = tmp_path / "split"
    s = split_mod.split(te, split)
    assert s["pass_rows"] == 3 and (split / "parts-order.tsv").read_text() == "part-000001\tSRC\t3\n"
    rows = [dict(id=f"slm-SRC-{k}", set="speechlm-partial", lang="Korean", duration_s=d, source_key=k, candidate_only=True,
                 training_eligible=False, words=[dict(i=0, text="네", a=0, b=1, end_time=0.3, seg=0, tags=["punct_final"])])
            for k, d in zip(keys, (3.0, 12.0, 9.0, 31.0))]
    cand = tmp_path / "part-000001.words.jsonl"
    cand.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "part-000001.stats.json").write_text(json.dumps(dict(candidate_only=True, words_sha256=sha(cand))))
    w = tmp_path / "part" / "words.jsonl"; w.parent.mkdir()
    ok = approve_mod.approve_part(te, split, "part-000001", cand, w)
    assert ok["counts"] == {"kept_short": 1, "kept_long": 2, "drop_not_qc_pass": 1}
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(dict(id=r["id"], candidates=[])) + "\n" for r in rows[:2]))   # id 3 (31 s) unlabeled
    done = w.parent / "DONE.json"
    rec = fin_mod.finalize("part-000001", w, labels, None, w.parent / "pools-a", done,
                           dict(job="1", words_dir=str(w.parent), thresholds="/x/thresholds.json"))   # worker --meta (key clash regression)
    assert json.loads(done.read_text())["meta"]["thresholds"] == "/x/thresholds.json"
    assert rec["counts"] == dict(words=3, labeled=2, main=1, short=1, over_30s=0)
    for pool in ("main", "short"):
        ws = [json.loads(l)["id"] for l in open(rec["files"][pool]["words"])]
        assert ws == [json.loads(l)["id"] for l in open(rec["files"][pool]["labels"])]
    with pytest.raises(FileExistsError):
        fin_mod.finalize("part-000001", w, labels, None, w.parent / "pools-b", done)
    labels_root = tmp_path / "root"; (labels_root / "parts").mkdir(parents=True)
    (labels_root / "parts" / "part-000001").symlink_to(w.parent)
    (labels_root / "parts" / "part-000002").mkdir()                                        # unfinished part: no DONE.json
    snap = col_mod.collect(labels_root, tmp_path / "snap", verify=True, order=split / "parts-order.tsv")
    assert snap["parts"] == 1 and snap["main_files"] == 1 and snap["short_files"] == 1 and snap["bad_parts"] == []
    assert (tmp_path / "snap" / "short-words.list").read_text().strip() == rec["files"]["short"]["words"]
