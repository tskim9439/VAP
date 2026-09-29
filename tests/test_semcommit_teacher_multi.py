"""semcommit_teacher.py multi — several parts' stages in one process with the judge model kept resident between jobs that
use the same model (part worker GROUP_PARTS). Fake model injected like test_semcommit_llm.test_cli_pipeline_resume_and_errors."""
import json

import pytest

from tests.test_semcommit_llm import FakeLLM, KO1, KO2, EN1
from vapasr.data import semcommit_llm as sc


def _jobs(tmp_path, parts):
    """Stage A for every part, then B(qwen3) and C(qwen3), then B(exaone35) — ordered by model like the worker does."""
    J, paths = [], {}
    for part, streams in parts.items():
        d = tmp_path / part; d.mkdir()
        (d / "words.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in streams))
        paths[part] = d
    base = lambda d: ["--words", str(d / "words.jsonl"), "--device", "cpu", "--batch-size", "2"]
    for part, d in paths.items():
        J.append(dict(tag=part, argv=["stageA", *base(d), "--model", "q", "--kind", "qwen3", "--out", str(d / "A.jsonl")]))
    for part, d in paths.items():
        J.append(dict(tag=part, argv=["stageB", *base(d), "--model", "q", "--kind", "qwen3", "--stageA", str(d / "A.jsonl"),
                                      "--out", str(d / "B.qwen3.jsonl")]))
        J.append(dict(tag=part, argv=["stageC", *base(d), "--model", "q", "--kind", "qwen3", "--stageA", str(d / "A.jsonl"),
                                      "--out", str(d / "C.jsonl")]))
    for part, d in paths.items():
        J.append(dict(tag=part, argv=["stageB", *base(d), "--model", "e", "--kind", "exaone35", "--stageA", str(d / "A.jsonl"),
                                      "--out", str(d / "B.exaone35.jsonl"), "--flush-every", "1"]))
    jp = tmp_path / "jobs.jsonl"; jp.write_text("".join(json.dumps(j) + "\n" for j in J))
    return jp, paths


def test_multi_reuses_model_and_isolates_failures(tmp_path, monkeypatch, capsys):
    from experiments import semcommit_teacher as T
    made = []
    monkeypatch.setattr(T, "LLM", lambda path, kind, **kw: made.append((path, kind)) or FakeLLM(kind))
    jp, paths = _jobs(tmp_path, {"p1": [KO1, EN1], "p2": [KO2]})
    res = tmp_path / "result.json"
    with pytest.raises(SystemExit) as e:
        T.main(["multi", "--jobs", str(jp), "--result", str(res)])
    assert e.value.code == 0
    assert made == [("q", "qwen3"), ("e", "exaone35")]                        # one adapter per model for 8 jobs
    r = json.loads(res.read_text()); assert r["done"] == r["total"] == 8 and r["failed_tags"] == []
    assert all(j["status"] == "ok" for j in r["jobs"])
    for d in paths.values():
        for f in ("A.jsonl", "B.qwen3.jsonl", "C.jsonl", "B.exaone35.jsonl"): assert sc.read_jsonl(d / f)
    assert T._RESIDENT == {}                                                  # released at the end
    # a failing job marks its tag; the other part keeps going and the failed tag's later jobs are skipped
    made.clear()
    (tmp_path / "second").mkdir(); jp2, _ = _jobs(tmp_path / "second", {"p3": [KO1], "p4": [KO2]})
    J = [json.loads(l) for l in jp2.read_text().splitlines()]
    for j in J:
        if j["tag"] == "p3" and j["argv"][0] == "stageB" and "qwen3" in j["argv"]: j["argv"][j["argv"].index("--stageA") + 1] = str(tmp_path / "missing.jsonl")
    jp2.write_text("".join(json.dumps(j) + "\n" for j in J))
    with pytest.raises(SystemExit) as e:
        T.main(["multi", "--jobs", str(jp2), "--result", str(res)])
    assert e.value.code == 2
    r = json.loads(res.read_text()); st = {(j["tag"], j["cmd"], (j["out"] or "").rsplit("/", 1)[-1]): j["status"] for j in r["jobs"]}
    assert r["failed_tags"] == ["p3"] and st[("p3", "stageA", "A.jsonl")] == "ok" and st[("p3", "stageB", "B.qwen3.jsonl")].startswith("error")
    assert st[("p3", "stageC", "C.jsonl")] == "skipped" and st[("p3", "stageB", "B.exaone35.jsonl")] == "skipped"
    assert all(v == "ok" for (t, _, _), v in st.items() if t == "p4")
    assert "MULTI_SUMMARY jobs=8 failed_tags=1" in capsys.readouterr().out


def test_single_command_does_not_cache(tmp_path, monkeypatch):
    from experiments import semcommit_teacher as T
    made = []
    monkeypatch.setattr(T, "LLM", lambda path, kind, **kw: made.append(kind) or FakeLLM(kind))
    W = tmp_path / "w.jsonl"; W.write_text(json.dumps(KO1, ensure_ascii=False) + "\n")
    for n in ("A1", "A2"):
        T.main(["stageA", "--words", str(W), "--device", "cpu", "--model", "q", "--kind", "qwen3", "--out", str(tmp_path / f"{n}.jsonl")])
    assert made == ["qwen3", "qwen3"] and T._RESIDENT == {}


def test_multi_without_gpu_exits_75_and_runs_nothing(tmp_path, monkeypatch):
    """A task bound to a GPU the node does not expose: exit NO_GPU_EXIT before any job (the worker then stops claiming parts)."""
    import torch
    from experiments import semcommit_teacher as T
    made = []
    monkeypatch.setattr(T, "LLM", lambda path, kind, **kw: made.append(kind) or FakeLLM(kind))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    d = tmp_path / "p"; d.mkdir(); (d / "words.jsonl").write_text(json.dumps(KO1, ensure_ascii=False) + "\n")
    jp = tmp_path / "jobs.jsonl"
    jp.write_text(json.dumps(dict(tag="p", argv=["stageA", "--words", str(d / "words.jsonl"), "--model", "q", "--kind", "qwen3", "--out", str(d / "A.jsonl")])) + "\n")
    with pytest.raises(SystemExit) as e:
        T.main(["multi", "--jobs", str(jp), "--result", str(tmp_path / "r.json")])
    assert e.value.code == T.NO_GPU_EXIT == 75 and made == [] and not (d / "A.jsonl").exists() and not (tmp_path / "r.json").exists()
