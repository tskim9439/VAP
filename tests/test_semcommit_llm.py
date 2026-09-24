"""Semantic-commit LLM relabeling — prompts, JSON 추출·검증, future window, disfluency, 등급 truth table, resume,
label scoring 수학(bigram stub), fake LLM 으로 Stage A/B/C·CLI 전 과정, (transformers≥4.50 이면) tiny HF 모델 경로."""
import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vapasr.data import semcommit_llm as sc


def mk_stream(sid, lang, texts, tags=None, dt=0.4, t0=0.3):
    tags = tags or {}
    words = [dict(i=k, text=t, a=k, b=k + 1, end_time=round(t0 + dt * (k + 1), 3), seg=0, tags=list(tags.get(k, [])))
             for k, t in enumerate(texts)]
    return dict(id=sid, set="synthetic", lang=lang, K=100, duration_s=t0 + dt * len(texts) + 0.5,
                text=" ".join(texts), tokens=[], segments=[], words=words)


KO1 = mk_stream("k1", "Korean", ["어", "저는", "내일", "아니", "오늘", "갔어요", "그리고", "밥을", "먹었어요"], {0: ["filler"]})
KO2 = mk_stream("k2", "Korean", ["병원에", "갔다가", "음", "회사에", "왔어요"], {2: ["filler"], 4: ["punct_final"]})
EN1 = mk_stream("e1", "English", ["i", "went", "home", "no", "to", "work", "today"], {6: ["punct_final"]})


# ───────────── prompts ─────────────
def test_stage_a_prompt_indices_and_hints():
    m = sc.stage_a_messages(KO2["words"], "Korean")
    assert [x["role"] for x in m] == ["system", "user"] and "Precision first" in m[0]["content"]
    u = m[1]["content"]
    assert "[000] 병원에\n" in u and "[002] 음 {filler}\n" in u and u.endswith("[004] 왔어요 {.}")
    assert "EXAMPLE OUTPUT" in u and '"semantic_boundaries": [[13, "같아요"], [17, "만나려고요"]]' in u   # KO example from plan §8 (v0.2 anchors)
    assert "EXAMPLE TRANSCRIPT\n[000] 어 {filler}\n" in u and "[008] 음 {filler}\n" in u
    en = sc.stage_a_messages(["book"] * 13, "English", tags=[[]] * 12 + [["punct_final", "filler"]])[1]["content"]
    assert en.endswith("[012] book {filler} {.}") and "let's" in en
    wide = sc.stage_a_messages([str(k) for k in range(1001)], "English", hints=False)[1]["content"]
    assert "[0000] 0" in wide and wide.endswith("[1000] 1000")


def test_stage_b_prompt_has_no_future_and_c_has_future():
    s = mk_stream("x", "English", ["we", "met", "yesterday", "FUTUREWORD", "again"], {2: ["punct_final"]})
    b = sc.stage_b_messages(s["words"][:3], "English")
    blob = json.dumps(b, ensure_ascii=False)
    assert "FUTUREWORD" not in blob and "again" not in blob and "{.}" not in blob
    assert b[1]["content"].endswith("SPOKEN SO FAR:\nwe met yesterday\nLAST WORD: yesterday")
    c = sc.stage_c_messages(s["words"][:3], s["words"][3:], "English", future_end=True)[1]["content"]
    assert c.endswith("PREFIX:\nwe met yesterday\nFUTURE:\nFUTUREWORD again [end of speech]")
    assert "아니 오늘" in sc.stage_c_messages(["내일"], ["아니"], "Korean")[1]["content"]   # KO examples


def test_prompt_version_tracks_prompt_text(monkeypatch):
    v = sc.prompt_version()
    assert v.startswith(sc.PROMPT_VERSION + "+") and len(v.split("+")[1]) == 8 and sc.prompt_version() == v
    with monkeypatch.context() as m:
        m.setitem(sc.PROMPT_TEXTS, "rules", sc.RULES + " changed")
        assert sc.prompt_version() != v
    assert sc.prompt_version() == v


def _edit_render(monkeypatch, name, old, new):
    orig = getattr(sc, name)
    monkeypatch.setattr(sc, name, lambda *a, **k: [dict(m, content=m["content"].replace(old, new)) for m in orig(*a, **k)])


@pytest.mark.parametrize("name,old,new", [("stage_b_messages", "SPOKEN SO FAR:", "WORDS HEARD UNTIL NOW:"),
                                          ("stage_c_messages", " [end of speech]", " [stream end]"),
                                          ("stage_a_messages", "EXAMPLE OUTPUT", "SAMPLE OUTPUT"),
                                          ("stage_a_messages", "{unclear}", "{?}")])
def test_prompt_version_tracks_scaffold_literals(monkeypatch, name, old, new):
    v = sc.prompt_version()
    _edit_render(monkeypatch, name, old, new)   # literals outside PROMPT_TEXTS are hashed via the rendered probe
    assert sc.prompt_version() != v
    monkeypatch.undo()
    monkeypatch.setattr(sc, "_lines", lambda words, tags=None, hints=True: "\n".join(f"<{k}> {sc._text(w)}"
                                                                                      for k, w in enumerate(words)))
    assert sc.prompt_version() != v             # the '[idx] word {hint}' line format is hashed too
    monkeypatch.undo()
    monkeypatch.setattr(sc, "C_TYPE_MID", '", "kind": "')
    assert sc.prompt_version() != v


# ───────────── JSON extraction / validation ─────────────
@pytest.mark.parametrize("text,expect", [
    ('<think>\nmaybe {"semantic_boundaries": [0]}\n</think>\n{"semantic_boundaries": [1]}', {"semantic_boundaries": [1]}),
    ('reasoning opened by template {"x": 0}</think>{"a": 1}', {"a": 1}),
    ('```json\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
    ('Sure! Here it is: {"decision": "SAFE"} hope this helps {"decision": "WAIT"}', {"decision": "SAFE"}),
    ('<|channel|>final<|message|>{"decision": "WAIT"}<|return|>', {"decision": "WAIT"}),
    ('{broken} then {"a": {"b": 2}}', {"a": {"b": 2}}),
])
def test_extract_json_ok(text, expect):
    assert sc.extract_json(text) == expect


@pytest.mark.parametrize("text", ["", "no json here", '[1, 2]', 'x <think> {"a": 1} (truncated reasoning', '{"a": '])
def test_extract_json_fails(text):
    with pytest.raises(ValueError):
        sc.extract_json(text)


BACKENDS = ["std"] + (["pydantic"] if sc._pyd_models() else [])
A_OK = {"semantic_boundaries": [5, 2, 5], "fillers": [[0, 0]], "repetitions": [],
        "repairs": [{"reparandum": [1, 1], "repair": [2, 3]}]}
A_BAD = [dict(A_OK, extra=1),                                       # extra key
         {"semantic_boundaries": [6]}, {"semantic_boundaries": [-1]},  # out of range for n=6
         {"semantic_boundaries": [True]}, {"semantic_boundaries": ["2"]}, {"semantic_boundaries": [2.0]},
         {"fillers": [[0, 0]]},                                          # missing required key
         {"semantic_boundaries": [], "fillers": [[3, 1]]},               # start > end
         {"semantic_boundaries": [], "fillers": [[0, 9]]},
         {"semantic_boundaries": [], "repetitions": [[0]]},
         {"semantic_boundaries": [], "repairs": [{"reparandum": [0, 0], "repair": [1, 1], "why": "x"}]},
         {"semantic_boundaries": [], "repairs": [{"reparandum": [0, 0]}]},
         {"semantic_boundaries": 3}, []]


@pytest.mark.parametrize("backend", BACKENDS)
def test_validate_stage_a(backend):
    out = sc.validate_stage_a(A_OK, 6, backend)
    assert out == {"semantic_boundaries": [2, 5], "fillers": [[0, 0]], "repetitions": [],
                   "repairs": [{"reparandum": [1, 1], "repair": [2, 3]}]}
    assert sc.validate_stage_a({"semantic_boundaries": []}, 1, backend)["repairs"] == []
    for bad in A_BAD:
        with pytest.raises(ValueError) as e:
            sc.validate_stage_a(bad, 6, backend)
        assert "\n" not in sc.error_text(e.value) and sc.error_text(e.value)


def test_validate_decision():
    assert sc.validate_decision({"decision": "SAFE"}) == {"decision": "SAFE"}
    assert sc.validate_decision({"relation": "REVISION", "type": "SELF_REPAIR"}, "C")["relation"] == "REVISION"
    for bad, st in [({"decision": "MAYBE"}, "B"), ({"decision": "SAFE", "why": "x"}, "B"), ({"relation": "UNOBSERVED"}, "C")]:
        with pytest.raises(ValueError):
            sc.validate_decision(bad, st)


# ───────────── future window / disfluency / grade ─────────────
def test_future_window():
    s = mk_stream("f", "English", [f"w{k}" for k in range(30)], dt=0.2)["words"]
    assert [w["i"] for w in sc.future_window(s, 3, next_candidate=6)] == [4, 5, 6]        # next candidate inclusive
    assert len(sc.future_window(s, 3)) == 12                                              # 12 words (2.4 s < 3 s)
    slow = mk_stream("g", "English", [f"w{k}" for k in range(30)], dt=0.5)["words"]
    assert [w["i"] for w in sc.future_window(slow, 3)] == [4, 5, 6, 7, 8, 9]               # 3.0 s of end_time
    gap = [dict(w) for w in slow]
    for w in gap[4:]:
        w["end_time"] += 10.0
    assert [w["i"] for w in sc.future_window(gap, 3)] == [4]                              # long pause: keep ≥1 word
    assert sc.future_window(s, 29) == []                                                  # stream end → UNOBSERVED


def test_disfluency_reason():
    a = {"fillers": [[0, 0]], "repetitions": [[7, 8]],
         "repairs": [{"reparandum": [2, 3], "repair": [5, 6]}]}   # 4 = interregnum
    tags = [[], ["rep"], [], [], [], [], [], [], [], ["unclear"], ["filler"], ["punct_final"]]
    got = {i: sc.disfluency_reason(i, a, tags) for i in range(12)}
    assert got == {0: "filler", 1: "tag_rep", 2: "reparandum", 3: "reparandum", 4: "before_repair",
                   5: "inside_repair", 6: None, 7: "repetition", 8: "repetition", 9: "tag_unclear",
                   10: "tag_filler", 11: None}
    assert sc.disfluency_conflict(4, {"repairs": [{"reparandum": [1, 1], "repair": [5, 6]}]})   # gap before onset
    assert not sc.disfluency_conflict(3, None, None)


def _g(B, rel="STABLE", lang="Korean", disfl=None, strict=True, stageA=True):
    return sc.grade(dict(stageA=stageA, B=B, C={"relation": rel} if rel else {}, disfluency=disfl), lang, strict=strict)


def test_grade_truth_table_korean():
    S, W, U = "SAFE", "WAIT", "UNCERTAIN"
    ok = {"exaone35": S, "qwen3": S}
    assert _g(ok)[:2] == ("A", "safe+stable")
    assert _g(ok, "UNOBSERVED")[0] == "A"                                     # stream end still eligible
    assert _g(ok, "REVISION") == ("N", "C_revision", False)
    assert _g({"exaone35": W, "qwen3": W}) == ("N", "B_all_wait", False)
    assert _g(ok, disfl="filler") == ("B", "disflA_filler", False)                 # Stage-A-only conflict → masked
    assert _g(ok, disfl="tag_filler") == ("N", "disfl_tag_filler", False)           # human transcript marker → hard negative
    assert _g({"exaone35": W, "qwen3": W}, "REVISION", disfl="reparandum")[1] == "C_revision+B_all_wait"          # Stage A span: not an N reason
    assert _g({"exaone35": W, "qwen3": W}, "REVISION", disfl="tag_rep")[1] == "C_revision+B_all_wait+disfl_tag_rep"
    assert _g({"exaone35": S, "qwen3": W}) == ("B", "B_disagree", False)
    assert _g({"exaone35": S, "qwen3": U})[0] == "B" and _g({"exaone35": W, "qwen3": U})[0] == "B"
    assert _g({"exaone35": U, "qwen3": U}) == ("B", "B_uncertain", False)
    res_safe = {"exaone35": S, "qwen3": W, "gptoss": S}
    assert _g(res_safe) == ("B", "B_disagree(resolved:SAFE)", True)          # strict: judge-resolved stays B
    assert _g(res_safe, strict=False) == ("A", "safe+stable+resolved", True)
    assert _g(res_safe, "REVISION", strict=False)[0] == "N"
    res_wait = {"exaone35": S, "qwen3": W, "gptoss": W}
    assert _g(res_wait) == ("B", "B_disagree(resolved:WAIT)", True) and _g(res_wait, strict=False)[0] == "B"
    assert _g({"exaone35": S, "qwen3": W, "gptoss": U}) == ("B", "B_disagree", False)   # no majority formed
    assert _g({"exaone35": S}) == ("B", "B_missing:qwen3", False)
    assert _g({"exaone35": S}, strict=False)[0] == "A"
    assert _g({}, strict=False)[0] == "B"
    assert _g(ok, rel=None) == ("B", "C_missing", False) and _g(ok, rel="MISSING") == ("B", "C_missing", False)
    assert _g(ok, stageA=False) == ("B", "not_stageA", False)
    cont = dict(relation="REVISION", type="CONTINUATION")    # A/C disagreement on completeness → masked, not N
    assert sc.grade(dict(B=ok, C=cont), "Korean") == ("B", "C_revision_masked:CONTINUATION", False)
    assert sc.grade(dict(B=ok, C=cont), "Korean", c_mask_types=())[0] == "N"
    assert sc.grade(dict(B={"exaone35": W, "qwen3": W}, C=cont), "Korean") == ("N", "B_all_wait", False)
    for t in ("SELF_REPAIR", "QUALIFICATION", "RESTART"):
        assert sc.grade(dict(B=ok, C=dict(relation="REVISION", type=t)), "Korean") == ("N", "C_revision", False)


def test_grade_truth_table_english():
    S, W = "SAFE", "WAIT"
    assert _g({"qwen3": S, "gptoss": S}, lang="English")[0] == "A"
    assert _g({"qwen3": S, "gptoss": W}, lang="English") == ("B", "B_disagree", False)   # no EN tie-break
    assert _g({"qwen3": W, "gptoss": W}, lang="English")[0] == "N"
    assert _g({"qwen3": S, "exaone35": S}, lang="English") == ("B", "B_missing:gptoss", False)   # EN judges used
    assert _g({"qwen3": S, "gptoss": S}, "REVISION", lang="English")[0] == "N"
    assert sc.grade(dict(B={"a": S, "b": S}, C={"relation": "STABLE"}), "English", judges=["a", "b"])[0] == "A"


# ───────────── resumable store ─────────────
def test_store_resume(tmp_path):
    out = tmp_path / "B.jsonl"
    fp = dict(model_path="/m", kind="qwen3", prompt_version=sc.prompt_version(), params={"k": (1, 2)})
    st = sc.JsonlStore(out, fp)
    assert out.with_suffix(".fingerprint.json").exists() and out.read_text() == "" and st.done == set()
    assert sc.JsonlStore(out, fp).n_rows == 0                     # empty output (nothing pending) resumes cleanly
    rows = [dict(key=["B", "s1", 3, "qwen3", "pv"], status="ok"), dict(key=["B", "s1", 5, "qwen3", "pv"], status="ok"),
            dict(key=["B", "s2", 1, "qwen3", "pv"], status="inference_error")]
    st.write(rows)
    st2 = sc.JsonlStore(out, fp)
    assert st2.done == {("B", "s1", 3, "qwen3", "pv"), ("B", "s1", 5, "qwen3", "pv")} and st2.n_rows == 3
    with pytest.raises(ValueError, match="kind"):
        sc.JsonlStore(out, dict(fp, kind="exaone35"))
    with out.open("a") as f:
        f.write('{"key": ["B", "s3", 1, "qwen3", "pv"], "sta')        # torn final line (crash mid-write)
    st3 = sc.JsonlStore(out, fp)
    assert st3.n_rows == 3 and out.read_text().endswith("}\n")
    st3.write([dict(key=["B", "s3", 1, "qwen3", "pv"], status="ok")])
    assert ("B", "s3", 1, "qwen3", "pv") in sc.JsonlStore(out, fp).done
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:1] + ["{not json"] + lines[1:]) + "\n")
    with pytest.raises(ValueError):
        sc.JsonlStore(out, fp)
    other = tmp_path / "C.jsonl"
    other.write_text('{"key": []}\n')
    with pytest.raises(ValueError, match="unfingerprinted"):
        sc.JsonlStore(other, fp)
    nan = tmp_path / "N.jsonl"
    st4 = sc.JsonlStore(nan, fp)
    with pytest.raises(ValueError, match="JSON compliant"):     # whole block serialized first → nothing written
        st4.write([dict(key=["B", "a", 1, "q", "pv"], status="ok"), dict(key=["B", "a", 2, "q", "pv"], margin=float("nan"))])
    assert nan.read_text() == "" and st4.done == set() and st4.n_rows == 0


# ───────────── model adapter pieces (stubs, no transformers needed) ─────────────
class CharTok:
    pad_token_id, eos_token_id, eos_token, pad_token, padding_side, unk_token_id = 0, 1, "<eos>", "<pad>", "right", None

    def convert_tokens_to_ids(self, t):
        return {"<|return|>": 7, "<|end|>": 8, "<|call|>": 9}.get(t)
    chat_template = "{{ messages }}"

    def __init__(self, tail="<|start|>assistant"):
        self.tail, self.kwargs = tail, None

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [2 + ord(c) % 30 for c in text]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        assert tokenize is False and add_generation_prompt is True
        self.kwargs = kw
        return "|".join(m["content"] for m in messages) + self.tail


class Bigram(torch.nn.Module):
    """Stub LM: logits[t] = W[input_ids[t]] (ignores mask/positions) — exact oracle for the label-scoring math."""

    def __init__(self, V=32):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(V, V, generator=torch.Generator().manual_seed(0)))
        self.emb = torch.nn.Embedding(V, 1)
        self.calls = []

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, logits_to_keep=0):
        self.calls.append(tuple(input_ids.shape))
        logits = self.W[input_ids]
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def brute_logprob(W, ids, c):
    lsm = W.log_softmax(-1)
    return sum(lsm[ids[t - 1], ids[t]].item() for t in range(len(ids) - c, len(ids)))


def test_score_labels_math_matches_bruteforce_and_oom_split(monkeypatch):
    m, tok = Bigram(), CharTok()
    llm = sc.LLM(None, "qwen3", device="cpu", model=m, tokenizer=tok)
    prompts, labels = ["short", "a much longer prompt text", "mid length"], ["SAFE", "WAIT", "UNCERTAIN"]
    res = llm.score_labels(prompts, labels, prefix=sc.B_PREFIX, batch_size=4)
    for p, r in zip(prompts, res):
        ctx = llm.encode(p + sc.B_PREFIX)
        exp = [brute_logprob(m.W.detach(), ctx + llm.encode(l + '"}'), len(llm.encode(l + '"}'))) for l in labels]
        assert [r["logprobs"][l] for l in labels] == pytest.approx(exp, abs=1e-4)
        top = sorted(exp, reverse=True)
        assert r["label"] == labels[exp.index(top[0])] and r["margin"] == pytest.approx(top[0] - top[1], abs=1e-4)
        assert r["tok_ok"] is True                                   # char tokenizer: separate == joint
    assert all(shape[0] <= 4 for shape in m.calls)
    assert llm.score_labels(prompts, labels, batch_size=1) == res                        # batch-size invariant
    real = llm._score_batch

    def flaky(jobs):
        if len(jobs) > 1:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return real(jobs)
    monkeypatch.setattr(llm, "_score_batch", flaky)
    assert llm.score_labels(prompts, labels, batch_size=4) == res                        # halved to 1 recursively


def test_oom_split_is_sticky(monkeypatch):
    llm = sc.LLM(None, "qwen3", device="cpu", model=Bigram(), tokenizer=CharTok())
    prompts, labels = [f"prompt number {k}" * (k + 1) for k in range(6)], ["SAFE", "WAIT", "UNCERTAIN"]
    ref = llm.score_labels(prompts, labels, batch_size=1)
    real, sizes, ooms = llm._score_batch, [], []

    def flaky(jobs):
        sizes.append(len(jobs))
        if len(jobs) > 2:
            ooms.append(len(jobs))
            raise torch.cuda.OutOfMemoryError("fake OOM")
        return real(jobs)
    monkeypatch.setattr(llm, "_score_batch", flaky)
    assert llm.score_labels(prompts, labels, batch_size=8) == ref            # 18 jobs: 8 → OOM, 4 → OOM, then 2s
    assert ooms == [8, 4] and llm.max_batch == 2 and max(sizes[len(ooms):]) == 2   # no re-OOM on later batches
    with pytest.raises(torch.cuda.OutOfMemoryError):
        sc.LLM(None, "qwen3", device="cpu", model=Bigram(), tokenizer=CharTok())._oom_split(
            lambda b: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")), [1])   # a batch of one re-raises


class MergeTok(CharTok):
    """Merges '"S' into one token, so the '...: "' | 'SAFE' boundary tokenizes differently when joint."""

    def __call__(self, text, add_special_tokens=False):
        ids, k = [], 0
        while k < len(text):
            ids.append(1 if text[k:k + 2] == '"S' else 2 + ord(text[k]) % 30)
            k += 2 if ids[-1] == 1 else 1
        return {"input_ids": ids}


def test_score_labels_flags_boundary_merge(capsys):
    llm = sc.LLM(None, "qwen3", device="cpu", model=Bigram(), tokenizer=MergeTok())
    res = llm.score_labels(["p1", "p2"], ["WAIT", "SAFE"], prefix=sc.B_PREFIX)
    assert [r["tok_ok"] for r in res] == [False, False] and "WARNING: label tokenization" in capsys.readouterr().out
    assert all(r["tok_ok"] for r in llm.score_labels(["p1"], ["WAIT", "UNCERTAIN"], prefix=sc.B_PREFIX))


def test_continuation_logprobs_and_helpers():
    logits = torch.zeros(1, 3, 4)
    logits[0, 1, 2] = 10.0   # position 1 predicts token at position 2
    got = sc.continuation_logprobs(logits, torch.tensor([[0, 1, 2]]), [1])[0]
    assert got == pytest.approx(10.0 - math.log(math.exp(10) + 3), abs=1e-5)
    assert sc.label_result(["A", "B"], [-1.0, -1.0]) == {"label": "A", "logprobs": {"A": -1.0, "B": -1.0}, "margin": 0.0}
    for bad in ([float("nan"), -1.0], [-1.0, float("-inf")], [float("inf"), 0.0]):
        r = sc.label_result(["A", "B"], bad)
        assert r["label"] is None and r["nonfinite"] and r["margin"] is None
        json.dumps(r, allow_nan=False)                               # stays serializable (scores as strings)
    assert sc.cut_at_eos([5, 6, 1, 7], {1}) == ([5, 6], False) and sc.cut_at_eos([5, 6], {1}) == ([5, 6], True)
    assert sc.parse_max_memory("0:37GiB,cpu:120GiB") == {0: "37GiB", "cpu": "120GiB"} and sc.parse_max_memory(None) is None


def test_build_prompt_kinds_and_gptoss_final_channel():
    tok = CharTok("<|start|>assistant")
    g = sc.LLM(None, "gptoss", device="cuda", tokenizer=tok)
    assert g.build_prompt([{"role": "user", "content": "hi"}]).endswith("<|start|>assistant<|channel|>final<|message|>")
    assert tok.kwargs == {"reasoning_effort": "low"} and g.max_memory is None and g.gpu_layers == 18   # explicit layer placement is the gptoss default (rack4 OOM with device_map=auto)
    assert g.eos_ids() == [1, 7, 8, 9]                              # tokenizer eos + harmony <|return|>/<|end|>/<|call|>
    assert sc.LLM(None, "gptoss", device="cpu", tokenizer=CharTok()).max_memory is None
    already = "<|start|>assistant<|channel|>final<|message|>"
    assert sc.gptoss_final_prefill(already) == already
    with pytest.raises(ValueError):
        sc.gptoss_final_prefill("<|im_start|>assistant\n")
    q = CharTok("<|im_start|>assistant\n")
    assert sc.LLM(None, "qwen3", device="cpu", tokenizer=q).build_prompt([{"role": "user", "content": "x"}]).endswith("assistant\n")
    assert q.kwargs == {"enable_thinking": False}
    assert sc.KINDS["exaone35"]["trust_remote_code"] and not sc.KINDS["qwen3"]["trust_remote_code"]
    with pytest.raises(ValueError):
        sc.LLM(None, "llama", tokenizer=CharTok())
    fp = sc.LLM(None, "qwen3", device="cpu", tokenizer=q).fingerprint()
    assert fp["template_kwargs"] == {"enable_thinking": False} and fp["chat_template_sha256"] == sc.digest("{{ messages }}")


# ───────────── fake LLM: stage logic end to end ─────────────
class FakeLLM:
    """Deterministic judge. Stage A: boundary at words ending in 요/home/today, fillers from {filler} hints, a repair at
    아니/no. Stage B: per-kind rules (qwen3 says WAIT on 먹었어요). Stage C: REVISION iff FUTURE starts with 아니/no."""
    fail_first = 0

    def __init__(self, kind="qwen3", bad_first_a=False, nan_last=None, nan_fut=None):
        self.kind, self.bad_first_a, self.prompts, self.calls, self.suffixes = kind, bad_first_a, [], Counter(), []
        self.nan_last, self.nan_fut = nan_last, nan_fut

    def load(self):
        return self

    def fingerprint(self):
        return {"kind": self.kind, "fake": True}

    def build_prompt(self, messages):
        return json.dumps(messages, ensure_ascii=False)

    def generate_json(self, prompts, max_new_tokens=768, batch_size=8):
        self.calls["gen"] += 1
        out = []
        for p in prompts:
            msgs = json.loads(p)
            self.prompts.append(msgs)
            if self.bad_first_a and len(msgs) == 2:
                out.append({"text": '{"semantic_boundaries": [999]}', "truncated": False})
                continue
            lines = msgs[1]["content"].split("TRANSCRIPT\n")[-1].splitlines()
            words = [ln.split("] ", 1)[1].split(" {")[0] for ln in lines]
            b = [k for k, w in enumerate(words) if w.endswith("요") or w in ("home", "today")]
            fil = [[k, k] for k, ln in enumerate(lines) if "{filler}" in ln]
            rep = [{"reparandum": [k - 1, k - 1], "repair": [k, k + 1]} for k, w in enumerate(words) if w in ("아니", "no")]
            out.append({"text": "<think>\n\n</think>\n\n```json\n" + json.dumps(
                {"semantic_boundaries": b, "fillers": fil, "repetitions": [], "repairs": rep}) + "\n```", "truncated": False})
        return out

    def score_labels(self, prompts, labels, prefix=sc.B_PREFIX, batch_size=8, suffix=sc.B_SUFFIX):
        self.suffixes.append((prefix[:14], suffix))
        if FakeLLM.fail_first:
            FakeLLM.fail_first -= 1
            raise RuntimeError("fake CUDA failure")
        self.calls["score"] += 1
        res = []
        for p in prompts:
            u = json.loads(p)[1]["content"]
            self.prompts.append(u)
            nan = False
            if prefix == sc.B_PREFIX:
                last = u.rsplit("LAST WORD: ", 1)[1]
                pick = "WAIT" if (self.kind == "qwen3" and last == "먹었어요") or last in ("갔다가",) else "SAFE"
                nan = last == self.nan_last
            elif prefix == sc.C_PREFIX:
                fut = u.rsplit("FUTURE:\n", 1)[1]
                pick = "REVISION" if fut.split()[0] in ("아니", "no") else "STABLE"
                nan = fut.split()[0] == self.nan_fut
            else:
                pick = labels[0]
            res.append(sc.label_result(list(labels), [math.nan if nan else 0.0 if l == pick else -2.0 for l in labels]))
        return res


def test_stage_functions_with_fake_llm():
    pv = sc.prompt_version()
    fake = FakeLLM(bad_first_a=True)
    a = sc.run_stage_a(fake, [KO1, EN1], "qwen3", pv=pv)
    assert [r["status"] for r in a] == ["ok", "ok"] and [r["attempts"] for r in a] == [2, 2]
    assert fake.prompts[-1][-2]["role"] == "assistant" and "out of range" in fake.prompts[-1][-1]["content"]
    assert a[0]["out"]["semantic_boundaries"] == [5, 8] and a[0]["key"] == ["A", "k1", None, "qwen3", pv]
    assert a[0]["n_words"] == 9 and a[0]["word_texts_sha256"] == sc.word_texts_sha256(KO1["words"])
    bad = sc.run_stage_a(FakeLLM(bad_first_a=True), [KO2], "qwen3", retries=0)[0]
    assert bad["status"] == "invalid_json" and bad["out"] is None and "999" in bad["raw"]
    items = sc.stage_b_items([KO1, EN1], a)
    assert [(s["id"], i) for s, i in items] == [("k1", 5), ("k1", 8), ("e1", 2), ("e1", 6)]
    fb = FakeLLM("qwen3")
    b = sc.run_stage_b(fb, items, "qwen3", pv=pv)
    assert [r["decision"] for r in b] == ["SAFE", "WAIT", "SAFE", "SAFE"] and b[0]["margin"] == 2.0
    assert "그리고" not in fb.prompts[0] and "work" not in fb.prompts[2]            # prefix only, no future
    fc = FakeLLM("qwen3")
    c = sc.run_stage_c(fc, sc.stage_c_items([KO1, EN1], a), "qwen3", pv=pv)
    assert [(r["after_word"], r["relation"], r["type"], r["next_candidate"]) for r in c] == [
        (5, "STABLE", "NEW_UNIT", 8), (8, "UNOBSERVED", "", None), (2, "REVISION", "SELF_REPAIR", 6),
        (6, "UNOBSERVED", "", None)]
    assert all(r["status"] == "ok" for r in c)
    assert c[0]["future_n"] == 3 and c[0]["future_end"] is True and fc.calls["score"] == 3   # rel + 2 type groups
    assert fc.suffixes == [('{"relation": "', '",'), ('{"relation": "', '"}'), ('{"relation": "', '"}')]
    assert fb.suffixes == [(sc.B_PREFIX, '"}')]
    t_items = sc.tiebreak_items([KO1, EN1], a, b + sc.run_stage_b(FakeLLM("exaone35"), items[:2], "exaone35"))
    assert [(s["id"], i) for s, i in t_items] == [("k1", 8)]
    fn = FakeLLM("qwen3", nan_last="갔어요")                        # bf16-overflow-like NaN score on one item
    bn = sc.run_stage_b(fn, items, "qwen3", pv=pv)
    assert [r["status"] for r in bn] == ["nonfinite", "ok", "ok", "ok"] and bn[0]["decision"] is None
    sc.encode_rows(bn)                                               # serializable
    cn = sc.run_stage_c(FakeLLM("qwen3", nan_fut="그리고"), sc.stage_c_items([KO1], a), "qwen3", pv=pv)
    assert [(r["status"], r["relation"], r["type"]) for r in cn] == [("nonfinite", None, ""), ("ok", "UNOBSERVED", "")]


def _pipeline(streams):
    a = sc.run_stage_a(FakeLLM(), [s for s in streams if s["id"] != "k3"], "qwen3")
    items = sc.stage_b_items(streams, a)
    ko = [it for it in items if it[0]["lang"] == "Korean"]
    en = [it for it in items if it[0]["lang"] == "English"]
    b = sc.run_stage_b(FakeLLM("exaone35"), ko, "exaone35") + sc.run_stage_b(FakeLLM("qwen3"), items, "qwen3") \
        + sc.run_stage_b(FakeLLM("gptoss"), en, "gptoss")
    c = sc.run_stage_c(FakeLLM(), sc.stage_c_items(streams, a), "qwen3")
    t = sc.run_stage_b(FakeLLM("gptoss"), sc.tiebreak_items(streams, a, b), "gptoss", stage="T")
    return a, b, c, t


def _grades(labels):
    return {r["id"]: [(x["after_word"], x["grade"], x["stageA"]) for x in r["candidates"]] for r in labels}


def test_build_labels_grades_and_stats():
    streams = [KO1, KO2, EN1, mk_stream("k3", "Korean", ["네"])]
    a, b, c, t = _pipeline(streams)
    labels, st = sc.build_labels(streams, a, b, c, t)
    assert _grades(labels) == {"k1": [(5, "A", True), (8, "B", True)], "k2": [(4, "A", True)],
                               "e1": [(2, "N", True), (6, "A", True)]}   # default: only Stage A candidates are rows
    assert not st["disfl_negatives"] and all("N_disfl_only" not in v for v in st["by_lang"].values())
    k1 = {x["after_word"]: x for x in labels[0]["candidates"]}
    assert k1[8]["resolved_by_judge"] and k1[8]["future_unobserved"] and k1[8]["B"] == {
        "exaone35": "SAFE", "gptoss": "SAFE", "qwen3": "WAIT"}
    assert k1[5]["C"] == {"relation": "STABLE", "type": "NEW_UNIT"} and k1[5]["B_margin"]["qwen3"] == 2.0
    assert k1[8]["C"] == {"relation": "UNOBSERVED", "type": ""}
    e1 = {x["after_word"]: x for x in labels[2]["candidates"]}
    assert e1[2]["why"] == "C_revision" and labels[0]["turn_end"] is True           # Stage A reparandum is not an N reason
    assert st["streams"] == {"total": 4, "labeled": 3, "stageA_missing": 1}
    assert st["by_lang"]["Korean"]["A"] == 2 and st["by_lang"]["Korean"]["B"] == 1
    assert st["by_lang"]["Korean"]["future_unobserved_A"] == 1
    assert st["agreement"]["Korean/exaone35~qwen3"] == {"n": 3, "agree": 2, "kappa": 0.0}
    assert st["C_type"]["English"] == {"REVISION/SELF_REPAIR:N": 1, "UNOBSERVED/-:A": 1}
    assert st["row_status"]["B/exaone35"] == {"ok": 3} and st["tok_mismatch"] == {}
    json.dumps(st, allow_nan=False)
    # opt-in ablation: disfluency word ends that Stage A did not propose become stageA=false N rows
    lab, st2 = sc.build_labels(streams, a, b, c, t, disfl_negatives=True)
    assert _grades(lab) == {"k1": [(0, "N", False), (2, "N", False), (3, "N", False), (5, "A", True), (8, "B", True)],
                            "k2": [(2, "N", False), (4, "A", True)],
                            "e1": [(2, "N", True), (3, "N", False), (6, "A", True)]}   # 4 = end of repair [3, 4]
    assert st2["by_lang"]["English"]["N_disfl_only"] == 1
    assert all(x["C"] == {"relation": "MISSING", "type": ""} for r in lab for x in r["candidates"] if not x["stageA"])
    lab2, _ = sc.build_labels(streams, a, b, c, t, strict=False, turn_end=False)
    assert [(x["after_word"], x["grade"]) for x in lab2[0]["candidates"]] == [(5, "A"), (8, "A")]
    assert lab2[0]["turn_end"] is False
    for r in lab + labels:   # schema: C always {relation, type: str}
        for x in r["candidates"]:
            assert set(x["C"]) == {"relation", "type"} and isinstance(x["C"]["type"], str)
            assert x["C"]["relation"] in ("STABLE", "REVISION", "UNOBSERVED", "MISSING")


def test_build_labels_tok_mismatch_missing_c_and_continuation():
    streams = [KO1, KO2, EN1]
    a, b, c, t = _pipeline(streams)
    bad_b = [dict(r, tok_ok=False) if (r["id"], r["after_word"], r["judge"]) == ("k2", 4, "qwen3") else r for r in b]
    lab, st = sc.build_labels(streams, a, bad_b, c, t)
    k2 = lab[1]["candidates"][0]
    assert (k2["grade"], k2["why"], k2["B"]) == ("B", "B_missing:qwen3", {"exaone35": "SAFE"})   # strict: dropped
    assert st["tok_mismatch"] == {"B/qwen3": 1}
    assert sc.build_labels(streams, a, bad_b, c, t, strict=False)[0][1]["candidates"][0]["grade"] == "A"
    bad_c = [dict(r, type_tok_ok=False) if (r["id"], r["after_word"]) == ("k1", 5) else r for r in c]
    lab, st = sc.build_labels(streams, a, b, bad_c, t)
    assert lab[0]["candidates"][0]["C"] == {"relation": "MISSING", "type": ""} and lab[0]["candidates"][0]["grade"] == "B"
    assert st["tok_mismatch"] == {"C/qwen3": 1} and st["C_relation"]["Korean"]["MISSING"] == 1
    lab, _ = sc.build_labels(streams, a, b, [r for r in c if r["id"] != "k2"], t)    # no Stage C row at all
    assert lab[1]["candidates"][0]["C"] == {"relation": "MISSING", "type": ""}
    assert lab[1]["candidates"][0]["why"] == "C_missing"
    cont = [dict(r, relation="REVISION", type="CONTINUATION") if (r["id"], r["after_word"]) == ("k1", 5) else r for r in c]
    lab, st = sc.build_labels(streams, a, b, cont, t)
    assert (lab[0]["candidates"][0]["grade"], lab[0]["candidates"][0]["why"]) == ("B", "C_revision_masked:CONTINUATION")
    assert st["C_type"]["Korean"]["REVISION/CONTINUATION:B"] == 1 and st["c_mask_types"] == ["CONTINUATION"]
    assert sc.build_labels(streams, a, b, cont, t, c_mask_types=())[0][0]["candidates"][0]["grade"] == "N"


def test_build_labels_refuses_mismatched_words_or_stage_a():
    streams = [KO1, KO2, EN1]
    a, b, c, t = _pipeline(streams)
    shifted = dict(KO1, words=[dict(KO1["words"][0], text="음")] + [dict(w, i=w["i"] + 1) for w in KO1["words"]])
    with pytest.raises(ValueError, match="n_words 9 != 10"):             # rebuilt words.jsonl, indices shifted
        sc.build_labels([shifted, KO2, EN1], a, b, c, t)
    renamed = dict(KO1, words=[dict(w, text="X") if w["i"] == 3 else w for w in KO1["words"]])
    with pytest.raises(ValueError, match="word_texts_sha256"):           # same length, different words
        sc.build_labels([renamed, KO2, EN1], a, b, c, t)
    a2 = [dict(r, out=dict(r["out"], semantic_boundaries=[5, 6, 8])) if r["id"] == "k1" else r for r in a]
    with pytest.raises(ValueError, match="next_candidate 8, Stage A gives 6"):   # C windows from another Stage A file
        sc.build_labels(streams, a2, b, c, t)


# ───────────── CLI end to end (fake model injected) ─────────────
def test_cli_pipeline_resume_and_errors(tmp_path, monkeypatch, capsys):
    from experiments import semcommit_teacher as T
    W = tmp_path / "words.jsonl"
    W.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in [KO1, KO2, EN1, KO1]))   # KO1 repeated
    made = []
    monkeypatch.setattr(T, "LLM", lambda path, kind, **kw: made.append(FakeLLM(kind)) or made[-1])
    base = ["--words", str(W), "--device", "cpu", "--batch-size", "2"]
    A, C, TB, L, S = (tmp_path / n for n in ("A.jsonl", "C.jsonl", "T.jsonl", "labels.jsonl", "stats.json"))
    T.main(["stageA", *base, "--model", "m", "--kind", "qwen3", "--out", str(A)])
    n_a = len(A.read_text().splitlines())
    T.main(["stageA", *base, "--model", "m", "--kind", "qwen3", "--out", str(A)])       # resume: nothing pending
    assert len(A.read_text().splitlines()) == n_a == 3 and "pending=0" in capsys.readouterr().out
    with pytest.raises(ValueError, match="params"):
        T.main(["stageA", *base, "--model", "m", "--kind", "qwen3", "--out", str(A), "--max-new-tokens", "9"])
    Bs = []
    for kind, lf in [("exaone35", "Korean"), ("qwen3", "all"), ("gptoss", "English")]:
        Bs.append(tmp_path / f"B.{kind}.jsonl")
        args = ["stageB", *base, "--model", "m", "--kind", kind, "--lang-filter", lf, "--stageA", str(A),
                "--out", str(Bs[-1]), "--flush-every", "1"]
        if kind == "exaone35":
            FakeLLM.fail_first = 1                                   # first block fails → explicit error rows, exit 2
            with pytest.raises(SystemExit) as e:
                T.main(args)
            assert e.value.code == 2 and "INCOMPLETE: 1 inference_error" in capsys.readouterr().out
        else:
            T.main(args)
    ex = sc.read_jsonl(Bs[0])
    assert [r["status"] for r in ex].count("inference_error") == 1
    T.main(["stageB", *base, "--model", "m", "--kind", "exaone35", "--lang-filter", "Korean", "--stageA", str(A),
            "--out", str(Bs[0])])                                   # resume retries the failed item only
    ex = sc.read_jsonl(Bs[0])
    assert len(ex) == 4 and sum(r["status"] == "ok" for r in ex) == 3
    assert "COMPLETE" in capsys.readouterr().out
    T.main(["stageC", *base, "--model", "m", "--kind", "qwen3", "--stageA", str(A), "--out", str(C)])
    A2 = tmp_path / "A2.jsonl"                                      # Stage A re-run elsewhere with other boundaries
    A2.write_text("".join(json.dumps(dict(r, out=dict(r["out"], semantic_boundaries=[5, 6, 8])) if r["id"] == "k1"
                                     else r, ensure_ascii=False) + "\n" for r in sc.read_jsonl(A)))
    with pytest.raises(ValueError, match="params"):                  # C windows depend on Stage A → refuse resume
        T.main(["stageC", *base, "--model", "m", "--kind", "qwen3", "--stageA", str(A2), "--out", str(C)])
    T.main(["tiebreak", *base, "--model", "m", "--kind", "gptoss", "--stageA", str(A), "--stageB", *map(str, Bs),
            "--out", str(TB)])
    assert [(r["stage"], r["id"], r["after_word"], r["judge"]) for r in sc.read_jsonl(TB)] == [("T", "k1", 8, "gptoss")]
    labels, stats = T.main(["grade", *base, "--stageA", str(A), "--stageB", *map(str, Bs), "--stageC", str(C),
                            "--tiebreak", str(TB), "--out", str(L), "--stats", str(S)])
    assert [json.loads(x) for x in L.read_text().splitlines()] == labels and len(labels) == 3
    st = json.loads(S.read_text())
    assert st["strict"] is True and st["by_lang"]["Korean"]["A"] == 2 and st["by_lang"]["English"]["A"] == 1
    assert str(A) in st["inputs"] and st["streams"] == {"total": 4, "labeled": 3, "dup_id_skipped": 1}
    assert st["prompt_version"] == sc.prompt_version() and st["provenance"][str(C)]["stage"] == "C"
    assert not st["disfl_negatives"] and all(x["stageA"] for r in labels for x in r["candidates"])
    grade = ["grade", "--device", "cpu", "--stageA", str(A), "--stageB", *map(str, Bs), "--stageC", str(C),
             "--tiebreak", str(TB), "--out", str(tmp_path / "L2.jsonl"), "--stats", str(tmp_path / "S2.json")]
    W2 = tmp_path / "words2.jsonl"                                  # rebuilt words: k1 gains a leading '음'
    k1b = dict(KO1, words=[dict(KO1["words"][0], text="음")] + [dict(w, i=w["i"] + 1) for w in KO1["words"]])
    W2.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in [k1b, KO2, EN1]))
    with pytest.raises(ValueError, match="words sha256"):
        T.main([*grade, "--words", str(W2)])
    with pytest.raises(ValueError, match="fingerprint stage 'B', expected 'C'"):   # swapped arguments
        T.main([*grade[:grade.index("--stageC") + 1], str(Bs[1]), *grade[grade.index("--stageC") + 2:], "--words", str(W)])
    C3 = tmp_path / "C3.jsonl"                                      # same stage, other prompt version
    C3.write_text("".join(json.dumps(dict(r, prompt_version="semcommit-prompt-v0.0+deadbeef"), ensure_ascii=False) + "\n"
                          for r in sc.read_jsonl(C)))
    fp3 = json.loads(C.with_suffix(".fingerprint.json").read_text())
    C3.with_suffix(".fingerprint.json").write_text(json.dumps(dict(fp3, prompt_version="semcommit-prompt-v0.0+deadbeef")))
    g3 = [*grade[:grade.index("--stageC") + 1], str(C3), *grade[grade.index("--stageC") + 2:], "--words", str(W)]
    with pytest.raises(ValueError, match="mix prompt versions"):
        T.main(g3)
    C3.with_suffix(".fingerprint.json").write_text(json.dumps(fp3))
    with pytest.raises(ValueError, match="rows with prompt_version"):
        T.main(g3)
    C3.with_suffix(".fingerprint.json").unlink()
    with pytest.raises(ValueError, match="no fingerprint sidecar"):
        T.main(g3)
    FakeLLM.fail_first = 0


def test_cli_failures_exit_nonzero_and_abort(tmp_path, monkeypatch, capsys):
    from experiments import semcommit_teacher as T
    W = tmp_path / "words.jsonl"
    W.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in [KO1, KO2, EN1]))
    monkeypatch.setattr(T, "LLM", lambda path, kind, **kw: FakeLLM(kind))
    base = ["--words", str(W), "--device", "cpu", "--model", "m", "--kind", "qwen3"]
    A = tmp_path / "A.jsonl"
    T.main(["stageA", *base, "--out", str(A)])
    monkeypatch.setattr(FakeLLM, "fail_first", 100)                 # every block fails (e.g. a template error)
    B = tmp_path / "B.jsonl"
    with pytest.raises(SystemExit) as e:
        T.main(["stageB", *base, "--stageA", str(A), "--out", str(B), "--flush-every", "1", "--max-failed-blocks", "2"])
    out = capsys.readouterr().out
    assert e.value.code == 2 and "ABORT: 2 consecutive" in out and "3 item(s) not run" in out
    assert [r["status"] for r in sc.read_jsonl(B)] == ["inference_error"] * 2
    monkeypatch.setattr(FakeLLM, "fail_first", 0)
    real = sc.run_stage_b                                           # a NaN reaching serialization → error rows, no crash
    monkeypatch.setattr(sc, "run_stage_b", lambda llm, b, judge, bs, stage, pv: [
        dict(r, margin=math.nan) if r["after_word"] == 5 else r for r in real(llm, b, judge, bs, stage, pv)])
    B2 = tmp_path / "B2.jsonl"
    with pytest.raises(SystemExit):
        T.main(["stageB", *base, "--stageA", str(A), "--out", str(B2), "--flush-every", "2"])
    rows = sc.read_jsonl(B2)
    assert [r["status"] for r in rows] == ["inference_error", "inference_error", "ok", "ok", "ok"]
    assert "Out of range float" in rows[0]["error"]
    monkeypatch.setattr(sc, "run_stage_b", real)
    T.main(["stageB", *base, "--stageA", str(A), "--out", str(B2)])        # resume retries only the failed block
    assert sum(r["status"] == "ok" for r in sc.read_jsonl(B2)) == 5
    B3 = tmp_path / "B3.jsonl"                                      # non-finite scores: written, counted, not retried
    monkeypatch.setattr(T, "LLM", lambda path, kind, **kw: FakeLLM(kind, nan_last="왔어요"))
    T.main(["stageB", *base, "--stageA", str(A), "--out", str(B3)])
    out = capsys.readouterr().out
    assert "WARNING: 1 row(s) with non-finite" in out and "COMPLETE" in out
    assert [r["status"] for r in sc.read_jsonl(B3)].count("nonfinite") == 1
    T.main(["stageB", *base, "--stageA", str(A), "--out", str(B3)])
    assert "pending=0" in capsys.readouterr().out


def test_cli_batch_size_defaults():
    from experiments import semcommit_teacher as T
    common = ["--words", "w", "--model", "m", "--stageA", "a", "--out", "o"]
    assert T.parse(["stageB", *common, "--kind", "gptoss"]).batch_size == 1
    assert T.parse(["stageB", *common, "--kind", "qwen3"]).batch_size == 8
    assert T.parse(["stageB", *common, "--kind", "gptoss", "--batch-size", "4"]).flush_every == 16
    g = T.parse(["grade", "--words", "w", "--stageA", "a", "--stageB", "b", "--stageC", "c", "--out", "o", "--stats", "s"])
    assert g.disfl_negatives is False and g.c_mask_types == "CONTINUATION"


# ───────────── real HF path on a tiny random model (skipped on old transformers) ─────────────
def _tiny_hf():
    tr = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    if tuple(int(x) for x in tr.__version__.split(".")[:2]) < (4, 50):
        pytest.skip(f"transformers {tr.__version__} < 4.50")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tk = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tk.pre_tokenizer, tk.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False), decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk)
    tok.add_special_tokens({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
                            "additional_special_tokens": ["<|im_start|>"]})
    tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                         "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and "
                         "enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, bos_token_id=None)
    model = LlamaForCausalLM(cfg).eval()
    gc = GenerationConfig(do_sample=True, temperature=5.0, top_p=0.5, eos_token_id=tok.eos_token_id,
                          pad_token_id=tok.pad_token_id)
    gc.transformers_version = tr.__version__          # ≥4.50 → model defaults would override a plain do_sample=False
    model.generation_config = gc
    return sc.LLM(None, "qwen3", device="cpu", model=model, tokenizer=tok)


def test_hf_generate_json_greedy_matches_manual_decode():
    llm = _tiny_hf()
    prompts = [llm.build_prompt([{"role": "user", "content": c}]) for c in ("hi", "a longer user message 안녕")]
    assert prompts[0].endswith("<think>\n\n</think>\n\n")
    torch.manual_seed(1)
    out = llm.generate_json(prompts, max_new_tokens=6, batch_size=2)
    torch.manual_seed(2)
    assert llm.generate_json(prompts, max_new_tokens=6, batch_size=1) == out        # greedy, batch-invariant
    eos = set(llm.eos_ids())
    for p, o in zip(prompts, out):
        ids, gen = llm.encode(p), []
        with torch.no_grad():
            for _ in range(6):
                nxt = int(llm.model(torch.tensor([ids + gen])).logits[0, -1].argmax())
                gen.append(nxt)
                if nxt in eos:
                    break
        keep, trunc = sc.cut_at_eos(gen, eos)
        assert o["text"] == llm.tok.decode(keep, skip_special_tokens=True) and o["truncated"] == trunc
        assert o["json"] is None and o["error"]


def test_hf_score_labels_matches_unpadded_forward():
    llm = _tiny_hf()
    prompts = [llm.build_prompt([{"role": "user", "content": c}]) for c in ("x", "some longer content", "mid")]
    res = llm.score_labels(prompts, sc.B_LABELS, prefix=sc.B_PREFIX, batch_size=4)
    for p, r in zip(prompts, res):
        for l in sc.B_LABELS:
            ctx, cont = llm.encode(p + sc.B_PREFIX), llm.encode(l + sc.B_SUFFIX)
            with torch.no_grad():
                lsm = llm.model(torch.tensor([ctx + cont])).logits[0].float().log_softmax(-1)
            exp = sum(lsm[len(ctx) + k - 1, t].item() for k, t in enumerate(cont))
            assert r["logprobs"][l] == pytest.approx(exp, abs=1e-3)


# ───────────── v0.2: Stage A [index, word] anchors ─────────────
def test_snap_boundaries_exact_snapped_dropped_bare():
    words = ["she", "said", "in", "a", "clear", "sweet", "voice", "i'm", "very", "glad", "to", "see", "you"]
    obj = {"semantic_boundaries": [[6, "voice"], [9, "you"], [8, "voice"], [3, "banana"], 12, {"i": 1, "w": "Said"}]}
    out, info = sc.snap_boundaries(obj, words)
    assert out["semantic_boundaries"] == [6, 12, 6, 12, 1] and info["exact"] == 2 and info["snapped"] == 2 and info["dropped"] == 1 and info["bare"] == 1
    assert info["moves"] == [[9, 12], [8, 6]]
    assert sc.validate_stage_a(out, len(words))["semantic_boundaries"] == [1, 6, 12]          # dedup + sort after snapping
    far, inf2 = sc.snap_boundaries({"semantic_boundaries": [[0, "you"]]}, words)                # 12 is beyond ±4 → dropped
    assert far["semantic_boundaries"] == [] and inf2["dropped"] == 1
    tie, _ = sc.snap_boundaries({"semantic_boundaries": [[5, "a"]]}, ["a", "x", "x", "a", "x", "x", "x", "a"])   # d=2 both sides → earlier
    assert tie["semantic_boundaries"] == [3]
    ko, _ = sc.snap_boundaries({"semantic_boundaries": [[2, "같아요."]]}, ["것", "같아요", "그리고"])        # punctuation-insensitive
    assert ko["semantic_boundaries"] == [1]
    with pytest.raises(ValueError):
        sc.snap_boundaries({"semantic_boundaries": [["2", "x"]]}, words)


def test_run_stage_a_snaps_anchor_output():
    class AnchorLLM:
        def build_prompt(self, m): return "p"
        def generate_json(self, prompts, max_new_tokens=0, batch_size=0):
            return [{"text": '{"semantic_boundaries": [[4, "갔어요"], [7, "먹었어요"]], "fillers": [[0, 0]]}', "truncated": False}]
    row = sc.run_stage_a(AnchorLLM(), [KO1], "qwen3", pv="t")[0]
    assert row["status"] == "ok" and row["out"]["semantic_boundaries"] == [5, 8] and row["snap"]["snapped"] == 2


# ───────────── recipe v0.3 ─────────────
def _two_seg():
    s = mk_stream("e2", "English", ["i", "went", "home", "and", "then", "we", "slept", "well"], {2: ["punct_final"], 7: ["punct_final"]})
    for w in s["words"]: w["seg"] = 0 if w["i"] <= 2 else 1
    return s


def test_extra_candidates_sets_and_items():
    s = _two_seg()
    assert sc.extra_candidates(s) == {7: ["last", "seg_end", "punct_final"], 2: ["seg_end", "punct_final"]}
    assert sc.extra_candidates(s, ("last",)) == {7: ["last"]} and sc.extra_candidates(dict(s, words=[])) == {}
    a = sc.run_stage_a(FakeLLM(), [s, KO1], "qwen3")                                    # Stage A: 'home' (2) · 갔어요/먹었어요 (5, 8)
    cs = sc.candidate_sets([s, KO1], a, sc.EXTRA_SOURCES)
    assert cs["e2"][0] == [2, 7] and cs["e2"][1] == {2} and cs["e2"][2] == {2: ["stageA", "seg_end", "punct_final"], 7: ["last", "seg_end", "punct_final"]}
    assert cs["k1"][0] == [5, 8] and cs["k1"][2][8] == ["stageA", "last", "seg_end"]
    assert sc.candidate_sets([s], a)["e2"][0] == [2]                                    # extra 없음 = v0.2 후보
    assert [i for _, i in sc.stage_b_items([s, KO1], a, sc.EXTRA_SOURCES)] == [2, 7, 5, 8]
    assert [i for _, i in sc.stage_b_items([s, KO1], a, sc.EXTRA_SOURCES, only_extra=True)] == [7]
    assert [(i, nc) for _, i, nc in sc.stage_c_items([s], a, sc.EXTRA_SOURCES)] == [(2, None), (7, None)]   # window stops at the next *Stage A* boundary
    assert [(i, nc) for _, i, nc in sc.stage_c_items([KO1], a)] == [(5, 8), (8, None)]                      # v0.2 rows unchanged


def test_label_probs_features_and_grade_v3():
    assert sc.label_probs({"SAFE": 0.0, "WAIT": -2.0, "UNCERTAIN": -2.0})["SAFE"] == pytest.approx(1 / (1 + 2 * math.exp(-2)))
    assert sc.label_probs(None) is None and sc.label_probs({"SAFE": float("nan"), "WAIT": 0.0}) is None
    s = _two_seg(); b = {"qwen3": dict(logprobs={"SAFE": 0.0, "WAIT": -2.0, "UNCERTAIN": -2.0}), "exaone35": dict(logprobs={"SAFE": -2.0, "WAIT": 0.0, "UNCERTAIN": -2.0})}
    f = sc.candidate_features(s, 7, b, dict(relation="UNOBSERVED"), ("qwen3", "exaone35"), ["last"])
    assert f["punct"] and f["p_rev"] == 0.0 and f["p_safe_mean"] == pytest.approx((0.787 + 0.1065) / 2, abs=1e-3) and f["sources"] == ["last"]
    assert sc.grade_v3(f, "English") == ("A", "v3_punct")                                               # 구두점 문장 끝: 판정자가 반대해도(기본 임계 0) A
    th = {"English": {"punct": {"p_safe": 0.6, "p_rev": 0.5}, "other": None}}
    assert sc.grade_v3(f, "English", th) == ("B", "v3_punct+p_safe_low")
    f2 = sc.candidate_features(s, 5, b, dict(relation="STABLE", logprobs={"REVISION": -2.0, "STABLE": 0.0}), ("qwen3", "exaone35"))
    assert not f2["punct"] and f2["p_rev"] == pytest.approx(math.exp(-2) / (1 + math.exp(-2)))
    assert sc.grade_v3(f2, "English") == ("B", "v3_other+off")                                           # 기본(v0.3.1): 그 외 가지 꺼짐
    assert sc.grade_v3(f2, "English", {"English": {"punct": None, "other": {"p_safe": 0.4, "p_rev": 0.2}}}) == ("A", "v3_other")
    assert sc.grade_v3(f2, "English", {"English": {"punct": None, "other": {"p_safe": 0.4, "p_rev": 0.1}}}) == ("B", "v3_other+p_rev_high")
    assert sc.grade_v3(dict(f2, disfluency="tag_filler"), "English") == ("N", "disfl_tag_filler")      # N = 사람 전사 표지만
    assert sc.grade_v3(dict(f2, disfluency="reparandum"), "English") == ("B", "disflA_reparandum")
    assert sc.grade_v3(sc.candidate_features(s, 5, {"qwen3": b["qwen3"]}, None, ("qwen3", "exaone35")), "English")[1] == "missing:exaone35,C"


def test_response_unit_branch_and_threshold_fallback():
    W = lambda toks: [dict(i=i, text=x.rstrip("."), end_time=0.3 * (i + 1), tags=["punct_final"] if x.endswith(".") else []) for i, x in enumerate(toks)]
    ko = W(["아", "네.", "저는", "좋아요."])
    assert sc.response_unit(ko, 1, "Korean") and not sc.response_unit(ko, 3, "Korean")          # 대답어만의 단위 + 뒤에 말 / 스트림 끝은 일반 문장 끝
    assert not sc.response_unit(W(["네."]), 0, "Korean")                                         # 대답어만으로 끝난 발화 = 확정(일반 구두점 가지)
    assert not sc.response_unit(W(["그건", "아니야.", "다시"]), 1, "Korean")                     # 단위에 대답어 아닌 말
    assert sc.response_unit(W(["좋아.", "네.", "그럼"]), 1, "Korean")                             # 단위 = 앞 구두점 뒤부터
    en = W(["No.", "I", "think", "so."])
    assert sc.response_unit(en, 0, "English") and not sc.response_unit(en, 0, "Korean") and not sc.response_unit(en, 0, None)
    f = sc.candidate_features(dict(id="k", lang="Korean", words=ko), 1, {}, None, ("exaone35",))
    assert f["resp_head"] and f["punct"] and sc.branch_of(f) == "punct_resp"
    f = dict(f, p_safe={"exaone35": 0.9}, p_safe_mean=0.9, p_rev=0.0)
    assert sc.branch_of(dict(f, resp_head=False)) == "punct" and sc.branch_of(dict(f, punct=False, resp_head=False)) == "other"
    old = {"Korean": {"punct": {"p_safe": 0.0, "p_rev": 1.01}, "other": None}}                  # v0.3 / v0.3.1 임계값 파일: punct_resp 키 없음 → 구두점 임계값
    assert sc.grade_v3(f, "Korean", old) == ("A", "v3_punct_resp")
    assert sc.grade_v3(f, "Korean", {"Korean": dict(old["Korean"], punct_resp=None)}) == ("B", "v3_punct_resp+off")
    assert sc.grade_v3(f, "Korean") == ("B", "v3_punct_resp+off") and sc.grade_v3(f, "English") == ("A", "v3_punct_resp")   # 기본 = v0.3.2 조정값


def test_build_labels_v3_pipeline():
    s = _two_seg(); streams = [s, KO1, KO2]
    a = sc.run_stage_a(FakeLLM(), streams, "qwen3")
    items = sc.stage_b_items(streams, a, sc.EXTRA_SOURCES)
    b = sc.run_stage_b(FakeLLM("qwen3"), items, "qwen3") + sc.run_stage_b(FakeLLM("exaone35"), items, "exaone35")
    c = sc.run_stage_c(FakeLLM(), sc.stage_c_items(streams, a, sc.EXTRA_SOURCES), "qwen3")
    J = {"English": ("qwen3", "exaone35"), "Korean": ("exaone35", "qwen3")}
    labels, st = sc.build_labels_v3(streams, a, b, c, judges=J, neg_rules=())                         # 규칙 음성 없음 = v0.3–v0.3.2
    g = {r["id"]: [(x["after_word"], x["grade"], x["stageA"], x["why"]) for x in r["candidates"]] for r in labels}
    assert g["e2"] == [(2, "A", True, "v3_punct"), (7, "A", False, "v3_punct")]                         # 7 = 추가 후보(마지막·구간 끝·구두점)
    assert g["k2"] == [(4, "A", True, "v3_punct")]
    assert [x[1] for x in g["k1"]] == ["B", "B"] and st["recipe"] == sc.RECIPE_V3                      # 한국어 그 외 가지 꺼짐(기본) → B
    labels2, _ = sc.build_labels_v3(streams, a, b, c, judges=J, neg_rules=(),
                                    thresholds={"English": sc.V3_THRESHOLDS["English"], "Korean": {"punct": None, "other": {"p_safe": 0.4, "p_rev": 0.2}}})
    assert [x["grade"] for x in labels2[1]["candidates"]] == ["A", "A"] and st["sources"]["English"]["last:A"] == 1
    # v0.3.3 기본: 규칙 음성 — 스트림 머리 대답어(어) · 연결어미(그리고 · 갔다가)는 후보가 아니어도 N 행(stageA 거짓)
    labels3, st3 = sc.build_labels_v3(streams, a, b, c, judges=J)
    g3 = {r["id"]: [(x["after_word"], x["grade"], x["stageA"], x["why"]) for x in r["candidates"]] for r in labels3}
    assert g3["e2"] == g["e2"] and st3["neg_rules"] == list(sc.NEG_RULES)
    assert g3["k1"] == [(0, "N", False, "rule_reply_prefix"), (5, "B", True, "v3_other+off"), (6, "N", False, "rule_conn_mid"), (8, "B", True, "v3_other+off")]
    assert g3["k2"] == [(1, "N", False, "rule_conn_mid"), (4, "A", True, "v3_punct")]
    bad = [dict(r, next_candidate=99) if r["after_word"] == 7 else r for r in c]
    with pytest.raises(ValueError, match="next Stage A boundary"): sc.build_labels_v3(streams, a, b, bad, judges=J)


def test_rule_negatives():
    W = lambda toks, tags=None: dict(id="s", lang="Korean", words=[dict(i=i, text=x.rstrip("."), tags=(["punct_final"] if x.endswith(".") else []) + ((tags or {}).get(i) or []))
                                                              for i, x in enumerate(toks)])
    assert sc.rule_negatives(W(["어", "네", "저는", "갔어요."])) == {0: "reply_prefix", 1: "reply_prefix"}
    assert sc.rule_negatives(W(["맞아", "맞아"])) == {0: "reply_prefix"}                                 # 스트림 끝 대답어는 음성 아님
    assert sc.rule_negatives(W(["밥", "먹고", "갔는데", "비가", "와서"])) == {1: "conn_mid", 2: "conn_mid", 4: "conn_final"}
    assert sc.rule_negatives(W(["어제", "갔는데"])) == {}                                                # 말끝 -는데 = 골드 AMBIG
    assert sc.rule_negatives(W(["엄청", "좋더라고"])) == {} and sc.rule_negatives(W(["간다고"])) == {}  # 문장 끝 -더라고 · 인용 -다고
    assert sc.rule_negatives(W(["비가", "와서."])) == {}                                                 # 구두점이 있으면 규칙 밖
    assert sc.rule_negatives(W(["먹고", "와서"]), ()) == {} and sc.rule_negatives(W(["먹고", "와서"]), ("conn_mid",)) == {0: "conn_mid"}
    en = dict(id="e", lang="English", words=[dict(i=i, text=x, tags=[]) for i, x in enumerate(["yeah", "so", "we", "went"])])
    assert sc.rule_negatives(en) == {0: "reply_prefix"}                                                 # 영어는 대답어 머리만
