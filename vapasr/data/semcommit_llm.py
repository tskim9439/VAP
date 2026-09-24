"""Semantic-commit LLM relabeling v0 (plan §5–11, §16, §23): prompt, JSON 검증, 모델 어댑터, Stage A/B/C 로직, 등급.

원칙(정본 plan): precision-first(애매하면 commit 하지 않음), LLM 은 word index 만 반환하고 transcript 를 다시 쓰지 않는다,
Stage B 는 prefix 만 본다(미래 금지), SEM_END = A ∧ B ∧ ¬C. 표준 라이브러리만 import 하며 pydantic 은 있으면 쓰고
(rack4 vapasr env), 없으면 같은 검사를 하는 stdlib 검증으로 대체한다. torch/transformers 는 LLM 사용 시점에만 import.

Grades per candidate boundary (candidate = Stage A boundary after word i):
  'A' insert <SEM_END>;  'B' uncertain → no insert, label −100 at the decision position;
  'N' hard negative (Stage C REVISION of a future-repair type, all primary Stage B judges WAIT, or a disfluency conflict)
  → no insert, loss weight overridden to hardneg_weight. Stage C REVISION/CONTINUATION (the future only continues the
  sentence: Stage A and C disagree, and the streaming model cannot see it at δ=2–4 chunks) is masked as 'B'
  (c_mask_types). Non-candidate word ends stay implicit negatives with default weight; stageA=false disfluency N rows
  are an opt-in ablation (disfl_negatives) because the dataset gives every N row hardneg_weight.

Row files (JSONL, append-only, resumable): every row carries key=[stage, id, after_word, judge, prompt_version];
the sidecar <out>.fingerprint.json binds model path/kind, prompt hash (texts + rendered probe prompts), chat-template
hash, template kwargs, generation config and the words.jsonl digest. A resume with a different fingerprint is refused;
grade refuses inputs whose sidecars name another words.jsonl or mix prompt versions (check_provenance).
"""
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

PROMPT_VERSION = "semcommit-prompt-v0.2"   # v0.2 (2026-09-23): Stage A [index, word] anchors + snapping; Stage C REVISION narrowed to plan §10 (continuation/added detail = STABLE)
KINDS = {"qwen3": dict(template_kwargs={"enable_thinking": False}, trust_remote_code=False),
         "exaone35": dict(template_kwargs={}, trust_remote_code=True),
         "gptoss": dict(template_kwargs={"reasoning_effort": "low"}, trust_remote_code=False)}
PRIMARY_JUDGES = {"Korean": ("exaone35", "qwen3"), "English": ("qwen3", "gptoss")}
TIEBREAK_JUDGE = {"Korean": "gptoss"}
# bf16 dequant ≈ 42 GB > A100-40GB → CPU offload. accelerate reserves the largest offloaded layer inside the GPU cap and
# transformers 4.57 gpt-oss runs all 32 experts densely with eager attention → leave ~10 GB for activations
# (unmeasured; check peak memory in the rack4 smoke). The CLI also defaults --batch-size 1 for gptoss.
GPTOSS_MAX_MEMORY = {0: "30GiB", "cpu": "120GiB"}
B_LABELS = ("WAIT", "UNCERTAIN", "SAFE")   # conservative first: an exact score tie never yields SAFE/STABLE
C_LABELS = ("REVISION", "STABLE")
C_TYPES = {"STABLE": ("NEW_UNIT", "ADDITIONAL_INFORMATION"),
           "REVISION": ("SELF_REPAIR", "QUALIFICATION", "RESTART")}   # v0.2: CONTINUATION removed (it adds, not revises — plan §10)
B_PREFIX, C_PREFIX = '{"decision": "', '{"relation": "'
# Scored continuation = label + the terminator the full answer would have: Qwen/o200k BPE merge '"}' and '",' into one
# token, so a bare '"' would score an off-distribution token (checked per prompt → row field tok_ok).
B_SUFFIX, C_REL_SUFFIX, C_TYPE_SUFFIX = '"}', '",', '"}'
C_TYPE_MID = '", "type": "'   # type scoring prefix = C_PREFIX + relation + C_TYPE_MID
C_MASK_TYPES = ("CONTINUATION",)   # Stage C REVISION types graded B (masked) instead of N
HINTS = {"filler": "filler", "rep": "rep", "unclear": "unclear", "punct_final": ".", "punct_comma": ","}
DISFL_TAGS = ("filler", "rep", "unclear")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ───────────────────────── prompts ─────────────────────────
RULES = """You label SEMANTIC COMMIT points in speech transcripts for a streaming speech recognizer. A commit after a word tells the downstream system that everything said so far is a finished, stable unit of meaning.
Commit after a word only if BOTH hold:
- COMPLETE: the words since the previous commit form a self-contained unit of meaning (statement, question, request, answer, or a pragmatically complete fragment).
- STABLE: what follows is very unlikely to correct, cancel or reinterpret it.
Rules:
1. The transcript is fixed. Never rewrite, correct, reorder or delete words. Refer to words only by index.
2. Commit only at semantic completion, not at every pause or punctuation mark.
3. No commit where a correction is likely to follow ("내일... 아니 오늘", "Friday, no, Thursday").
4. Silence or hesitation alone is not a boundary; an incomplete meaning stays open across a long pause.
5. Never commit right after a filler (어, 음, 그, 저, uh, um, er).
6. Never commit inside a self-repair: not after the abandoned words, not inside the correction.
7. Never commit inside a repetition or stutter ("내 내일", "그 그 사람이", "the the").
8. No commit after structures that demand continuation: Korean connective endings (-고, -다가, -면서, -려고, -니까, -아서/-어서, -면, -지만, -거나, -도록), English subordinators and openers (because, if, when, although, "I think that", "the person who"), or a dangling article, preposition or conjunction. These are strong cues, not absolute rules.
9. Pragmatic completion beats grammatical form: "저는 그렇게 생각 안 하는데요" and "Probably tomorrow" can be complete.
10. Never commit inside a quotation, complement or embedded clause ("그 사람이 온다고 | 생각해요" has no commit at |).
11. Discourse markers (그리고, 근데, 그래서, and, but, so) begin the NEXT unit; commit before them, never right after them.
12. Split an enumeration only where each item is complete by itself ("첫째 가격이 싸고" is not complete).
13. A short fragment may be committed when it is complete as an answer or reaction ("세 시쯤이요", "Tomorrow morning").
14. If the speech stops while the meaning is incomplete, do not commit there.
15. Precision first: a premature commit is much worse than a late one. When in doubt, do not commit."""

LANG_NOTE = {
    "Korean": "The transcript is Korean conversational speech. Words are space-separated eojeol. Judge by meaning and by sentence-final versus connective endings, not by spacing.",
    "English": "The transcript is English speech, possibly unpunctuated read speech (audiobook) or conversation. Judge by meaning; do not split a long sentence at a clause joint that still needs continuation."}

A_TASK = """TASK: Read the whole transcript and propose commit points.
Each line is "[index] word". Hints in braces may follow a word: {filler} filler, {rep} repeated or cut-off word, {unclear} uncertain transcription, {.} sentence-final and {,} comma punctuation in the original transcript. Hints are not labels.
Return ONLY one JSON object with exactly these keys:
{"semantic_boundaries": [[index, "word"], ...], "fillers": [[s, e], ...], "repetitions": [[s, e], ...], "repairs": [{"reparandum": [s, e], "repair": [s, e]}, ...]}
- semantic_boundaries: one [index, "word"] pair per committed unit, in ascending order, where index is the LAST word of the unit and "word" is that word copied exactly from its line (it is used to check the index).
- fillers, repetitions: inclusive [start, end] index spans (for a repetition mark the abandoned copy).
- repairs: reparandum = abandoned words; repair = replacing words, including an editing term such as 아니 or no.
- Use [] when there is none. Every index must exist in the transcript. No explanation."""

A_EXAMPLE_TAGS = {"Korean": {0: ["filler"], 8: ["filler"]}, "English": {}}   # Kspon marks fillers; LibriSpeech has none
A_EXAMPLE = {
    "Korean": (["어", "저는", "내일", "아니", "오늘", "오후에", "병원에", "갔다가요", "음", "회사로", "바로", "갈", "것",
                "같아요", "그리고", "저녁에는", "친구를", "만나려고요"],
               '{"semantic_boundaries": [[13, "같아요"], [17, "만나려고요"]], "fillers": [[0, 0], [8, 8]], "repetitions": [], '
               '"repairs": [{"reparandum": [2, 2], "repair": [3, 4]}]}'),
    "English": (["uh", "let's", "meet", "friday", "no", "thursday", "because", "the", "room", "is", "booked", "and",
                 "the", "the", "projector", "is", "broken"],
                '{"semantic_boundaries": [[10, "booked"], [16, "broken"]], "fillers": [[0, 0]], "repetitions": [[12, 12]], '
                '"repairs": [{"reparandum": [3, 3], "repair": [4, 5]}]}')}

B_TASK = """TASK: Streaming decision. You see ONLY the words spoken so far; nothing after the last word is known.
Decide whether a commit right after the LAST word is safe now.
SAFE = the last word completes a unit of meaning and a correction is unlikely.
WAIT = the meaning is incomplete, a continuation is expected, or the last word is inside a filler, repetition or self-repair.
UNCERTAIN = the prefix does not allow a confident decision.
Examples:
{examples}
Return ONLY {{"decision": "SAFE"}}, {{"decision": "WAIT"}} or {{"decision": "UNCERTAIN"}}."""
B_EXAMPLES = {
    "Korean": "제가 병원에 갔다가 → WAIT\n제가 병원에 갔다가 회사에 왔어요 → SAFE\n저는 별로 그렇게 생각 안 하는데요 → SAFE\n그 그 → WAIT\n오늘 제가 → WAIT",
    "English": "because I thought → WAIT\nif we go tomorrow → WAIT\nI think we should → WAIT\nThat's probably fine → SAFE\nwe met at the station and → WAIT"}

C_TASK = """TASK: Future stability check. PREFIX is everything up to a candidate commit point; FUTURE is the speech that immediately follows it.
Decide whether FUTURE changes what PREFIX already expressed.
REVISION = FUTURE changes what PREFIX asserted: it corrects or replaces words of PREFIX (self-repair such as "no, thursday"), cancels or restarts it, negates it, or attaches a condition or exception that changes whether PREFIX holds (such as "if it doesn't rain").
STABLE = FUTURE leaves what PREFIX asserted intact: it starts a new unit, or it keeps talking and only adds detail (time, place, reason, result, a further clause, who said it).
Continuing the same sentence is STABLE unless it changes what PREFIX asserted. Whether PREFIX itself was complete is judged elsewhere; do not judge it here.
type for REVISION: SELF_REPAIR, QUALIFICATION, RESTART. type for STABLE: NEW_UNIT, ADDITIONAL_INFORMATION.
Examples:
{examples}
Return ONLY {{"relation": "...", "type": "..."}}."""
C_EXAMPLES = {
    "Korean": "PREFIX 내일 갈게요 | FUTURE 아 아니 오늘 갈게요 → REVISION SELF_REPAIR\n"
              "PREFIX 내일 갈게요 | FUTURE 김 대리도 같이 간다고 하네요 → STABLE ADDITIONAL_INFORMATION\n"
              "PREFIX 병원에 갔어요 | FUTURE 그리고 회사에 갔어요 → STABLE NEW_UNIT\n"
              "PREFIX 제가 갈 수 있어요 | FUTURE 비가 안 오면요 → REVISION QUALIFICATION\n"
              "PREFIX 병원에 다녀왔어요 | FUTURE 감기가 심해서요 → STABLE ADDITIONAL_INFORMATION",
    "English": "PREFIX let's meet friday | FUTURE no thursday → REVISION SELF_REPAIR\n"
               "PREFIX I will come tomorrow | FUTURE if it doesn't rain → REVISION QUALIFICATION\n"
               "PREFIX I went home | FUTURE after the meeting ended → STABLE ADDITIONAL_INFORMATION\n"
               "PREFIX that's probably fine | FUTURE but we should check again → STABLE NEW_UNIT"}
RETRY = "Your previous output was invalid: {error}. Return ONLY the corrected JSON object with the required keys."
PROMPT_TEXTS = dict(version=PROMPT_VERSION, rules=RULES, lang=LANG_NOTE, a=A_TASK, a_ex=A_EXAMPLE,
                    a_ex_tags={k: {str(i): v for i, v in d.items()} for k, d in A_EXAMPLE_TAGS.items()}, b=B_TASK, b_ex=B_EXAMPLES,
                    c=C_TASK, c_ex=C_EXAMPLES, retry=RETRY, hints=HINTS, b_prefix=B_PREFIX, c_prefix=C_PREFIX,
                    suffixes=(B_SUFFIX, C_REL_SUFFIX, C_TYPE_SUFFIX), c_type_mid=C_TYPE_MID,
                    b_labels=B_LABELS, c_labels=C_LABELS, c_types=C_TYPES)


def _probe_prompts():
    """Stage A/B/C messages rendered for fixed KO/EN probe words (every hint, with and without hints / future_end):
    hashes the scaffold literals and line formats that live in the render functions, not in PROMPT_TEXTS."""
    probe = {"Korean": [("어", ["filler"]), ("내", ["rep"]), ("내일", ["unclear"]), ("갈게요", ["punct_final"]),
                        ("그리고", ["punct_comma"]), ("오늘", [])],
             "English": [("uh", ["filler"]), ("the", ["rep"]), ("room", ["unclear"]), ("is", ["punct_comma"]),
                         ("booked", ["punct_final", "filler"]), ("now", [])]}
    out = []
    for lang, ws in probe.items():
        w = [dict(i=k, text=t, tags=g) for k, (t, g) in enumerate(ws)]
        out += [stage_a_messages(w, lang), stage_a_messages(w, lang, hints=False), stage_b_messages(w[:3], lang),
                stage_c_messages(w[:3], w[3:], lang), stage_c_messages(w[:3], w[3:], lang, future_end=True)]
    return out + [RETRY.format(error="probe"), f"{C_PREFIX}REVISION{C_TYPE_MID}"]


def prompt_sha256():
    return digest(dict(texts=PROMPT_TEXTS, probe=_probe_prompts()))


def prompt_version():
    """Row-key prompt version: name + 8 hex of the hash of every prompt text and of rendered probe prompts (an edit
    of a text, a scaffold literal or the line format changes the key)."""
    return f"{PROMPT_VERSION}+{prompt_sha256()[:8]}"


def word_texts_sha256(words):
    """Per-stream digest of the word texts (Stage A rows carry it; build_labels checks it against the stream)."""
    return digest([_text(w) for w in words])


def _text(w):
    return w["text"] if isinstance(w, dict) else str(w)


def _lines(words, tags=None, hints=True):
    width = max(3, len(str(max(len(words) - 1, 0))))
    out = []
    for k, w in enumerate(words):
        t = (tags[k] if tags is not None and k < len(tags) else (w.get("tags") if isinstance(w, dict) else None)) or ()
        h = "".join(" {" + HINTS[x] + "}" for x in sorted(t, key=list(HINTS).index) if x in HINTS) if hints else ""
        out.append(f"[{k:0{width}d}] {_text(w)}{h}")
    return "\n".join(out)


def _system(lang):
    return RULES + "\n" + LANG_NOTE[lang]


def stage_a_messages(words, lang, tags=None, hints=True):
    """Full-context annotation prompt: indexed words ("[003] 어 {filler}"), index-based JSON output."""
    ex_words, ex_out = A_EXAMPLE[lang]
    ex_tags = [A_EXAMPLE_TAGS[lang].get(k, []) for k in range(len(ex_words))]
    user = (f"{A_TASK}\n\nEXAMPLE TRANSCRIPT\n{_lines(ex_words, ex_tags)}\nEXAMPLE OUTPUT\n{ex_out}\n\n"
            f"TRANSCRIPT\n{_lines(words, tags, hints)}")
    return [{"role": "system", "content": _system(lang)}, {"role": "user", "content": user}]


def stage_b_messages(prefix_words, lang):
    """Causal prefix judge. Only the prefix is rendered (plain words, no hints: punctuation hints encode future context)."""
    prefix = " ".join(_text(w) for w in prefix_words)
    user = (B_TASK.format(examples=B_EXAMPLES[lang]) +
            f"\n\nSPOKEN SO FAR:\n{prefix}\nLAST WORD: {_text(prefix_words[-1]) if prefix_words else ''}")
    return [{"role": "system", "content": _system(lang)}, {"role": "user", "content": user}]


def stage_c_messages(prefix_words, future_words, lang, future_end=False):
    """Future stability judge: prefix + bounded future window (future_end marks that speech ends after the window)."""
    fut = " ".join(_text(w) for w in future_words) + (" [end of speech]" if future_end else "")
    user = (C_TASK.format(examples=C_EXAMPLES[lang]) +
            f"\n\nPREFIX:\n{' '.join(_text(w) for w in prefix_words)}\nFUTURE:\n{fut}")
    return [{"role": "system", "content": _system(lang)}, {"role": "user", "content": user}]


# ───────────────────────── JSON extraction / validation ─────────────────────────
_SPECIAL = re.compile(r"<\|[^|>]{1,40}\|>")


def extract_json(text):
    """First JSON object in an LLM answer. Drops <think> reasoning (closed, unclosed or template-opened), gpt-oss
    channel headers and code fences; json.JSONDecoder.raw_decode from each '{' in order. ValueError if none."""
    t = text or ""
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    if "<think>" in t:
        t = t.split("<think>", 1)[0]
    if "<|message|>" in t:
        t = t.rsplit("<|message|>", 1)[1]
    t = _SPECIAL.sub("", t)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            obj, _ = dec.raw_decode(t, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object in output")


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _check_ranges(d, n):
    """Shared Stage A range checks (both backends): 0 ≤ index < n, span s ≤ e. Returns the normalized dict."""
    def idx(x, what):
        if not 0 <= x < n:
            raise ValueError(f"{what} index {x} out of range [0, {n - 1}]")
        return x

    def span(s, what):
        s0, s1 = idx(s[0], what), idx(s[1], what)
        if s0 > s1:
            raise ValueError(f"{what} span {list(s)} has start > end")
        return [s0, s1]
    return {"semantic_boundaries": sorted({idx(x, "semantic_boundaries") for x in d["semantic_boundaries"]}),
            "fillers": [span(s, "fillers") for s in d.get("fillers", [])],
            "repetitions": [span(s, "repetitions") for s in d.get("repetitions", [])],
            "repairs": [{"reparandum": span(r["reparandum"], "reparandum"), "repair": span(r["repair"], "repair")}
                        for r in d.get("repairs", [])]}


def _std_stage_a(obj, n):
    def keys(d, allowed, required, what):
        if not isinstance(d, dict):
            raise ValueError(f"{what} must be an object")
        extra, missing = set(d) - set(allowed), set(required) - set(d)
        if extra or missing:
            raise ValueError(f"{what}: extra keys {sorted(extra)} missing keys {sorted(missing)}")

    def ints(v, what):
        if not isinstance(v, list) or not all(_is_int(x) for x in v):
            raise ValueError(f"{what} must be a list of integers")
        return v

    def spans(v, what):
        if not isinstance(v, list) or not all(isinstance(s, list) and len(s) == 2 for s in v):
            raise ValueError(f"{what} must be a list of [start, end]")
        for s in v:
            ints(s, what)
        return v
    keys(obj, ("semantic_boundaries", "fillers", "repetitions", "repairs"), ("semantic_boundaries",), "stage A")
    ints(obj["semantic_boundaries"], "semantic_boundaries")
    spans(obj.get("fillers", []), "fillers"), spans(obj.get("repetitions", []), "repetitions")
    if not isinstance(obj.get("repairs", []), list):
        raise ValueError("repairs must be a list")
    for r in obj.get("repairs", []):
        keys(r, ("reparandum", "repair"), ("reparandum", "repair"), "repair")
        spans([r["reparandum"], r["repair"]], "repair")
    return _check_ranges(obj, n)


_PYD = {}


def _pyd_models():
    """pydantic v2 models (extra='forbid', strict ints, range checks via validation context n). None if unavailable."""
    if "a" not in _PYD:
        try:
            from typing import List, Literal, Tuple
            from pydantic import BaseModel, ConfigDict, StrictInt, ValidationInfo, model_validator
        except ImportError:
            _PYD["a"] = None
            return None
        Span = Tuple[StrictInt, StrictInt]

        class Repair(BaseModel):
            model_config = ConfigDict(extra="forbid")
            reparandum: Span
            repair: Span

        class StageA(BaseModel):
            model_config = ConfigDict(extra="forbid")
            semantic_boundaries: List[StrictInt]
            fillers: List[Span] = []
            repetitions: List[Span] = []
            repairs: List[Repair] = []

            @model_validator(mode="after")
            def _ranges(self, info: ValidationInfo):
                _check_ranges(self.model_dump(), (info.context or {})["n"])
                return self

        class StageB(BaseModel):
            model_config = ConfigDict(extra="forbid")
            decision: Literal["SAFE", "WAIT", "UNCERTAIN"]

        class StageC(BaseModel):
            model_config = ConfigDict(extra="forbid")
            relation: Literal["STABLE", "REVISION"]
            type: str = ""
        _PYD.update(a=StageA, b=StageB, c=StageC)
    return _PYD if _PYD["a"] else None


def validate_stage_a(obj, n, backend=None):
    """Validated, normalized Stage A dict (boundaries sorted/deduplicated, spans as lists). Raises ValueError
    (pydantic.ValidationError is a ValueError) on extra keys, non-int indices or indices outside [0, n)."""
    backend = backend or ("pydantic" if _pyd_models() else "std")
    if backend == "std":
        return _std_stage_a(obj, n)
    m = _pyd_models()["a"].model_validate(obj, context={"n": n})
    return _check_ranges(json.loads(m.model_dump_json()), n)


def validate_decision(obj, stage="B"):
    """Stage B {"decision"} / Stage C {"relation","type"} (generation-mode fallback; scoring mode never needs it)."""
    allowed = dict(B=("decision",), C=("relation", "type"))[stage]
    if _pyd_models():
        return _pyd_models()[stage.lower()].model_validate(obj).model_dump()
    if not isinstance(obj, dict) or set(obj) - set(allowed) or allowed[0] not in obj:
        raise ValueError(f"stage {stage}: keys must be {allowed}")
    if obj[allowed[0]] not in (B_LABELS if stage == "B" else C_LABELS):
        raise ValueError(f"stage {stage}: bad label {obj[allowed[0]]!r}")
    return dict(obj) if stage == "B" else {"relation": obj["relation"], "type": str(obj.get("type", ""))}


# ───────────────────────── model adapter ─────────────────────────
def gptoss_final_prefill(s):
    """Harmony generation prompt ends in '<|start|>assistant'; append the final-channel header so the answer is
    generated/scored directly in the final channel (reasoning skipped). Keep it if a channel header is already there."""
    if re.search(r"<\|channel\|>[^<]*<\|message\|>$", s):
        return s
    if s.endswith("<|start|>assistant"):
        return s + "<|channel|>final<|message|>"
    raise ValueError(f"unexpected gpt-oss generation prompt tail: {s[-60:]!r}")


def parse_max_memory(s):
    """'0:37GiB,cpu:120GiB' → {0: '37GiB', 'cpu': '120GiB'}."""
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[int(k) if k.strip().isdigit() else k.strip()] = v.strip()
    return out


def cut_at_eos(ids, eos_ids):
    """Generated ids → (ids before the first eos, truncated = no eos generated)."""
    for k, t in enumerate(ids):
        if t in eos_ids:
            return ids[:k], False
    return list(ids), True


def continuation_logprobs(logits, ids, cont_lens):
    """Σ log p(continuation) for left-padded, right-aligned rows. logits/ids: (B, T, V)/(B, T) of the LAST T positions
    (T > max cont_len); the token at t is predicted by logits at t−1, so row b sums its last cont_lens[b] targets."""
    lp = logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return [lp[b, lp.shape[1] - c:].sum().item() for b, c in enumerate(cont_lens)]


def label_result(labels, lps):
    """argmax label (first label on exact ties), per-label log-probs and margin = top1 − top2 (None for one label).
    A non-finite score (NaN/inf, e.g. bf16 overflow) gives label None, nonfinite=True and the log-probs as strings, so the
    row stays JSON-serializable and is written as status 'nonfinite' (graded as a missing decision)."""
    if not all(math.isfinite(v) for v in lps):
        return {"label": None, "logprobs": {l: str(v) for l, v in zip(labels, lps)}, "margin": None, "nonfinite": True}
    order = sorted(range(len(labels)), key=lambda j: (-lps[j], j))
    margin = lps[order[0]] - lps[order[1]] if len(labels) > 1 else None
    return {"label": labels[order[0]], "logprobs": {l: round(v, 5) for l, v in zip(labels, lps)},
            "margin": None if margin is None else round(margin, 5)}


class LLM:
    """HF causal-LM adapter for one teacher. Tokenizer loads at construction (fingerprint without GPU); the model loads
    lazily on first use. kind ∈ {qwen3, exaone35, gptoss}. `model`/`tokenizer` may be injected (tests)."""

    def __init__(self, path, kind, device="cuda", max_memory=None, dtype="bfloat16", model=None, tokenizer=None):
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
        self.path, self.kind, self.device_name, self.dtype = (str(path) if path else None), kind, device, dtype
        self.template_kwargs = dict(KINDS[kind]["template_kwargs"])
        self.trust_remote_code = KINDS[kind]["trust_remote_code"]
        # gptoss on one A100-40GB: MXFP4 is dequantized to bf16 (~42 GB) under triton 3.2, and device_map='auto' plans with the
        # packed size and OOMs mid-dequant (rack4 smoke 2026-09-23). Default = explicit placement: the first N decoder layers
        # (+embed/norm/lm_head) on the GPU, the rest on CPU (N=18 measured 34.5 GiB peak). --max-memory still selects 'auto'.
        self.gpu_layers = None
        if max_memory is None and kind == "gptoss" and str(device).startswith("cuda"):
            self.gpu_layers = int(os.environ.get("SEMCOMMIT_GPTOSS_GPU_LAYERS", "18"))
        self.max_memory = max_memory
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=self.trust_remote_code,
                                                      local_files_only=True)
        self.tok, self._model = tokenizer, (model.eval() if model is not None and hasattr(model, "eval") else model)
        self.max_batch = None   # sticky batch cap after a CUDA OOM split
        self.tok.padding_side = "left"
        if getattr(self.tok, "pad_token_id", None) is None and getattr(self.tok, "eos_token", None) is not None:
            self.tok.pad_token = self.tok.eos_token

    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM
            kw = dict(dtype=getattr(torch, self.dtype), trust_remote_code=self.trust_remote_code,
                      local_files_only=True, low_cpu_mem_usage=True)
            if self.gpu_layers is not None:
                import json as _json
                nl = int(_json.load(open(os.path.join(self.path, "config.json")))["num_hidden_layers"])
                dm = {"model.embed_tokens": 0, "model.norm": 0, "model.rotary_emb": 0, "lm_head": 0}
                dm.update({f"model.layers.{i}": (0 if i < self.gpu_layers else "cpu") for i in range(nl)})
                kw.update(device_map=dm)
            else:
                kw.update(dict(device_map="auto", max_memory=self.max_memory) if self.max_memory
                          else dict(device_map={"": "cuda:0" if self.device_name == "cuda" else self.device_name}))
            self._model = AutoModelForCausalLM.from_pretrained(self.path, **kw).eval()
        return self._model

    def load(self):
        return self.model

    def _device(self):
        import torch
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return torch.device("cpu")

    @property
    def pad_id(self):
        return self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_ids()[0]

    def eos_ids(self):
        e = getattr(getattr(self._model, "generation_config", None), "eos_token_id", None)
        e = e if e is not None else self.tok.eos_token_id
        ids = [e] if isinstance(e, int) else list(e)
        if self.kind == "gptoss":   # harmony message ends; stop even if the final message closes with <|end|>
            unk = getattr(self.tok, "unk_token_id", None)
            for t in ("<|return|>", "<|end|>", "<|call|>"):
                i = self.tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i != unk and i not in ids:
                    ids.append(i)
        return ids

    def build_prompt(self, messages):
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **self.template_kwargs)
        return gptoss_final_prefill(s) if self.kind == "gptoss" else s

    def encode(self, text):
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def fingerprint(self):
        tpl = getattr(self.tok, "chat_template", None)
        p = Path(self.path) if self.path else None
        cfg = p / "config.json" if p else None
        pk = {}
        for name in ("torch", "transformers", "pydantic"):
            try:
                pk[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pk[name] = None
        return dict(model_path=self.path, kind=self.kind, dtype=self.dtype, trust_remote_code=self.trust_remote_code,
                    template_kwargs=self.template_kwargs, gptoss_final_prefill=self.kind == "gptoss",
                    chat_template_sha256=digest(tpl) if tpl is not None else None,
                    config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest() if cfg and cfg.exists() else None,
                    weights={q.name: q.stat().st_size for q in sorted(p.glob("*.safetensors"))} if p and p.is_dir() else {},
                    max_memory={str(k): v for k, v in (self.max_memory or {}).items()}, gpu_layers=self.gpu_layers, packages=pk)

    def generation_config(self, max_new_tokens):
        return dict(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                    max_new_tokens=int(max_new_tokens), pad_token_id=self.pad_id, eos_token_id=self.eos_ids())

    def _oom_split(self, fn, items):
        """Run fn(items); on CUDA OOM halve the batch (a batch of one re-raises). The halved size sticks (max_batch), so
        later batches start at a size that fit instead of re-OOMing every time (gpt-oss offload: each failed forward
        streams CPU-offloaded experts)."""
        import torch
        try:
            return fn(items)
        except torch.cuda.OutOfMemoryError:
            if len(items) == 1:
                raise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.max_batch = min(self.max_batch or len(items), len(items) // 2)
        print(f"OOM split {len(items)} -> max batch {self.max_batch}", flush=True)
        return self._run_batched(fn, items, self.max_batch)

    def _run_batched(self, fn, items, batch_size):
        """fn over consecutive batches of min(batch_size, max_batch) items (results in item order)."""
        out, s = [], 0
        while s < len(items):
            b = items[s:s + min(batch_size, self.max_batch or batch_size)]
            out += self._oom_split(fn, b)
            s += len(b)
        return out

    def _left_pad(self, seqs):
        import torch
        L = max(len(s) for s in seqs)
        ids = torch.tensor([[self.pad_id] * (L - len(s)) + list(s) for s in seqs], dtype=torch.long)
        mask = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in seqs], dtype=torch.long)
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
        dev = self._device()
        return ids.to(dev), mask.to(dev), pos.to(dev)

    def _gen_batch(self, max_new_tokens, seqs):
        import inspect
        import torch
        from transformers import GenerationConfig
        ids, mask, _ = self._left_pad(seqs)
        kw = {}
        if "use_model_defaults" in inspect.signature(self.model.generate).parameters:
            kw["use_model_defaults"] = False   # ≥4.50: else the model's do_sample=True silently overrides greedy
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=mask,
                                      generation_config=GenerationConfig(**self.generation_config(max_new_tokens)), **kw)
        eos = set(self.eos_ids())
        res = []
        for row in out[:, ids.shape[1]:].tolist():
            keep, truncated = cut_at_eos(row, eos)
            text = self.tok.decode(keep, skip_special_tokens=self.kind != "gptoss")
            if self.kind == "gptoss":
                text = _SPECIAL.sub("", text.rsplit("<|message|>", 1)[-1])
            res.append({"text": text, "truncated": truncated, "n_new": len(keep)})
        return res

    def generate_json(self, prompts, max_new_tokens=768, batch_size=8):
        """Greedy decode of each prompt (prompt tokens sliced off). Returns per prompt {text, truncated, n_new,
        json (first JSON object or None), error}. Length-sorted batches; CUDA OOM halves the batch (sticky)."""
        enc = [self.encode(p) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: (len(enc[i]), i))
        out = [None] * len(prompts)
        for i, r in zip(order, self._run_batched(lambda b: self._gen_batch(max_new_tokens, b), [enc[i] for i in order],
                                                 batch_size)):
            out[i] = r
        for r in out:
            try:
                r["json"], r["error"] = extract_json(r["text"]), None
            except ValueError as e:
                r["json"], r["error"] = None, str(e)
        return out

    def _score_batch(self, jobs):
        import torch
        seqs, clens = [j[0] for j in jobs], [j[1] for j in jobs]
        ids, mask, pos = self._left_pad(seqs)
        keep = max(clens) + 1
        with torch.inference_mode():
            try:
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                                 logits_to_keep=keep)
            except TypeError:   # older/remote-code forward without logits_to_keep
                out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
            return continuation_logprobs(out.logits[:, -keep:], ids[:, -keep:], clens)

    def score_labels(self, prompts, labels, prefix=B_PREFIX, batch_size=8, suffix=B_SUFFIX):
        """Teacher-forced log p(label + suffix | prompt + prefix) for every label. Context and continuation are tokenized
        separately (same context ids for every label). Returns per prompt {label (argmax; exact ties → first label),
        logprobs, margin, tok_ok}; tok_ok = the separate ids equal the joint tokenization of prompt+prefix+label+suffix."""
        labels = list(labels)
        ctx = [self.encode(p + prefix) for p in prompts]
        cont = [self.encode(l + suffix) for l in labels]
        tok_ok = [all(self.encode(p + prefix + l + suffix) == c + k for l, k in zip(labels, cont))
                  for p, c in zip(prompts, ctx)]
        if not all(tok_ok):
            print(f"WARNING: label tokenization differs from joint tokenization for {tok_ok.count(False)} prompt(s) "
                  f"(prefix={prefix!r}, suffix={suffix!r})", flush=True)
        jobs = sorted(((i, j) for i in range(len(prompts)) for j in range(len(labels))),
                      key=lambda ij: (len(ctx[ij[0]]), ij))
        lp = dict(zip(jobs, self._run_batched(self._score_batch, [(ctx[i] + cont[j], len(cont[j])) for i, j in jobs],
                                              batch_size)))
        return [dict(label_result(labels, [lp[i, j] for j in range(len(labels))]), tok_ok=tok_ok[i])
                for i in range(len(prompts))]


# ───────────────────────── stage logic (pure; llm = anything with build_prompt/generate_json/score_labels) ─────────
def future_window(words, i, next_candidate=None, max_words=12, max_s=3.0, min_words=1):
    """Future words after word i for Stage C: stop at the next candidate (inclusive), after max_words, or at words ending
    later than end_time[i] + max_s, whichever comes first. The first min_words future words are always kept (a long pause
    must not turn an observed future into UNOBSERVED). [] only when i is the last word → UNOBSERVED."""
    out, t0 = [], words[i]["end_time"]
    for j in range(i + 1, len(words)):
        if len(out) >= max_words or (len(out) >= min_words and words[j]["end_time"] - t0 > max_s):
            break
        out.append(words[j])
        if next_candidate is not None and j >= next_candidate:
            break
    return out


def disfluency_reason(i, stage_a=None, tags=None):
    """Why a boundary after word i breaks Rules 5–7 (None if it does not): i is a filler/repetition/unclear word (tags or
    Stage A spans), lies in a reparandum, between reparandum and repair, is the last word before a repair onset, or
    lies inside a repair span (not at its end)."""
    t = set((tags[i] if tags is not None and i < len(tags) else None) or ())
    for tag in DISFL_TAGS:
        if tag in t:
            return f"tag_{tag}"
    a = stage_a or {}
    for name in ("fillers", "repetitions"):
        if any(s <= i <= e for s, e in a.get(name, [])):
            return name[:-1]
    for r in a.get("repairs", []):
        (rs, re_), (ps, pe) = r["reparandum"], r["repair"]
        if rs <= i <= re_:
            return "reparandum"
        if i == ps - 1 or re_ < i < ps:
            return "before_repair"
        if ps <= i < pe:
            return "inside_repair"
    return None


def disfluency_conflict(i, stage_a=None, tags=None):
    return disfluency_reason(i, stage_a, tags) is not None


def grade(candidate, lang, judges=None, strict=True, tiebreak_judge=None, c_mask_types=C_MASK_TYPES):
    """Grade one candidate → (grade, why, resolved_by_judge).
    candidate: {stageA: bool, B: {judge: SAFE|WAIT|UNCERTAIN}, C: {relation, type}, disfluency: reason|None}.
    N iff C == REVISION with a type outside c_mask_types, all primary judges WAIT, or a disfluency conflict from a human
    transcript tag (reason 'tag_*'; a Stage-A-only conflict grades B 'disflA_*'). A iff Stage A
    candidate, all primary judges SAFE, C ∈ {STABLE, UNOBSERVED} and no conflict. Otherwise B (incl. REVISION of a masked
    type: 'C_revision_masked:<type>'; relation None/'MISSING' = no Stage C decision). A primary-judge disagreement that
    the tie-break judge resolves is recorded (resolved_by_judge) but stays B under strict; strict=False lets a resolved
    SAFE majority and all-SAFE among available (≥1) primary judges count as SAFE."""
    judges = list(judges or PRIMARY_JUDGES[lang])
    tb = tiebreak_judge if tiebreak_judge is not None else TIEBREAK_JUDGE.get(lang)
    C = candidate.get("C") or {}
    B, rel, ctype = candidate.get("B") or {}, C.get("relation"), C.get("type")
    rel = None if rel == "MISSING" else rel
    dec = [B.get(j) for j in judges]
    present = [d for d in dec if d is not None]
    # Disfluency conflicts: a human transcript marker (Kspon '/', '+', '*' → reason 'tag_*') is trusted → hard negative N.
    # A conflict that comes only from Stage A's own spans (fillers/repetitions/repairs) is not: on the rack4 smoke Qwen3-8B
    # marked content words ('같애', '있잖아', '소주밤이라') as fillers, which would turn a correct boundary into a trained
    # negative. Such candidates are masked (B) instead (precision-first without teaching false negatives).
    disfl = candidate.get("disfluency")
    n_why = (["C_revision"] if rel == "REVISION" and ctype not in c_mask_types else []) + (["B_all_wait"] if dec and all(d == "WAIT" for d in dec) else []) \
        + ([f"disfl_{disfl}"] if disfl and str(disfl).startswith("tag_") else [])
    disagree = len(set(present)) > 1
    resolved = bool(disagree and tb and tb not in judges and B.get(tb) in present)
    if n_why:
        return "N", "+".join(n_why), resolved
    if not candidate.get("stageA", True):
        return "B", "not_stageA", resolved
    if disfl:                                   # Stage-A-only disfluency conflict (see above) → mask
        return "B", f"disflA_{disfl}", resolved
    safe = bool(dec) and all(d == "SAFE" for d in dec)
    if not strict and not safe:
        votes = present + ([B[tb]] if resolved else [])
        safe = (resolved and votes.count("SAFE") * 2 > len(votes)) or \
               (not disagree and bool(present) and all(d == "SAFE" for d in present))
    if safe and rel in ("STABLE", "UNOBSERVED"):
        return "A", "safe+" + rel.lower() + ("+resolved" if resolved else ""), resolved
    why = []
    if None in dec:
        why.append("B_missing:" + ",".join(j for j, d in zip(judges, dec) if d is None))
    if disagree:
        why.append("B_disagree" + (f"(resolved:{B[tb]})" if resolved else ""))
    elif present and present[0] != "SAFE":
        why.append("B_" + present[0].lower())
    if rel not in ("STABLE", "UNOBSERVED"):
        why.append("C_missing" if rel is None else f"C_revision_masked:{ctype}" if rel == "REVISION" else f"C_{rel}")
    return "B", "+".join(why) or "B", resolved


def row_key(stage, sid, after_word, judge, pv):
    return (stage, sid, after_word, judge, pv)


def _row(stage, s, i, judge, pv, **kw):
    return dict(key=list(row_key(stage, s["id"], i, judge, pv)), stage=stage, id=s["id"], lang=s["lang"],
                after_word=i, judge=judge, prompt_version=pv, **kw)


def error_text(e):
    """Short one-line validation message (pydantic errors() flattened; also used as retry feedback to the LLM)."""
    if hasattr(e, "errors"):
        return "; ".join((".".join(map(str, d["loc"])) + ": " if d["loc"] else "") + d["msg"] for d in e.errors())[:500]
    return str(e)[:500]


A_SNAP_RADIUS = 4
_SNAP_STRIP = re.compile(r"[^\w']+", re.UNICODE)


def _snap_norm(x):
    return _SNAP_STRIP.sub("", str(x)).lower()


def snap_boundaries(obj, words, radius=A_SNAP_RADIUS):
    """v0.2 Stage A anchors: semantic_boundaries entries are [index, "word"] (a bare int is accepted unchecked).
    An anchor whose word does not match words[index] is moved to the nearest index within ±radius whose word matches
    (ties → the smaller distance, then the earlier index); no match → dropped. Returns (obj with int boundaries, info).
    Rack4 smoke (2026-09-23): Qwen3-8B mis-indexed long unpunctuated LibriSpeech streams by 2–3 words."""
    if not isinstance(obj, dict) or not isinstance(obj.get("semantic_boundaries"), list):
        return obj, dict(anchors=0, exact=0, snapped=0, dropped=0, bare=0)
    texts = [_snap_norm(_text(w)) for w in words]; out = []; info = dict(anchors=0, exact=0, snapped=0, dropped=0, bare=0, moves=[])
    for x in obj["semantic_boundaries"]:
        if _is_int(x):
            out.append(x); info["bare"] += 1; continue
        if isinstance(x, dict):
            x = [x.get("i", x.get("index")), x.get("w", x.get("word"))]
        if not (isinstance(x, (list, tuple)) and len(x) == 2 and _is_int(x[0]) and isinstance(x[1], str)):
            raise ValueError(f"semantic_boundaries entry {x!r} must be [index, \"word\"]")
        i, w = x[0], _snap_norm(x[1]); info["anchors"] += 1
        if 0 <= i < len(texts) and texts[i] == w:
            out.append(i); info["exact"] += 1; continue
        cand = [j for d in range(1, radius + 1) for j in (i - d, i + d) if 0 <= j < len(texts) and texts[j] == w]
        if cand and w:
            out.append(cand[0]); info["snapped"] += 1; info["moves"].append([i, cand[0]])
        else:
            info["dropped"] += 1
    return dict(obj, semantic_boundaries=out), info


def _parse_a(o, words):
    n = len(words)
    try:
        obj, info = snap_boundaries(extract_json(o["text"]), words)
        return "ok", validate_stage_a(obj, n), None, info
    except ValueError as e:
        return ("truncated" if o.get("truncated") else "invalid_json"), None, error_text(e), None


def run_stage_a(llm, streams, judge, max_new_tokens=768, batch_size=8, retries=1, pv=None):
    """Stage A rows (one per stream): status ok|invalid_json|truncated, validated `out`, raw text on failure.
    An invalid (non-truncated) answer gets `retries` feedback rounds (previous answer + error message)."""
    pv = pv or prompt_version()
    msgs = [stage_a_messages(s["words"], s["lang"]) for s in streams]
    outs = llm.generate_json([llm.build_prompt(m) for m in msgs], max_new_tokens=max_new_tokens, batch_size=batch_size)
    res = [list(_parse_a(o, s["words"])) + [o, 1] for s, o in zip(streams, outs)]
    for _ in range(retries):
        redo = [k for k, r in enumerate(res) if r[0] == "invalid_json"]
        if not redo:
            break
        rm = [msgs[k] + [{"role": "assistant", "content": res[k][4]["text"]},
                         {"role": "user", "content": RETRY.format(error=res[k][2])}] for k in redo]
        for k, o in zip(redo, llm.generate_json([llm.build_prompt(m) for m in rm], max_new_tokens=max_new_tokens,
                                                batch_size=batch_size)):
            res[k] = list(_parse_a(o, streams[k]["words"])) + [o, res[k][5] + 1]
    rows = []
    for s, (status, out, err, snap, o, att) in zip(streams, res):
        extra = dict(raw=o.get("text", "")[:4000], error=err) if status != "ok" else dict(snap=snap)
        rows.append(_row("A", s, None, judge, pv, status=status, out=out, n_words=len(s["words"]),
                         word_texts_sha256=word_texts_sha256(s["words"]), attempts=att,
                         truncated=bool(o.get("truncated")), **extra))
    return rows


def run_stage_b(llm, items, judge, batch_size=8, stage="B", pv=None):
    """Causal prefix judge rows for items [(stream, after_word)] by teacher-forced label scoring (B_LABELS).
    A non-finite score gives status 'nonfinite' (decision None; not retried on resume, graded as missing)."""
    pv = pv or prompt_version()
    prompts = [llm.build_prompt(stage_b_messages(s["words"][:i + 1], s["lang"])) for s, i in items]
    res = llm.score_labels(prompts, B_LABELS, prefix=B_PREFIX, batch_size=batch_size, suffix=B_SUFFIX) if prompts else []
    return [_row(stage, s, i, judge, pv, status="nonfinite" if r.get("nonfinite") else "ok", decision=r["label"],
                 logprobs=r["logprobs"], margin=r["margin"], tok_ok=r.get("tok_ok")) for (s, i), r in zip(items, res)]


def run_stage_c(llm, items, judge, batch_size=8, max_words=12, max_s=3.0, pv=None):
    """Future stability rows for items [(stream, after_word, next_candidate)]. Empty future → UNOBSERVED (no LLM call).
    relation scored over C_LABELS, then type scored among C_TYPES[relation]. Rows keep next_candidate (the window
    depends on the Stage A file; build_labels checks it) and type as str ('' when none); a non-finite relation or type
    score gives status 'nonfinite'."""
    pv = pv or prompt_version()
    rows, todo = [None] * len(items), []
    for k, (s, i, nc) in enumerate(items):
        fut = future_window(s["words"], i, nc, max_words, max_s)
        if not fut:
            rows[k] = _row("C", s, i, judge, pv, status="ok", relation="UNOBSERVED", type="", future_n=0,
                           future_end=True, next_candidate=nc, margin=None, logprobs=None)
        else:
            todo.append((k, s, i, fut, i + len(fut) == len(s["words"]) - 1, nc))
    prompts = [llm.build_prompt(stage_c_messages(s["words"][:i + 1], fut, s["lang"], end))
               for _, s, i, fut, end, _ in todo]
    rel = llm.score_labels(prompts, C_LABELS, prefix=C_PREFIX, batch_size=batch_size, suffix=C_REL_SUFFIX) \
        if prompts else []
    typ = [None] * len(todo)
    for r_label in C_LABELS:
        sel = [n for n, r in enumerate(rel) if r["label"] == r_label]
        if sel:
            got = llm.score_labels([prompts[n] for n in sel], C_TYPES[r_label], prefix=f"{C_PREFIX}{r_label}{C_TYPE_MID}",
                                   batch_size=batch_size, suffix=C_TYPE_SUFFIX)
            for n, g in zip(sel, got):
                typ[n] = g
    for (k, s, i, fut, end, nc), r, t in zip(todo, rel, typ):
        t = t or {}
        rows[k] = _row("C", s, i, judge, pv, status="nonfinite" if r.get("nonfinite") or t.get("nonfinite") else "ok",
                       relation=r["label"], type=t.get("label") or "", future_n=len(fut), future_end=end,
                       next_candidate=nc, margin=r["margin"], logprobs=r["logprobs"], type_logprobs=t.get("logprobs"),
                       tok_ok=r.get("tok_ok"), type_tok_ok=t.get("tok_ok"))
    return rows


def candidates_of(a_rows):
    """{stream id: (sorted Stage A boundaries, Stage A out)} from ok Stage A rows (one row per id required)."""
    out = {}
    for r in a_rows:
        if r.get("status") == "ok":
            if r["id"] in out and out[r["id"]][1] != r["out"]:
                raise ValueError(f"conflicting Stage A rows for {r['id']}")
            out[r["id"]] = (r["out"]["semantic_boundaries"], r["out"])
    return out


EXTRA_SOURCES = ("last", "seg_end", "punct_final")   # recipe v0.3 candidate sources beside Stage A (gold v1: +0.06–0.15 candidate recall)


def extra_candidates(stream, kinds=EXTRA_SOURCES):
    """{word index: [source, ...]} for rule-based candidate sources: 'last' = the stream's last word, 'seg_end' = the last
    word of each segment (utterance end inside a multi-utterance stream), 'punct_final' = a word whose human transcript
    ends a sentence (. ? ! — LibriSpeech-PC / GigaSpeech punctuation, Kspon transcript punctuation; words tag)."""
    words, out = stream["words"], {}
    if not words:
        return out
    if "last" in kinds:
        out.setdefault(len(words) - 1, []).append("last")
    if "seg_end" in kinds:
        ends = {}
        for k, w in enumerate(words):
            ends[w.get("seg", 0)] = k
        for k in sorted(ends.values()):
            out.setdefault(k, []).append("seg_end")
    if "punct_final" in kinds:
        for k, w in enumerate(words):
            if "punct_final" in (w.get("tags") or ()):
                out.setdefault(k, []).append("punct_final")
    return out


def candidate_sets(streams, a_rows, extra=()):
    """{stream id: (sorted candidates, Stage A set, {i: sources})} — Stage A boundaries ∪ extra sources (streams without an
    ok Stage A row get none: the same rule as v0.2). Sources list 'stageA' first when Stage A proposed the index."""
    cands, out = candidates_of(a_rows), {}
    for s in streams:
        if s["id"] not in cands:
            continue
        sa = set(cands[s["id"]][0]); src = {i: ["stageA"] for i in sa}
        for i, ks in extra_candidates(s, tuple(extra)).items() if extra else ():
            src.setdefault(i, []).extend(ks)
        out[s["id"]] = (sorted(src), sa, src)
    return out


def next_stage_a(sa_sorted, i):
    """Next Stage A boundary after word i (None if none) — the Stage C future window stops there for every candidate
    (v0.2 rows keep their next_candidate; extra candidates use the same rule)."""
    return next((b for b in sa_sorted if b > i), None)


def stage_b_items(streams, a_rows, extra=(), only_extra=False):
    cs = candidate_sets(streams, a_rows, extra)
    return [(s, i) for s in streams for i in cs.get(s["id"], ((), set(), {}))[0] if not (only_extra and i in cs[s["id"]][1])]


def stage_c_items(streams, a_rows, extra=(), only_extra=False):
    cs, out = candidate_sets(streams, a_rows, extra), []
    for s in streams:
        if s["id"] not in cs:
            continue
        allc, sa, _ = cs[s["id"]]; sas = sorted(sa)
        out += [(s, i, next_stage_a(sas, i)) for i in allc if not (only_extra and i in sa)]
    return out


def index_rows(rows):
    """{(id, after_word, judge): row}; an ok row wins over an error row, a later ok row over an earlier one."""
    idx = {}
    for r in rows:
        k = (r["id"], r["after_word"], r["judge"])
        if r.get("status") == "ok" or k not in idx or idx[k].get("status") != "ok":
            idx[k] = r
    return idx


def tiebreak_items(streams, a_rows, b_rows, judges=None, lang="Korean"):
    """Candidates of `lang` streams whose primary judges all answered and disagree."""
    judges = list(judges or PRIMARY_JUDGES[lang])
    bi = index_rows(b_rows)
    out = []
    for s, i in stage_b_items([s for s in streams if s["lang"] == lang], a_rows):
        d = [bi.get((s["id"], i, j), {}).get("decision") for j in judges]
        if None not in d and len(set(d)) > 1:
            out.append((s, i))
    return out


def by_candidate(rows):
    """{(id, after_word): {judge: ok row}} (index_rows preference rules)."""
    out = {}
    for (sid, i, j), r in index_rows(rows).items():
        if r.get("status") == "ok":
            out.setdefault((sid, i), {})[j] = r
    return out


def _q(xs, q):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else None


def _kappa(pairs, cats=B_LABELS):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in cats)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 4)


def _tok_bad(r):
    return r.get("tok_ok") is False or r.get("type_tok_ok") is False


RECIPE_V3 = "semcommit-recipe-v0.3"
# gold v1 dev-half tuning with a per-branch precision floor 0.90 (labels v0.3.2; Qwen3-8B + EXAONE-3.5-7.8B judges, Qwen3-8B Stage C):
# punctuated sentence ends are A; a Korean reply-word unit before more speech (punct_resp: gold dev 0 COMMIT / 8 NO) and the other branch
# (no threshold reaches 0.90 on its own, English dev 0.40 at best) are off (None). Re-tune for other teachers (semcommit_gold.py tune).
V3_THRESHOLDS = {"English": {"punct": {"p_safe": 0.0, "p_rev": 1.01}, "punct_resp": {"p_safe": 0.0, "p_rev": 1.01}, "other": None},
                 "Korean": {"punct": {"p_safe": 0.0, "p_rev": 1.01}, "punct_resp": None, "other": None}}


def label_probs(logprobs):
    """Softmax over scored label log-probs ({label: logprob}) → {label: p}; None for missing or non-finite scores."""
    if not logprobs or any(v is None or not math.isfinite(v) for v in logprobs.values()):
        return None
    m = max(logprobs.values()); z = {k: math.exp(v - m) for k, v in logprobs.items()}; t = sum(z.values())
    return {k: v / t for k, v in z.items()}


def candidate_features(stream, i, b_rows_by_judge, c_row, judges, sources=(), stage_a_out=None):
    """Grade inputs for word i: per-judge P(SAFE) (softmax of the B label log-probs), mean over the primary judges (None if
    any is missing), P(REVISION) from Stage C (0 when the future is unobserved, None when missing), punctuation and
    disfluency flags. Deterministic, no LLM call."""
    ps = {}
    for j in judges:
        r = b_rows_by_judge.get(j)
        p = label_probs(r.get("logprobs")) if r else None
        ps[j] = None if p is None else round(p.get("SAFE", 0.0), 6)
    mean = None if any(v is None for v in ps.values()) or not ps else round(sum(ps.values()) / len(ps), 6)
    if c_row is None:
        prev = None
    elif c_row.get("relation") == "UNOBSERVED":
        prev = 0.0
    else:
        pr = label_probs(c_row.get("logprobs")); prev = None if pr is None else round(pr.get("REVISION", 0.0), 6)
    tags = stream["words"][i].get("tags") or []; punct = "punct_final" in tags
    return dict(p_safe=ps, p_safe_mean=mean, p_rev=prev, punct=punct, resp_head=punct and response_unit(stream["words"], i, stream.get("lang")),
                disfluency=disfluency_reason(i, stage_a_out, [w.get("tags") or [] for w in stream["words"]]), sources=list(sources))


# Answer particles / backchannels (gold v1 convention: a reply word opening a longer turn is NO, alone at the end it is COMMIT).
RESPONSE_TOKENS = {"Korean": frozenset("네 예 응 어 음 아 아니 아니야 아니요 아뇨 맞아 맞아요 맞지 그렇지 그렇죠 그래 그래요 그치 그쵸 아니지".split()),
                   "English": frozenset("yeah yes no okay ok right oh well sure yep nope uh-huh mm-hmm mhm".split())}


def response_unit(words, i, lang):
    """True when word i ends a punctuated unit made only of reply words (words after the previous punct_final, up to i) and
    speech follows in the stream — e.g. `네. 저는 …` (gold v1: NO), while a stream-final `네.` stays a normal sentence end."""
    toks = RESPONSE_TOKENS.get(lang)
    if not toks or i >= len(words) - 1:
        return False
    j = i - 1
    while j >= 0 and "punct_final" not in (words[j].get("tags") or []):
        j -= 1
    unit = [_tok(w["text"]) for w in words[j + 1:i + 1]]
    return bool(unit) and all(u in toks for u in unit)


# Rule negatives (recipe v0.3.3): word ends gold v1 marks NO by convention, graded N (hard negative) whether or not they are candidates.
KO_CONNECTIVE_ENDINGS = ("니까", "고", "서", "면", "지만", "가지고", "가지구", "갖고", "다가", "면서")
KO_NEUDE = ("는데", "은데", "던데")     # mid-stream = NO in gold v1, but a turn ending in -는데 (trailing off) is AMBIG → conn_mid only
KO_FINAL_GO = ("더라고", "다고", "라고", "자고", "냐고")   # sentence-final -고 (retrospective / quotative), not a connective
NEG_RULES = ("reply_prefix", "conn_final", "conn_mid")


def _tok(t):
    return re.sub(r"[^\w'-]", "", str(t).lower())


def rule_negatives(stream, rules=NEG_RULES):
    """{i: rule} for recipe v0.3.3 rule negatives (gold v1 COMMIT / AMBIG / NO in brackets). reply_prefix: every word of a
    stream-initial run of reply words (RESPONSE_TOKENS) except the stream's last word [0/0/111 Korean, 0/0/6 English];
    conn_final: a Korean stream's last word ending in a connective ending (KO_CONNECTIVE_ENDINGS, not a sentence-final -고) with no
    sentence-final punctuation [1/2/54]; conn_mid: an earlier Korean word ending in a connective ending or -는데, unpunctuated [0/5/494]."""
    words, lang, out = stream["words"], stream.get("lang"), {}
    if "reply_prefix" in rules:
        toks = RESPONSE_TOKENS.get(lang, ())
        for p, w in enumerate(words[:-1]):
            if _tok(w["text"]) not in toks:
                break
            out[p] = "reply_prefix"
    def conn(w, endings):
        t = _tok(w["text"])
        return "punct_final" not in (w.get("tags") or []) and not t.endswith(KO_FINAL_GO) and any(t.endswith(e) and len(t) > len(e) for e in endings)
    if lang == "Korean" and words:
        if "conn_final" in rules and conn(words[-1], KO_CONNECTIVE_ENDINGS):
            out[len(words) - 1] = "conn_final"
        if "conn_mid" in rules:
            for i, w in enumerate(words[:-1]):
                if i not in out and conn(w, KO_CONNECTIVE_ENDINGS + KO_NEUDE):
                    out[i] = "conn_mid"
    return out


def branch_of(f):
    """v0.3 grading branch: punct_resp (reply-word unit before more speech), punct (other punctuated sentence ends), other."""
    return "punct_resp" if f.get("resp_head") else "punct" if f["punct"] else "other"


def grade_v3(f, lang, thresholds=None):
    """Recipe v0.3 grade from candidate_features → (grade, why). N only from human transcript disfluency tags (gold v1: LLM-made
    N was 42–68 % correct); a Stage-A-only disfluency conflict or any missing score → B; A when the branch's thresholds hold
    (branch_of: punct_resp / punct / other; a branch set to None is off); everything else B (masked, not a trained negative)."""
    th = (thresholds or V3_THRESHOLDS)[lang]
    d = f.get("disfluency")
    if d and str(d).startswith("tag_"):
        return "N", f"disfl_{d}"
    if d:
        return "B", f"disflA_{d}"
    if f["p_safe_mean"] is None or f["p_rev"] is None:
        return "B", "missing:" + ",".join([j for j, v in f["p_safe"].items() if v is None] + (["C"] if f["p_rev"] is None else []))
    br = branch_of(f); t = th.get(br) if br in th or br != "punct_resp" else th.get("punct")   # thresholds without punct_resp (v0.3/v0.3.1) grade it as punct
    if t and f["p_safe_mean"] >= t["p_safe"] and f["p_rev"] <= t["p_rev"]:
        return "A", f"v3_{br}"
    why = [f"v3_{br}"] + (["off"] if not t else (["p_safe_low"] if f["p_safe_mean"] < t["p_safe"] else []) + (["p_rev_high"] if f["p_rev"] > t["p_rev"] else []))
    return "B", "+".join(why)


def build_labels_v3(streams, a_rows, b_rows, c_rows, judges=None, extra=EXTRA_SOURCES, thresholds=None, turn_end=False, neg_rules=NEG_RULES):
    """Recipe v0.3 labels.jsonl rows + stats: candidates = Stage A ∪ extra sources, grade_v3 per candidate. Same row fields
    as v0.2 (grade/why/stageA/B/C) plus p_safe/p_safe_mean/p_rev/punct/sources, so SemCommitDataset and the metrics read them
    unchanged. Stage C rows must use next_candidate = next Stage A boundary (stage_c_items) — checked like v0.2.
    neg_rules (v0.3.3, rule_negatives): those word ends are N (why rule_<name>) — a candidate's grade is overridden unless it is
    already N, and a non-candidate gets an N row (stageA false, sources [rule_<name>]; SemCommitDataset gives it hardneg_weight).
    neg_rules=() reproduces v0.3–v0.3.2."""
    judges = {**{k: tuple(v) for k, v in PRIMARY_JUDGES.items()}, **(judges or {})}
    cs, bc, cc = candidate_sets(streams, a_rows, extra), by_candidate(list(b_rows)), by_candidate(list(c_rows))
    a_out = {sid: out for sid, (_, out) in candidates_of(a_rows).items()}
    labels, problems, seen = [], [], set()
    st = dict(recipe=RECIPE_V3, prompt_version=prompt_version(), judges={k: list(v) for k, v in judges.items()}, extra=list(extra),
              thresholds=thresholds or V3_THRESHOLDS, neg_rules=list(neg_rules), streams=Counter(), by_lang={}, why={}, sources={})
    for s in streams:
        st["streams"]["total"] += 1
        if s["id"] in seen:
            st["streams"]["dup_id_skipped"] += 1; continue
        seen.add(s["id"])
        if s["id"] not in cs:
            st["streams"]["no_stageA"] += 1; continue
        lang = s["lang"]; L = st["by_lang"].setdefault(lang, Counter()); js = judges[lang]
        allc, sa, src = cs[s["id"]]; sas = sorted(sa); rows = []
        rn = rule_negatives(s, neg_rules) if neg_rules else {}
        for i in sorted(set(allc) | set(rn)):
            crow = next(iter(sorted(cc.get((s["id"], i), {}).items())), (None, None))[1]
            if crow is not None and "next_candidate" in crow and crow["next_candidate"] != next_stage_a(sas, i):
                problems.append(f"{s['id']}@{i}: Stage C next_candidate {crow['next_candidate']} != next Stage A boundary {next_stage_a(sas, i)}")
            sources = list(src.get(i, [])) + ([f"rule_{rn[i]}"] if i in rn else [])
            f = candidate_features(s, i, bc.get((s["id"], i), {}), crow, js, sources, a_out.get(s["id"]))
            g, w = grade_v3(f, lang, thresholds) if i in src else ("N", f"rule_{rn[i]}")
            if i in rn and g != "N":
                g, w = "N", f"rule_{rn[i]}"
            B = {j: r["decision"] for j, r in bc.get((s["id"], i), {}).items()}
            C = {"relation": crow["relation"], "type": crow.get("type") or ""} if crow else {"relation": "MISSING", "type": ""}
            rows.append(dict(after_word=i, grade=g, why=w, stageA=i in sa, sources=sources, B=B, C=C, p_safe=f["p_safe"], p_safe_mean=f["p_safe_mean"],
                             p_rev=f["p_rev"], punct=f["punct"], resp_head=f["resp_head"], future_unobserved=C["relation"] == "UNOBSERVED", resolved_by_judge=False))
            L["candidates"] += 1; L[g] += 1; L["future_unobserved"] += rows[-1]["future_unobserved"]
            st["why"].setdefault(lang, Counter())[w] += 1
            for k in sources:
                st["sources"].setdefault(lang, Counter())[f"{k}:{g}"] += 1
        st["streams"]["labeled"] += 1; L["streams"] += 1; L["words"] += len(s["words"])
        labels.append(dict(id=s["id"], lang=lang, candidates=rows, turn_end=bool(turn_end)))
    if problems:
        raise ValueError(f"{len(problems)} Stage C rows do not match the candidate set: " + "; ".join(problems[:5]))
    st["streams"] = dict(st["streams"]); st["by_lang"] = {k: dict(v) for k, v in st["by_lang"].items()}
    st["why"] = {k: dict(v) for k, v in st["why"].items()}; st["sources"] = {k: dict(v) for k, v in st["sources"].items()}
    return labels, st


def build_labels(streams, a_rows, b_rows, c_rows, t_rows=(), strict=True, judges=None, disfl_negatives=False,
                 turn_end=True, c_mask_types=C_MASK_TYPES):
    """labels.jsonl rows + stats. Streams without an ok Stage A row get no labels row (counted in stats); a repeated
    stream id is labeled once. Every candidate row has C = {relation: STABLE|REVISION|UNOBSERVED|MISSING, type: str}.
    Consistency (fail closed, one ValueError listing the problems): an ok Stage A row must match its stream (n_words
    and, when present, word_texts_sha256) and a Stage C row's next_candidate (when present) must equal the next Stage A
    boundary — rows made from another words.jsonl or Stage A file would put labels at the wrong words.
    strict: a B/T/C decision whose scored ids differed from the joint tokenization (tok_ok/type_tok_ok False) counts as
    missing (→ B); stats['tok_mismatch'] counts them per stage/judge either way.
    disfl_negatives (ablation, off by default): also emit stageA=false N rows at disfluency word ends Stage A did not
    propose. SemCommitDataset gives every N row hardneg_weight, so this breaks the contract that non-candidate word ends
    are implicit negatives with default weight (Kspon: ≈6% of word ends go from NEXT 0.15 to 1.0)."""
    judges = {**{k: tuple(v) for k, v in PRIMARY_JUDGES.items()}, **(judges or {})}
    cands, bc, cc = candidates_of(a_rows), by_candidate(list(b_rows) + list(t_rows)), by_candidate(c_rows)
    a_ok = {r["id"]: r for r in a_rows if r.get("status") == "ok"}
    a_status = {r["id"]: r.get("status") for r in a_rows if r["id"] not in cands}
    labels, why, problems = [], {}, []
    row_status = {}
    for r in list(a_rows) + list(b_rows) + list(t_rows) + list(c_rows):
        row_status.setdefault(f"{r.get('stage')}/{r.get('judge')}", Counter())[r.get("status")] += 1
    st = dict(prompt_version=prompt_version(), strict=strict, judges={k: list(v) for k, v in judges.items()},
              tiebreak_judge=TIEBREAK_JUDGE, disfl_negatives=disfl_negatives, c_mask_types=list(c_mask_types),
              streams=Counter(), by_lang={}, why={}, C_relation={}, C_type={}, agreement={}, tok_mismatch=Counter(),
              row_status={k: dict(v) for k, v in sorted(row_status.items())})
    pairs, margins, seen = {}, {}, set()
    for s in streams:
        st["streams"]["total"] += 1
        if s["id"] in seen:   # one labels row per stream id (the dataset reader rejects duplicates)
            st["streams"]["dup_id_skipped"] += 1
            continue
        seen.add(s["id"])
        lang = s["lang"]
        L = st["by_lang"].setdefault(lang, Counter())
        if s["id"] not in cands:
            st["streams"]["stageA_" + (a_status.get(s["id"]) or "missing")] += 1
            continue
        ar = a_ok[s["id"]]
        if ar.get("n_words") is not None and ar["n_words"] != len(s["words"]):
            problems.append(f"{s['id']}: Stage A n_words {ar['n_words']} != {len(s['words'])} words in the stream")
            continue
        if ar.get("word_texts_sha256") and ar["word_texts_sha256"] != word_texts_sha256(s["words"]):
            problems.append(f"{s['id']}: Stage A word_texts_sha256 differs from the stream's word texts")
            continue
        st["streams"]["labeled"] += 1
        bounds, a_out = cands[s["id"]]
        tags = [w.get("tags") or [] for w in s["words"]]
        js = judges[lang]
        rows = []
        for k, i in enumerate(bounds):
            nc = bounds[k + 1] if k + 1 < len(bounds) else None
            b = {}
            for j, r in sorted(bc.get((s["id"], i), {}).items()):
                if _tok_bad(r):
                    st["tok_mismatch"][f"{r.get('stage')}/{j}"] += 1
                    if strict:
                        continue
                b[j] = r
            B, Bm = {j: r["decision"] for j, r in b.items()}, {j: r.get("margin") for j, r in b.items()}
            c = []   # one Stage C judge expected; first by name otherwise
            for j, r in sorted(cc.get((s["id"], i), {}).items()):
                if "next_candidate" in r and r["next_candidate"] != nc:
                    problems.append(f"{s['id']}@{i}: Stage C row ({j}) used next_candidate {r['next_candidate']}, "
                                    f"Stage A gives {nc}")
                if _tok_bad(r):
                    st["tok_mismatch"][f"C/{j}"] += 1
                    if strict:
                        continue
                c.append(r)
            C = {"relation": c[0]["relation"], "type": c[0].get("type") or ""} if c else {"relation": "MISSING", "type": ""}
            cand = dict(after_word=i, stageA=True, B=B, C=C, disfluency=disfluency_reason(i, a_out, tags))
            g, w, res = grade(cand, lang, js, strict, c_mask_types=c_mask_types)
            rows.append(dict(after_word=i, grade=g, why=w, stageA=True, B=B, B_margin=Bm, C=C,
                             future_unobserved=C["relation"] == "UNOBSERVED", resolved_by_judge=res))
            if len(js) >= 2 and all(j in B for j in js[:2]):
                pairs.setdefault((lang, f"{js[0]}~{js[1]}"), []).append((B[js[0]], B[js[1]]))
            L["candidates"] += 1
            L[g] += 1
            L["future_unobserved"] += rows[-1]["future_unobserved"]
            L["future_unobserved_A"] += rows[-1]["future_unobserved"] and g == "A"
            L["resolved_by_judge"] += res
            L["B_disagree"] += len({B[j] for j in js if j in B}) > 1
            for j, m in Bm.items():
                if m is not None:
                    margins.setdefault(j, []).append(m)
            why.setdefault(lang, Counter())[w] += 1
            st["C_relation"].setdefault(lang, Counter())[C["relation"]] += 1
            st["C_type"].setdefault(lang, Counter())[f"{C['relation']}/{C['type'] or '-'}:{g}"] += 1
        if disfl_negatives:
            for i in range(len(s["words"])):
                r = disfluency_reason(i, a_out, tags)
                if r and i not in bounds:
                    rows.append(dict(after_word=i, grade="N", why=f"disfl_{r}", stageA=False, B={}, B_margin={},
                                     C={"relation": "MISSING", "type": ""}, future_unobserved=False,
                                     resolved_by_judge=False))
                    L["N_disfl_only"] += 1
        L["streams"] += 1
        L["words"] += len(s["words"])
        rows.sort(key=lambda r: r["after_word"])
        labels.append(dict(id=s["id"], lang=lang, candidates=rows, turn_end=bool(turn_end)))
    if problems:
        raise ValueError(f"{len(problems)} Stage A/C rows do not match the streams (wrong words.jsonl or Stage A "
                         f"file?): " + "; ".join(problems[:5]))
    for key in ("why", "C_relation", "C_type"):
        st[key] = {k: dict(v) for k, v in (why if key == "why" else st[key]).items()}
    st["by_lang"] = {k: dict(v) for k, v in st["by_lang"].items()}
    st["streams"], st["tok_mismatch"] = dict(st["streams"]), dict(st["tok_mismatch"])
    st["B_margin"] = {j: dict(n=len(m), p10=_q(m, .1), p50=_q(m, .5)) for j, m in sorted(margins.items())}
    st["agreement"] = {f"{lang}/{name}": dict(n=len(p), agree=sum(a == b for a, b in p), kappa=_kappa(p))
                       for (lang, name), p in pairs.items()}
    return labels, st


def check_provenance(words_digest, inputs):
    """grade 입력 출처 검사 (fail closed). inputs: [(stage, path, rows)]. Every row file needs its <out>.fingerprint.json
    with the expected stage and words_sha256 == words_digest (after_word indices refer to that words.jsonl), and every
    fingerprint and row must carry one shared prompt_version. Returns {prompt_version, files: {path: summary}}."""
    pvs, files = set(), {}
    for stage, path, rows in inputs:
        fp_path = Path(path).with_suffix(".fingerprint.json")
        if not fp_path.exists():
            raise ValueError(f"{path}: no fingerprint sidecar {fp_path.name} (grade takes semcommit_teacher outputs only)")
        fp = json.loads(fp_path.read_text())
        if fp.get("stage") != stage:
            raise ValueError(f"{path}: fingerprint stage {fp.get('stage')!r}, expected {stage!r}")
        if fp.get("words_sha256") != words_digest:
            raise ValueError(f"{path}: computed on words sha256 {str(fp.get('words_sha256'))[:12]}, but --words has "
                             f"{words_digest[:12]} (rebuilt words.jsonl → rerun the stages)")
        pv = fp.get("prompt_version")
        bad = sorted({str(r.get("prompt_version")) for r in rows} - {pv})
        if bad:
            raise ValueError(f"{path}: rows with prompt_version {bad} != fingerprint {pv}")
        pvs.add(pv)
        files[str(path)] = dict(stage=stage, judge=fp.get("judge"), kind=fp.get("kind"), model_path=fp.get("model_path"),
                                prompt_version=pv, rows=len(rows))
    if len(pvs) > 1:
        raise ValueError(f"grade inputs mix prompt versions {sorted(map(str, pvs))}")
    return dict(prompt_version=next(iter(pvs), None), files=files)


# ───────────────────────── resumable JSONL IO ─────────────────────────
def read_jsonl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_words(path, lang_filter="all", limit=None):
    rows = [r for r in read_jsonl(path) if lang_filter in (None, "all") or r["lang"] == lang_filter]
    return rows[:limit] if limit else rows


def encode_rows(rows):
    """JSONL lines; ValueError on a non-finite float (allow_nan=False) before anything is written."""
    return [json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


class JsonlStore:
    """Append-only row file bound to a fingerprint sidecar (<out>.fingerprint.json). On resume the sidecar must match
    exactly (else ValueError naming the differing keys); a torn final line (crash mid-write) is cut off, any other
    malformed line fails closed. `done` holds keys of every row except status=inference_error (retried on resume)."""

    def __init__(self, path, fingerprint):
        self.path, self.fp_path = Path(path), Path(path).with_suffix(".fingerprint.json")
        fp = json.loads(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True))
        if self.fp_path.exists():
            old = json.loads(self.fp_path.read_text())
            if digest(old) != digest(fp):
                diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
                raise ValueError(f"resume fingerprint mismatch in {diff}; use a new --out path")
        else:
            if self.path.exists() and self.path.stat().st_size:
                raise ValueError(f"unfingerprinted existing output {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.fp_path, fp)
        self.path.touch()   # an empty output is a valid (complete) stage result, e.g. no tie-break candidates
        self.done, self.n_rows = set(), 0
        if self.path.exists():
            data = self.path.read_bytes()
            lines = data.split(b"\n")
            if lines and lines[-1]:   # no trailing newline → last line may be torn
                try:
                    json.loads(lines[-1])
                    with self.path.open("ab") as f:
                        f.write(b"\n")
                except ValueError:
                    print(f"WARNING: dropping torn final line of {self.path}", flush=True)
                    with self.path.open("r+b") as f:
                        f.truncate(len(data) - len(lines[-1]))
                    lines = lines[:-1]
            for line in lines:
                if line.strip():
                    r = json.loads(line)   # malformed middle line fails closed
                    self.n_rows += 1
                    if r.get("status") != "inference_error":
                        self.done.add(tuple(r["key"]))

    def write(self, rows, lines=None):
        """Append rows + flush + fsync. All rows are serialized first (encode_rows), so a bad row writes nothing."""
        lines = encode_rows(rows) if lines is None else lines
        with self.path.open("a", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        for r in rows:
            if r.get("status") != "inference_error":
                self.done.add(tuple(r["key"]))
        self.n_rows += len(rows)
