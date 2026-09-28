"""Korean semcommit training targets → asr-tn-v1 (vapasr/data/textnorm.py target_ko), applied when SemCommitDataset loads a row.

The speechlm words.jsonl rows were aligned on the Qwen3-ASR transcript (punctuation, Arabic digits), so their `tokens` (the training
target ids) carry '.', '?', ',' and digits, while Phase 1 Korean targets (kspon asr-tn-v1), the Korean evaluation sets (KsponSpeech,
AIHub lecture / telephone / counseling: digit- and punctuation-free human transcripts) and every English row have none. Per word:
  target   = target_ko(word text): punctuation → space, NFKC; a character outside Hangul/Latin/space or a word split by
             normalisation ('A/S') rejects the row.
  digits   = a word with Arabic digits takes its spoken form from the segment's human transcript (segments[].raw_text — the source
             DB's own transcript, e.g. Qwen '2007년이었대' ↔ human '이천칠년이었대'): character alignment (difflib) of the segment's
             Qwen words against the human text; the digit word's span must map cleanly (no differing block crossing a word
             boundary, the replaced Qwen characters contain a digit, the human side is Hangul). The human span can be several words
             ('1980년에' → '천구백팔십 년에'); labels then point at the last of them (word_map). Otherwise the row is rejected
             (tn_digit). Only digit words are replaced — other Qwen/human spelling differences are left as they are.
  tokens   = the word's original tokens without punctuation-only tokens, times kept, when they decode to the target; otherwise the word
             is re-encoded (tokenizer, leading space after the first word) and the new tokens take the word's original token times
             spread in order (last new token = the word's end time), so the word still ends in the same chunk (<SEM_END> placement).
Rows already in asr-tn-v1 form come back unchanged (kspon, idempotent). Returns the new row — with word_map (old word index → new index
of its last word) when a digit word became several words — or (None, reason) with reason in
tn_digit · tn_charset · tn_word_split · tn_word_empty · tn_decode.
"""
import difflib
import re
from typing import List, Optional

from .semcommit_words import _pieces, piece_bytes
from .textnorm import target_ko, _KO_ALLOWED

_PUNCT_ONLY = re.compile(r"^[^\w]*$")
_DIGIT = re.compile(r"\d")
_HANGUL = re.compile(r"^[가-힣]+$")
MIN_SEGMENT_MATCH = 0.8          # share of the segment's non-digit Qwen characters matched in the human transcript before trusting it


def _tok_bytes(tok, ids: List[int]):
    return [piece_bytes(p) for p in _pieces(tok, ids)]


def _decode(bs):
    try:
        return b"".join(bs).decode("utf-8")
    except (UnicodeDecodeError, TypeError):
        return None


def _encode(tok, text):
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _spread(times, n):
    """n new tokens over the word's original token times, in order, last = last original time."""
    L = len(times)
    return [times[min(L - 1, ((j + 1) * L + n - 1) // n - 1)] for j in range(n)]


def spoken_digit_words(qwords: List[str], raw: str) -> Optional[List[Optional[List[str]]]]:
    """For one segment: Qwen words (target_ko form, may hold digits) + the human transcript → per word None (keep) or the list of
    human words replacing it (digit words only). None when a digit word cannot be mapped cleanly."""
    human = target_ko(raw or "", "aihub")
    if not any(_DIGIT.search(w) for w in qwords):
        return [None] * len(qwords)
    if not human or _DIGIT.search(human):
        return None
    q = "".join(w.replace(" ", "") for w in qwords)
    pos, h = [], []                                    # human characters without spaces → index in `human`
    for i, c in enumerate(human):
        if c != " ":
            pos.append(i); h.append(c)
    h = "".join(h)
    sm = difflib.SequenceMatcher(None, q, h, autojunk=False)
    ops = sm.get_opcodes()
    matched = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal")
    if matched < MIN_SEGMENT_MATCH * max(1, len(_DIGIT.sub("", q))):
        return None
    out, a = [], 0
    for w in qwords:
        b = a + len(w.replace(" ", ""))
        if not _DIGIT.search(w):
            out.append(None); a = b; continue
        rs = re_ = None; digit_block = False
        for tag, i1, i2, j1, j2 in ops:
            inside = (i1 < b and i2 > a) if i1 < i2 else (a < i1 < b)     # insertions only strictly inside the word
            if not inside:
                continue
            if tag == "equal":
                lo, hi = max(i1, a), min(i2, b)
                if lo >= hi:
                    continue
                s_, e_ = j1 + (lo - i1), j1 + (hi - i1)
            else:
                if i1 < a or i2 > b:                   # a differing block crosses the word boundary — ambiguous
                    return None
                s_, e_ = j1, j2
                if tag in ("replace", "delete") and _DIGIT.search(q[i1:i2]):
                    if not _HANGUL.match(h[j1:j2] or "x"):
                        return None
                    digit_block = True
            rs = s_ if rs is None else min(rs, s_); re_ = e_ if re_ is None else max(re_, e_)
        if not digit_block or rs is None or re_ is None or re_ <= rs:
            return None
        span = human[pos[rs]:pos[re_ - 1] + 1].split()
        if not span or any(_DIGIT.search(x) or not _KO_ALLOWED.match(x) for x in span):
            return None
        out.append(span)
        a = b
    return out


def normalize_ko_row(r: dict, tok):
    words, toks = r["words"], r["tokens"]
    segs = r.get("segments") or []
    targets = []                                       # per old word: list of new word texts
    by_seg = {}
    for k, w in enumerate(words):
        t = target_ko(w["text"], "aihub")
        if not t:
            return None, "tn_word_empty"
        if " " in t and not _DIGIT.search(t):
            return None, "tn_word_split"
        targets.append([t])
        by_seg.setdefault(w.get("seg", 0), []).append(k)
    if any(_DIGIT.search(t[0]) for t in targets):
        for j, ks in by_seg.items():
            if not any(_DIGIT.search(targets[k][0]) for k in ks):
                continue
            raw = segs[j].get("raw_text") if j < len(segs) else None
            rep = spoken_digit_words([targets[k][0] for k in ks], raw or "")
            if rep is None:
                return None, "tn_digit"
            for k, x in zip(ks, rep):
                if x is not None:
                    targets[k] = x
    for t in targets:
        for x in t:
            if " " in x:
                return None, "tn_word_split"
            if _DIGIT.search(x):
                return None, "tn_digit"
            if not _KO_ALLOWED.match(x):
                return None, "tn_charset"
    all_ids = [int(x[0]) for x in toks]
    same_text = all(t == [w["text"]] for t, w in zip(targets, words))
    bs = _tok_bytes(tok, all_ids) if callable(getattr(tok, "convert_ids_to_tokens", None)) else None
    if bs is None:
        if same_text:
            return r                                   # cannot inspect pieces (test tokenizer) and nothing to change
        return None, "tn_decode"
    new_toks, new_words, word_map, changed = [], [], [], False
    for k, (w, ts) in enumerate(zip(words, targets)):
        seg = toks[w["a"]:w["b"]]; seg_b = bs[w["a"]:w["b"]]
        groups = None
        if len(ts) == 1:                                   # same word: keep its tokens minus punctuation-only ones when they spell it
            t = ts[0]; want = t if k == 0 else " " + t
            keep = [(x, b) for x, b in zip(seg, seg_b) if b is None or _decode([b]) is None or not _PUNCT_ONLY.match(_decode([b]))]
            dec = _decode([b for _, b in keep]) if keep else None
            if dec is not None and dec.strip() == t and (dec == want or (k == 0 and dec.strip() == dec)):
                groups = [[[int(x[0]), float(x[1])] for x, _ in keep]]
        if groups is None:                                 # re-encode the word(s) and spread the old word's token times over them
            ids = [_encode(tok, (x if k == 0 and i == 0 else " " + x)) for i, x in enumerate(ts)]
            if not seg or not all(ids):
                return None, "tn_decode"
            it = iter(_spread([float(x[1]) for x in seg], sum(len(g) for g in ids)))
            groups = [[[i, next(it)] for i in g] for g in ids]
        for i, (g, x) in enumerate(zip(groups, ts)):
            a = len(new_toks); new_toks += g
            tags = w.get("tags", []) if i == len(ts) - 1 else []
            new_words.append(dict(w, text=x, a=a, b=len(new_toks), end_time=max(s for _, s in g), tags=tags))
        word_map.append(len(new_words) - 1)
        changed |= ts != [w["text"]] or [x for g in groups for x in g] != [[int(x[0]), float(x[1])] for x in seg]
    if not changed:
        return r
    for i, w in enumerate(new_words):
        w["i"] = i
    out = dict(r, tokens=new_toks, words=new_words, text=" ".join(x for t in targets for x in t), textnorm="asr-tn-v1")
    if len(new_words) != len(words):
        out["word_map"] = word_map
    return out
