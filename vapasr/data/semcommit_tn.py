"""Korean semcommit training targets → asr-tn-v1 (vapasr/data/textnorm.py target_ko), applied when SemCommitDataset loads a row.

The speechlm words.jsonl rows were aligned on punctuated text, so their `tokens` (the training target ids) carry '.', '?', ','
while Phase 1 Korean targets (kspon asr-tn-v1) and every English row have none. Per word:
  target   = target_ko(word text): punctuation → space, NFKC; a word that would contain a digit, a character outside Hangul/Latin/space,
             or split into two words (internal punctuation, 'A/S') rejects the row — word indices must not move, labels point at them.
  tokens   = the word's original tokens without punctuation-only tokens, times kept, when they decode to the target; otherwise the word
             is re-encoded (tokenizer, leading space after the first word) and the new tokens take the word's original token times
             spread in order (last new token = the word's end time), so the word still ends in the same chunk (<SEM_END> placement).
Rows already in asr-tn-v1 form come back unchanged (kspon, idempotent). Returns the new row, or (None, reason) with reason in
tn_digit · tn_charset · tn_word_split · tn_word_empty · tn_decode.
"""
import re
from typing import List

from .semcommit_words import _pieces, piece_bytes
from .textnorm import target_ko, _KO_ALLOWED

_PUNCT_ONLY = re.compile(r"^[^\w]*$")


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


def normalize_ko_row(r: dict, tok):
    words, toks = r["words"], r["tokens"]
    targets = []
    for w in words:
        t = target_ko(w["text"], "aihub")
        if not t:
            return None, "tn_word_empty"
        if " " in t:
            return None, "tn_word_split"
        if re.search(r"\d", t):
            return None, "tn_digit"
        if not _KO_ALLOWED.match(t):
            return None, "tn_charset"
        targets.append(t)
    all_ids = [int(x[0]) for x in toks]
    same_text = all(t == w["text"] for t, w in zip(targets, words))
    bs = _tok_bytes(tok, all_ids) if callable(getattr(tok, "convert_ids_to_tokens", None)) else None
    if bs is None:
        if same_text:
            return r                                   # cannot inspect pieces (test tokenizer) and nothing to change
        return None, "tn_decode"
    new_toks, new_words, changed = [], [], False
    for k, (w, t) in enumerate(zip(words, targets)):
        seg = toks[w["a"]:w["b"]]; seg_b = bs[w["a"]:w["b"]]
        want = t if k == 0 else " " + t
        keep = [(x, b) for x, b in zip(seg, seg_b) if b is None or _decode([b]) is None or not _PUNCT_ONLY.match(_decode([b]))]
        dec = _decode([b for _, b in keep]) if keep else None
        if dec is not None and dec.strip() == t and (dec == want or (k == 0 and dec.strip() == dec)):
            out = [[int(x[0]), float(x[1])] for x, _ in keep]
        else:
            ids = _encode(tok, want)
            if not ids:
                return None, "tn_decode"
            out = [[i, s] for i, s in zip(ids, _spread([float(x[1]) for x in seg], len(ids)))]
        changed |= out != [[int(x[0]), float(x[1])] for x in seg] or t != w["text"]
        a = len(new_toks); new_toks += out
        new_words.append(dict(w, text=t, a=a, b=len(new_toks), end_time=max(s for _, s in out)))
    if not changed:
        return r
    return dict(r, tokens=new_toks, words=new_words, text=" ".join(targets), textnorm="asr-tn-v1")
