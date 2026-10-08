"""German ASR text normalisation `asr-tn-de-v1.0.0` — training targets, row conversion and scoring for German.

Kept apart from vapasr/data/textnorm.py (asr-tn-v1, EN/KO) so that the EN/KO code and fingerprints stay unchanged.
The rules follow the English target of asr-tn-v1:
  target_de: tags removed → NFKC → lowercase (ẞ → ß) → ’ ‘ → ' (kept inside a word only) → hyphen · dash · slash → space →
             other punctuation and symbols → space (letters of any script stay, so a foreign letter is flagged as charset
             instead of disappearing). Digits are not converted (a year, a cardinal or an ordinal cannot be told apart
             without the audio), so a row whose target still has a digit is quarantined, as in English v1.
  flags:     empty · digit · symbol (% € $ & + @ § # = * \\ — a word was written as a sign) · abbrev (Dr. bzw. z.B. etc. — the
             spoken form is the full word) · charset (a letter outside German/Latin-1 letters) · control.
  score_de:  target_de applied to reference and hypothesis alike (no number conversion; evaluation sets drop digit rows).
normalize_de_row converts a words.jsonl row (experiments/yodas_de_align.py: Qwen3-ASR raw output with punctuation and case,
token times from the forced aligner) into the asr-tn-de-v1 target. Per lexical word: target_de of the word, split on hyphens;
its original tokens are kept without punctuation-only tokens when they already spell the target, otherwise the word(s) are
re-encoded (leading space after the first word) and take the word's original token times spread in order, so the word still
ends in the same chunk (as vapasr/data/semcommit_tn.normalize_ko_row does for Korean). Returns the new row or (None, reason).
"""
import re
import unicodedata
from typing import Optional, Set

from .semcommit_tn import _PUNCT_ONLY, _decode, _encode, _spread, _tok_bytes

TEXTNORM_DE_VERSION = "asr-tn-de-v1.1.0"         # v1.1.0: non-Latin letters (θ, ş) flag the row instead of being removed (v1.0.0 tn1, unused)
TEXTNORM_DE_ID = "asr-tn-de-v1"

_TAG = re.compile(r"<[^>\s]{1,20}>")
_WS = re.compile(r"\s+")
_LETTERS = "a-zäöüßàâæçéèêëîïôœùûÿáíóúñ"
_ALLOWED = re.compile(rf"^[{_LETTERS}' ]*$")
_PUNCT = re.compile(r"[^\w'\s]|_")
_SYMBOL = re.compile(r"[%€$&+@§#=*\\]")
_DIGIT = re.compile(r"\d")
# Abbreviations whose spoken form is the full word ("Dr." = Doktor): two or more one-to-four letter groups with periods (z.B., d.h.,
# u.a., e.V., www.x.de) or a listed abbreviation followed by a period.
_ABBREV = re.compile(r"(?:^|[\s(\"„])(?:(?:[^\W\d_]{1,4}\.){2,}|(?:bzw|usw|ca|dr|nr|mio|mrd|evtl|ggf|inkl|bspw|etc|vs|prof|st|hr|fr|abs|vgl|bzgl|ggü|sog|zzgl|max|min|tel|str)\.)",
                     re.I)


def _clean(t: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", _TAG.sub(" ", t))).strip()


def target_de(t: str) -> str:
    t = _clean(t).replace("ẞ", "ß").lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[-–—/]", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)
    return _WS.sub(" ", _PUNCT.sub(" ", t)).strip()


def score_de(t: str) -> str:
    return target_de(t)


def target_flags_de(raw: str) -> Set[str]:
    """Quarantine reasons for a raw transcript (empty set = trainable)."""
    f: Set[str] = set()
    if any(unicodedata.category(c) == "Cc" for c in raw):
        f.add("control")
    if _SYMBOL.search(raw):
        f.add("symbol")
    if _ABBREV.search(raw):
        f.add("abbrev")
    t = target_de(raw)
    if not t:
        f.add("empty")
    if _DIGIT.search(t):
        f.add("digit")
    elif t and not _ALLOWED.match(t):
        f.add("charset")
    return f


def normalize_de_row(r: dict, tok):
    """words.jsonl row (raw Qwen tokens) → asr-tn-de-v1 row, or (None, reason). reason: tn_<flag> · tn_word_empty · tn_decode."""
    raw = r.get("display_text") or r.get("text") or ""
    flags = target_flags_de(raw)
    if flags:
        return None, "tn_" + sorted(flags)[0]
    words, toks = r["words"], r["tokens"]
    targets = []
    for w in words:
        t = target_de(w["text"])
        if not t:
            return None, "tn_word_empty"
        targets.append(t.split(" "))
    bs = _tok_bytes(tok, [int(x[0]) for x in toks])
    new_toks, new_words, word_map = [], [], []
    for k, (w, ts) in enumerate(zip(words, targets)):
        seg, seg_b = toks[w["a"]:w["b"]], bs[w["a"]:w["b"]]
        groups = None
        if len(ts) == 1:                                   # tokens minus punctuation-only ones already spell the target
            want = ts[0] if k == 0 else " " + ts[0]
            keep = [(x, b) for x, b in zip(seg, seg_b) if b is None or _decode([b]) is None or not _PUNCT_ONLY.match(_decode([b]))]
            if keep and _decode([b for _, b in keep]) == want:
                groups = [[[int(x[0]), float(x[1])] for x, _ in keep]]
        if groups is None:
            ids = [_encode(tok, x if k == 0 and i == 0 else " " + x) for i, x in enumerate(ts)]
            if not seg or not all(ids):
                return None, "tn_decode"
            it = iter(_spread([float(x[1]) for x in seg], sum(len(g) for g in ids)))
            groups = [[[i_, next(it)] for i_ in g] for g in ids]
        for i, (g, x) in enumerate(zip(groups, ts)):
            a = len(new_toks)
            new_toks += g
            tags = w.get("tags", []) if i == len(ts) - 1 else []
            new_words.append(dict(w, text=x, a=a, b=len(new_toks), end_time=max(s for _, s in g), tags=tags))
        word_map.append(len(new_words) - 1)
    for i, w in enumerate(new_words):
        w["i"] = i
    out = dict(r, tokens=new_toks, words=new_words, text=" ".join(w["text"] for w in new_words), textnorm=TEXTNORM_DE_ID)
    if len(new_words) != len(words):
        out["word_map"] = word_map
    return out
