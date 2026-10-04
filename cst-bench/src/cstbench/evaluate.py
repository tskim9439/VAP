"""Scoring of streaming translation outputs on rendered sessions.

System output (hyp.jsonl): one committed text piece per line, never retracted.
  {"session": "taxi/SES0037", "lang": "English", "t": 3.42, "text": "which exit"}
  t        session time (s, from the start of mix.wav) at which the piece was emitted; with the
           ideal (non computation-aware) clock this is the amount of audio consumed so far
  lang     output language; it selects the translation direction (English output = German->English)
  turn_id  optional; when given (oracle segmentation) pieces are assigned to that turn directly
Any other field (e.g. "elapsed" for a computation-aware clock) is kept; --time-key picks the clock.

Per session and output language, the hypothesis words are aligned to the concatenated reference
translations of the turns in that direction with a word-level Levenshtein alignment (minimum edit
distance, the criterion of mwerSegmenter); every hypothesis word goes to the turn of its aligned
reference word, and inserted words to the turn of the preceding aligned word.

Metrics per direction:
  BLEU, chrF       sacrebleu corpus scores over turns, after normalisation (see `normalize`)
  StreamLAAL       LAAL per turn on the resegmented output, delays measured from the turn start (s)
  end_offset       time from the end of the source turn to the last word of its translation (s);
                   how long the listener waits after the speaker stopped
  empty_turns      turns that received no output
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
import json
import re
import statistics
import unicodedata

LANG_CODES = {"English": "en", "German": "de", "Korean": "ko", "Chinese": "zh"}
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_PUNCT = re.compile(r"[^\w\s']|_|(?<!\w)'|'(?!\w)")


def normalize(text: str, lang: str) -> str:
    """Lower case, numbers spelled out (num2words, if installed), punctuation removed.

    TAXI references are written without punctuation and with spelled-out numbers.
    """
    s = unicodedata.normalize("NFC", text)
    code = LANG_CODES.get(lang, lang)
    try:
        from num2words import num2words

        def spell(m):
            x = m.group(0).replace(",", ".") if code == "de" else m.group(0).replace(",", "")
            try:
                return " " + num2words(float(x) if "." in x else int(x), lang=code) + " "
            except (NotImplementedError, ValueError, OverflowError):
                return m.group(0)
        s = _NUM.sub(spell, s)
    except ImportError:
        pass
    s = _PUNCT.sub(" ", s.lower())
    return " ".join(s.split())


def align(hyp: Sequence[str], ref: Sequence[str]) -> List[int]:
    """Levenshtein alignment; returns for each hyp word the aligned ref index, or -1 if inserted."""
    n, m = len(hyp), len(ref)
    d = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        di, dp, h = d[i], d[i - 1], hyp[i - 1]
        for j in range(1, m + 1):
            di[j] = min(dp[j - 1] + (h != ref[j - 1]), dp[j] + 1, di[j - 1] + 1)
    out, i, j = [-1] * n, n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (hyp[i - 1] != ref[j - 1]):
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def resegment(hyp_words: Sequence[str], ref_turns: Sequence[Sequence[str]]) -> List[List[int]]:
    """Assign hypothesis word indices to reference turns (see module doc)."""
    owner = [k for k, words in enumerate(ref_turns) for _ in words]
    a = align(hyp_words, [w for words in ref_turns for w in words])
    out: List[List[int]] = [[] for _ in ref_turns]
    if not ref_turns:
        return out
    first = next((owner[j] for j in a if j >= 0), 0)
    cur = first
    for i, j in enumerate(a):
        if j >= 0:
            cur = owner[j]
        out[cur].append(i)
    return out


def laal(delays: Sequence[float], src_len: float, ref_len: int) -> float:
    """Length-adaptive average lagging (Papi et al. 2022) in the units of `delays`."""
    if not delays:
        return float("nan")
    rate = src_len / max(len(delays), ref_len)
    tau = next((j + 1 for j, x in enumerate(delays) if x >= src_len), len(delays))
    return sum(delays[j] - j * rate for j in range(tau)) / tau


def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": None, "median": None, "p90": None}
    ys = sorted(xs)
    return {"mean": round(statistics.fmean(ys), 3), "median": round(statistics.median(ys), 3),
            "p90": round(ys[min(len(ys) - 1, int(0.9 * len(ys)))], 3)}


def evaluate(sessions: Sequence[dict], hyps: Sequence[dict], time_key: str = "t") -> Tuple[dict, List[dict]]:
    """sessions: rendered timelines (sessions.jsonl records); hyps: hyp.jsonl records -> (report, segments)."""
    import sacrebleu

    by_sess: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for h in hyps:
        by_sess[(h["session"], h["lang"])].append(h)
    known = {s["session_id"] for s in sessions}
    unknown = sorted({k[0] for k in by_sess} - known)
    segments: List[dict] = []
    wrong_turn_lang = 0
    for s in sessions:
        turns = [t for t in s["turns"] if t.get("translation")]
        for lang in sorted({t["translation_lang"] for t in turns}):
            tt = [t for t in turns if t["translation_lang"] == lang]
            pieces = sorted(by_sess.get((s["session_id"], lang), []), key=lambda h: h[time_key])
            words: List[Tuple[str, float, str]] = []
            for h in pieces:
                words += [(w, float(h[time_key]), h.get("turn_id")) for w in normalize(h["text"], lang).split()]
            refs = [normalize(t["translation"], lang).split() for t in tt]
            if words and all(w[2] is not None for w in words):        # oracle segmentation
                idx = {t["turn_id"]: k for k, t in enumerate(tt)}
                groups: List[List[int]] = [[] for _ in tt]
                for i, w in enumerate(words):
                    if w[2] in idx:
                        groups[idx[w[2]]].append(i)
                    else:
                        wrong_turn_lang += 1
            else:
                groups = resegment([w[0] for w in words], refs)
            for t, ref, g in zip(tt, refs, groups):
                times = [words[i][1] for i in g]
                src_len = t["end_s"] - t["start_s"]
                segments.append(dict(session=s["session_id"], turn_id=t["turn_id"], src_lang=t["lang"], lang=lang,
                                     src=t["transcript"], ref=" ".join(ref), hyp=" ".join(words[i][0] for i in g),
                                     laal=laal([max(x - t["start_s"], 0.0) for x in times], src_len, len(ref)),
                                     end_offset=(times[-1] - t["end_s"]) if times else None))
    report = {"directions": {}, "unknown_sessions": unknown, "wrong_language_words": wrong_turn_lang}
    for lang in sorted({x["lang"] for x in segments}):
        seg = [x for x in segments if x["lang"] == lang]
        src = sorted({x["src_lang"] for x in seg})
        hyp, ref = [x["hyp"] for x in seg], [[x["ref"] for x in seg]]
        report["directions"][f"{'+'.join(src)}->{lang}"] = {
            "turns": len(seg),
            "bleu": round(sacrebleu.corpus_bleu(hyp, ref).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyp, ref).score, 2),
            "stream_laal_s": _stats([x["laal"] for x in seg if x["end_offset"] is not None]),
            "end_offset_s": _stats([x["end_offset"] for x in seg if x["end_offset"] is not None]),
            "empty_turns": sum(x["end_offset"] is None for x in seg),
        }
    return report, segments


def load_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
