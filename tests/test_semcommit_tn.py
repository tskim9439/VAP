"""semcommit_tn.normalize_ko_row — Korean speechlm targets to asr-tn-v1: punctuation tokens dropped (times kept), words re-encoded when
the kept tokens do not spell the target, rows with digits / other characters / split words rejected, split_words agrees afterwards."""
import pytest

from vapasr.data.semcommit_tn import normalize_ko_row
from vapasr.data.semcommit_words import _B2U, split_words


class BPE:
    """Byte-level toy tokenizer: greedy longest match over a fixed piece list (Qwen-style leading-space pieces)."""
    def __init__(self, pieces):
        self.pieces = list(pieces); self.special_ids = set()
    def _bl(self, s): return "".join(_B2U[b] for b in s.encode("utf-8"))
    def convert_ids_to_tokens(self, ids): return [self._bl(self.pieces[i]) for i in ids]
    def __call__(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            k = max((k for k, p in enumerate(self.pieces) if text.startswith(p, i)), key=lambda k: len(self.pieces[k]))
            out.append(k); i += len(self.pieces[k])
        return {"input_ids": out}


P = ["안녕", "하세요", ".", " 여기", " 화곡동", "인데요", "?", " 도와", "주세요", "요.", " 10분", " A", "/S", "도와주세요", " 네"]
tok = BPE(P)


def row(pieces_times, words):
    toks = [[P.index(p), t] for p, t in pieces_times]
    ws, a = [], 0
    for i, (text, n, tags) in enumerate(words):
        ws.append(dict(i=i, text=text, a=a, b=a + n, end_time=toks[a + n - 1][1], seg=0, tags=tags)); a += n
    return dict(id="x", lang="Korean", text=" ".join(w[0] for w in words), tokens=toks, words=ws, K=40, duration_s=3.0, segments=[])


def test_punctuation_tokens_dropped_times_kept():
    r = row([("안녕", 0.3), ("하세요", 0.6), (".", 0.6), (" 여기", 1.0), (" 화곡동", 1.5), ("인데요", 1.8), ("?", 1.8)],
            [("안녕하세요", 3, ["punct_final"]), ("여기", 1, []), ("화곡동인데요", 3, ["punct_final"])])
    n = normalize_ko_row(r, tok)
    assert [P[t[0]] for t in n["tokens"]] == ["안녕", "하세요", " 여기", " 화곡동", "인데요"] and [t[1] for t in n["tokens"]] == [0.3, 0.6, 1.0, 1.5, 1.8]
    assert [(w["text"], w["a"], w["b"], w["end_time"]) for w in n["words"]] == [("안녕하세요", 0, 2, 0.6), ("여기", 2, 3, 1.0), ("화곡동인데요", 3, 5, 1.8)]
    assert n["words"][0]["tags"] == ["punct_final"] and n["text"] == "안녕하세요 여기 화곡동인데요" and n["textnorm"] == "asr-tn-v1"
    ws = split_words([t[0] for t in n["tokens"]], [t[1] for t in n["tokens"]], tok)
    assert [(w["text"], w["a"], w["b"]) for w in ws] == [(w["text"], w["a"], w["b"]) for w in n["words"]]


def test_merged_punctuation_piece_is_reencoded_with_word_times():
    t2 = BPE(P + ["요"])
    r = row([("안녕", 0.5), ("요.", 0.9)], [("안녕요", 2, ["punct_final"])])          # the period sits inside the piece '요.'
    r["tokens"] = [[t2.pieces.index(p), t] for p, t in (("안녕", 0.5), ("요.", 0.9))]
    n = normalize_ko_row(r, t2)
    assert [t2.pieces[t[0]] for t in n["tokens"]] == ["안녕", "요"] and [t[1] for t in n["tokens"]] == [0.5, 0.9]
    assert (n["words"][0]["text"], n["words"][0]["end_time"]) == ("안녕요", 0.9)


def test_already_normalized_row_is_returned_as_is():
    r = row([("안녕", 0.3), ("하세요", 0.6), (" 여기", 1.0)], [("안녕하세요", 2, []), ("여기", 1, [])])
    assert normalize_ko_row(r, tok) is r


@pytest.mark.parametrize("words,reason", [([(" 10분", "10분")], "tn_digit"), ([(" A", "A"), ("/S", None)], "tn_word_split")])
def test_rejected_rows(words, reason):
    if reason == "tn_digit":
        r = row([("안녕", 0.2), (" 10분", 0.8)], [("안녕", 1, []), ("10분", 1, [])])
    else:
        r = row([("안녕", 0.2), (" A", 0.5), ("/S", 0.8)], [("안녕", 1, []), ("A/S", 2, [])])
    assert normalize_ko_row(r, tok) == (None, reason)
