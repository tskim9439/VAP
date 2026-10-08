import os

import pytest

from vapasr.data.single_turn_eval import canonical_reference, primary_metric, score_pair
from vapasr.data.textnorm_de import normalize_de_row, target_de, target_flags_de


def test_target_de_follows_english_rules():
    assert target_de("Ich mache den YouTube-Kanal. Es macht mir Spaß!") == "ich mache den youtube kanal es macht mir spaß"
    assert target_de("Na, wie geht's dir? „Gut“, sagt er – ganz ruhig.") == "na wie geht's dir gut sagt er ganz ruhig"
    assert target_de("STRAẞE") == "straße"


@pytest.mark.parametrize("raw,flag", [("also 300 Gramm", "digit"), ("Herr Dr. Martin", "abbrev"), ("z.B. so", "abbrev"),
                                      ("RE/MAX & Co", "symbol"), ("Randwinkel θ", "charset"), ("…", "empty")])
def test_flags(raw, flag):
    assert flag in target_flags_de(raw)


def test_clean_text_has_no_flags():
    assert target_flags_de("Daran hatten Sie doch Spaß.") == set()


def test_german_scoring():
    assert canonical_reference("Daran hatten Sie doch Spaß.", "German") == "daran hatten sie doch spaß"
    assert primary_metric("German") == "wer" and primary_metric("Korean") == "cer_nospace"
    m = score_pair("daran hatten sie doch spaß", "Daran hatten Sie Spaß!", "German")
    assert set(m) == {"wer"} and m["wer"]["errors"] == 1 and m["wer"]["n_ref"] == 5


QWEN = "/soundai/Model/Qwen3-ASR-1.7B"


@pytest.mark.skipif(not os.path.isdir(QWEN), reason="needs the Qwen3-ASR tokenizer")
def test_normalize_row_keeps_word_end_times():
    from transformers import AutoTokenizer
    from vapasr.data.semcommit_words import lexical_words, split_words
    tok = AutoTokenizer.from_pretrained(QWEN)
    text = "Das ist der YouTube-Kanal, oder?"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    times = [0.1 * (i + 1) for i in range(len(ids))]
    words, _ = lexical_words(split_words(ids, times, tok, text=text))
    row = dict(id="x", text=" ".join(w["text"] for w in words), display_text=text, words=words,
               tokens=[[i, t] for i, t in zip(ids, times)], duration_s=5.0)
    out = normalize_de_row(row, tok)
    assert isinstance(out, dict) and out["text"] == "das ist der youtube kanal oder"
    got = split_words([t[0] for t in out["tokens"]], [t[1] for t in out["tokens"]], tok, text=out["text"])
    assert [(w["text"], w["a"], w["b"]) for w in got] == [(w["text"], w["a"], w["b"]) for w in out["words"]]
    assert out["word_map"][3] == 4                                           # YouTube-Kanal → youtube kanal
    # a re-encoded word keeps its original end time (spread over all its tokens, comma included); a kept word drops only its
    # punctuation-only tokens ("oder?" → "oder"), as normalize_ko_row does
    assert out["words"][4]["end_time"] == pytest.approx(words[3]["end_time"])
    j = max(i for i, t in enumerate(ids) if "oder" in tok.decode([t]))
    assert out["words"][5]["end_time"] == pytest.approx(times[j])
    assert all(b["end_time"] >= a["end_time"] for a, b in zip(out["words"], out["words"][1:]))
