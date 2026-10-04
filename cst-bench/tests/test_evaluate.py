"""Scoring tests on a hand-made two-turn session (no corpus data required)."""
import json

import pytest

from cstbench.cli import main
from cstbench.evaluate import align, evaluate, laal, normalize, resegment

pytest.importorskip("sacrebleu")

SESSION = {"session_id": "taxi/SES0001", "turns": [
    {"turn_id": "a", "lang": "German", "start_s": 0.5, "end_s": 2.5, "transcript": "wohin soll der Wagen kommen",
     "translation": "where should the car come", "translation_lang": "English"},
    {"turn_id": "b", "lang": "English", "start_s": 2.8, "end_s": 4.8, "transcript": "to the north exit",
     "translation": "zum Nordausgang", "translation_lang": "German"},
    {"turn_id": "c", "lang": "German", "start_s": 5.0, "end_s": 6.0, "transcript": "gut fünfzehn Minuten",
     "translation": "ok fifteen minutes", "translation_lang": "English"},
]}


def test_normalize():
    assert normalize("OK, it'll take 15 minutes!", "English") == "ok it'll take fifteen minutes"
    assert normalize("Etwa 15 Minuten.", "German") == "etwa fünfzehn minuten"
    assert normalize("'quoted' Nord-Ausgang", "German") == "quoted nord ausgang"


def test_align_and_resegment():
    assert align("a x b".split(), "a b".split()) == [0, -1, 1]
    turns = ["where should the car come".split(), "ok fifteen minutes".split()]
    hyp = "where should car come now okay fifteen minutes".split()
    assert resegment(hyp, turns) == [[0, 1, 2, 3, 4], [5, 6, 7]]
    assert resegment([], turns) == [[], []]


def test_laal():
    # 2 s source, 4 words emitted at 1, 2, 2, 2.5 s after the turn start; ideal rate 0.5 s per word
    assert laal([1.0, 2.0, 2.0, 2.5], 2.0, 4) == pytest.approx(((1.0 - 0) + (2.0 - 0.5)) / 2)
    assert laal([], 2.0, 3) != laal([], 2.0, 3)                               # nan


def test_evaluate_resegmented_and_oracle():
    perfect = [{"session": "taxi/SES0001", "lang": "English", "t": 2.5, "text": "Where should the car come?"},
               {"session": "taxi/SES0001", "lang": "German", "t": 5.3, "text": "Zum Nordausgang."},
               {"session": "taxi/SES0001", "lang": "English", "t": 6.0, "text": "OK, 15 minutes."}]
    rep, seg = evaluate([SESSION], perfect)
    en = rep["directions"]["German->English"]
    assert en["bleu"] == 100.0 and en["turns"] == 2 and en["empty_turns"] == 0
    assert en["end_offset_s"]["mean"] == 0.0
    de = rep["directions"]["English->German"]
    assert de["end_offset_s"]["mean"] == pytest.approx(0.5)
    assert [x["hyp"] for x in seg if x["lang"] == "English"] == ["where should the car come", "ok fifteen minutes"]

    oracle = [dict(h, turn_id=tid) for h, tid in zip(perfect, ["a", "b", "c"])]
    oracle.append({"session": "taxi/SES0001", "lang": "English", "t": 4.0, "text": "stray", "turn_id": "b"})
    rep2, _ = evaluate([SESSION], oracle)
    assert rep2["wrong_language_words"] == 1 and rep2["directions"]["German->English"]["bleu"] == 100.0

    rep3, _ = evaluate([SESSION], perfect[:1])
    assert rep3["directions"]["German->English"]["empty_turns"] == 1
    assert rep3["directions"]["English->German"]["empty_turns"] == 1


def test_cli_eval(tmp_path):
    (tmp_path / "s.jsonl").write_text(json.dumps(SESSION) + "\n")
    (tmp_path / "h.jsonl").write_text(json.dumps({"session": "taxi/SES0001", "lang": "German", "t": 5.0,
                                                  "text": "zum Nordausgang"}) + "\n")
    main(["eval", "--sessions", str(tmp_path / "s.jsonl"), "--hyp", str(tmp_path / "h.jsonl"),
          "--out", str(tmp_path / "r.json"), "--segments-out", str(tmp_path / "seg.jsonl")])
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["directions"]["English->German"]["chrf"] == 100.0             # 2 words: no 4-grams for BLEU
    assert len((tmp_path / "seg.jsonl").read_text().splitlines()) == 3
