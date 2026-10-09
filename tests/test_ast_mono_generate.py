import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import ast_mono_generate as G  # noqa: E402


def test_clean_text_annotations():
    assert G.clean_text("o/ b/ 좀비 물리면 죽는 거 아니야? b/") == "좀비 물리면 죽는 거 아니야?"
    assert G.clean_text("(70%)/(칠십 퍼센트) 정도요") == "70% 정도요"
    assert G.clean_text("<LAUGH> yeah we are {laughing} fine [noise]") == "yeah we are fine"
    assert G.clean_text("== Plot summary == Electric Forest is") == "Plot summary Electric Forest is"
    assert G.clean_text("AND COMPARISON OF HYPOTHESES") == "and comparison of hypotheses"


def test_glue_symbol_tokens():
    assert G.glue(["Hello", "-", "world", "!"]) == ["Hello -", "world !"]
    assert G.glue(["–", "Ja", "genau"]) == ["– Ja", "genau"]


def test_align_one_to_one_and_numbers():
    words = [("on", 0.5), ("eleven", 0.9), ("october", 1.4), ("one", 1.6), ("thousand", 1.9), ("nine", 2.1),
             ("hundred", 2.4), ("one", 2.6), ("he", 3.0), ("died", 3.4)]
    toks = ["on", "11", "October", "1901", "he", "died."]
    times, a_share, b_share = G.align_times(toks, words)
    assert times[0] == 0.5 and times[2] == 1.4
    assert times[3] == 2.6                       # the year takes the end of "... hundred one"
    assert times[-1] == 3.4 and times == sorted(times)
    assert a_share == 1.0 and 0.3 < b_share < 0.6   # all letters found; the spelled-out numbers lower b_share
    same = G.align_times(["Ja,", "genau."], [("ja", 0.3), ("genau", 0.8)])
    assert same == ([0.3, 0.8], 1.0, 1.0)


def test_sentence_ends():
    assert G.is_sentence_end("data.", "This")
    assert not G.is_sentence_end("Mr.", "Smith")
    assert not G.is_sentence_end("11.", "Oktober")
    assert not G.is_sentence_end("etc.", "and")
    assert not G.is_sentence_end("so...", "and")          # lowercase continuation
    assert G.is_sentence_end("거지.", "그래서")             # caseless script
    assert G.is_sentence_end("end?", None)


def test_split_units_sentences_and_long():
    toks = ["I", "agree.", "But", "we", "should", "wait."]
    assert G.split_units(toks, [float(i) for i in range(6)], 40) == [(0, 2), (2, 6)]
    long_toks = [f"w{i}" for i in range(50)]
    long_toks[24] = "w24,"
    units = G.split_units(long_toks, [float(i) for i in range(50)], 40)
    assert units == [(0, 25), (25, 50)]
    pause = [f"w{i}" for i in range(50)]
    times = [float(i) for i in range(50)]
    times[30:] = [t + 5.0 for t in times[30:]]                # 5 s pause before word 30
    assert G.split_units(pause, times, 40) == [(0, 30), (30, 50)]


def test_row_units_prefers_punctuated_text():
    r = dict(id="x", lang="English", duration_s=4.0,
             segments=[dict(raw_text="THANKS A LOT")], pnc_text="Thanks a lot. See you.",
             words=[dict(text=w, end_time=t, seg=0) for w, t in
                    [("thanks", 0.4), ("a", 0.5), ("lot", 0.9), ("see", 1.5), ("you", 1.8)]])
    units, srcs = G.row_units(r, 40)
    assert srcs == {"pnc_text": 1}
    assert [u["tokens"] for u in units] == [["Thanks", "a", "lot."], ["See", "you."]]
    assert units[1]["times"] == [1.5, 1.8]
    multi = dict(id="y", lang="English", duration_s=9.0, segments=[dict(raw_text="AND SO ON"), dict(raw_text="<LAUGH> RIGHT")],
                 words=[dict(text="and", end_time=0.3, seg=0), dict(text="so", end_time=0.5, seg=0),
                        dict(text="on", end_time=0.8, seg=0), dict(text="right", end_time=5.2, seg=1)])
    units, srcs = G.row_units(multi, 40)
    assert srcs == {"raw_text": 2}
    assert [(u["seg"], u["tokens"]) for u in units] == [(0, ["and", "so", "on"]), (1, ["right"])]


def test_v3_prompt_is_cacheable():
    P = G.P
    assert set(P.EXAMPLES_V3) == {("English", "German"), ("English", "Korean"), ("German", "English"), ("Korean", "English")}
    for src, tgt in P.EXAMPLES_V3:
        a = P.p_translate_v3(src, tgt, "one two", True)
        b = P.p_translate_v3(src, tgt, "three", False)
        assert a[0] == b[0] == P.p_system_v3(src, tgt)          # one fixed system message per direction
        assert a[1].endswith("one two") and "has not finished yet" in b[1]
        msgs = P.chat(a)
        assert [m["role"] for m in msgs] == ["system", "user"]
    assert P.chat("x") == [{"role": "user", "content": "x"}]
