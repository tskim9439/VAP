import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import simulst_common as C  # noqa: E402


def test_words_with_times_last_char_owner():
    # Seamless SPM pieces / IWSLT increments: a word's time is the time of the fragment holding its last character
    text, times = C.words_with_times([(" Es", 1.0), (" ge", 2.0), ("ht", 3.0), (" um.", 4.0), ("", 5.0)])
    assert text == "Es geht um." and times == [1.0, 3.0, 4.0]
    text, times = C.words_with_times([("  ", 1.0)])
    assert text == "" and times == []


def test_pred_row_checks_word_count():
    r = dict(id="x/1", dataset="fleurs-de-en", lang="German", tgt_lang="English", audio_s=2.5, translation_reference="a b")
    row = C.pred_row("sys", "s1", r, "a b", [1.0, 2.0], 1.0, extra=1)
    assert row["translation_word_times"] == [1.0, 2.0] and row["raw"] == {"extra": 1} and row["tail_s"] == 1.0
    with pytest.raises(AssertionError):
        C.pred_row("sys", "s1", r, "a b c", [1.0, 2.0])


def test_load_rows_and_shard(tmp_path):
    m = tmp_path / "m.jsonl"
    rows = [dict(id=f"d/{i}", dataset="fleurs-de-en" if i % 2 else "librispeech", tgt_lang="English" if i % 2 else None) for i in range(6)]
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = C.load_rows([str(m)])
    assert [r["id"] for r in got] == ["d/1", "d/3", "d/5"]
    assert [r["id"] for r in C.shard(got, 1, 2)] == ["d/3"]


def test_cv_en_eval_texts_and_shard_name(tmp_path):
    import cv_en_align as V
    m = tmp_path / "eval.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in (
        dict(lang="English", raw_reference="He said hi."), dict(lang="German", tgt_lang="English", translation_reference="Good day."),
        dict(lang="Korean", tgt_lang="German", translation_reference="Guten Tag."))))
    assert V.eval_texts([str(m)]) == ["He said hi.", "Good day."]
    assert V.shard_name("train-00003-of-00008.parquet", 7) == "train-00003-of-00008-b007"
    assert V._ngrams("a b c".split(), 2) == {"a b", "b c"}


def test_encoder_chunk_times_waits_for_whole_encoder_chunk():
    # chunked_limited [56,3]: frames 0-3 arrive together at 0.32 s, 4-7 at 0.64 s; the last partial chunk at the input end (K frames);
    # flush rounds (k >= K) keep their time; chunk 1 leaves times unchanged
    K = 10
    times = [round((k + 1) * 0.08, 3) for k in (0, 3, 4, 7, 8, 9, 10, 12)]
    assert C.encoder_chunk_times(times, K, 4) == [0.32, 0.32, 0.64, 0.64, 0.8, 0.8, 0.88, 1.04]
    assert C.encoder_chunk_times(times, K, 1) == times


def test_stream_rtf_queue_times_backlog_and_per_chunk_encoder_cost():
    # 10 frames → chunks [0-3] arrive 0.32 s, [4-7] 0.64 s, [8-9] 0.80 s (input end); 100 ms per frame makes a 400 ms chunk that backs up
    import stream_rtf as R
    ticks = [100.0] * 10
    t, chunk_ms, backlog = R.queue_times(10, ticks, [0.0, 20.0, 0.0], 1, 50.0)
    assert chunk_ms == [400.0, 420.0, 200.0]
    assert t[3] == pytest.approx(0.72) and t[4] == pytest.approx(0.72 + 0.12)      # chunk 1 waits for chunk 0 to finish
    assert t[9] == pytest.approx(1.14 + 0.2) and t[10] == pytest.approx(1.34 + 0.05)
    assert backlog == pytest.approx(1.34 - 0.8)
    assert R.queue_times(10, ticks, 0.0, 0, 0.0)[1] == [400.0, 400.0, 200.0]          # scalar encoder cost per chunk
