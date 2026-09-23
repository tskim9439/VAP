"""Protocol invariants: padding, corpus micro scores and independent decoding."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vapasr.data.single_turn_eval import aggregate, score_pair
from vapasr.hf.batch_decode import DecodeState


class EvaluationTests(unittest.TestCase):
    def test_edit_counts_and_micro_not_macro(self):
        a = score_pair("a b c d", "a x c d", "English")
        b = score_pair("a", "", "English")
        rows = [dict(dataset="en", delta=2, audio_s=1, forced=0, metrics=m) for m in (a, b)]
        stats = aggregate(rows)["en/delta-2"]["metrics"]["wer"]
        self.assertEqual((stats["substitutions"], stats["deletions"], stats["n_ref"]), (1, 1, 5))
        self.assertEqual(stats["rate"], .4)

    def test_korean_spaces_separate(self):
        s = score_pair("오늘 날씨", "오늘날씨", "Korean")
        self.assertEqual(s["cer_nospace"]["errors"], 0)
        self.assertEqual(s["cer_space"]["deletions"], 1)

    def test_padding_and_odd_pcm(self):
        path = Path(__file__).parents[1] / "experiments/eval_single_turn_asr.py"
        spec = importlib.util.spec_from_file_location("eval_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as td:
            pcm = Path(td) / "sample.pcm"
            pcm.write_bytes(np.array([100, -200, 300], dtype="<i2").tobytes() + b"x")
            x = cli.read_audio(dict(path=str(pcm), audio_format="pcm_s16le", num_samples=3), 1.0)
            self.assertEqual(len(x), 16003)
            np.testing.assert_array_equal(x[:3], np.array([100, -200, 300], dtype=np.float32) / 32768)
            self.assertTrue(np.all(x[3:] == 0))

    def test_state_chunk_advance_and_empty_flush(self):
        s = DecodeState(K=2, cap=2, max_flush=8, max_total=20)
        self.assertEqual(s.consume(11, 99), ("text", 11))
        self.assertEqual(s.consume(99, 99), ("next", 0))
        self.assertEqual(s.consume(11, 99), ("audio", 1))  # ignore NEXT logits
        self.assertEqual(s.consume(12, 99), ("text", 12))
        s.consume(99, 99)
        self.assertEqual(s.consume(0, 99), ("empty", 0))
        s.consume(13, 99)
        s.consume(99, 99)
        s.consume(0, 99)
        s.consume(99, 99)
        self.assertTrue(s.done)
        self.assertEqual(s.emitted, [(0, 11), (1, 12), (2, 13)])
        self.assertEqual(s.flush_rounds, 2)

    def test_cap_and_zero_flush(self):
        s = DecodeState(K=1, cap=1, max_flush=0, max_total=10)
        s.consume(11, 99)
        self.assertEqual(s.consume(12, 99), ("next", 0))
        self.assertTrue(s.done)
        self.assertEqual(s.forced, 1)
        self.assertEqual(s.emitted, [(0, 11)])


if __name__ == "__main__":
    unittest.main()
