"""외부 baseline 채점: 청크·speaker 순열·한국어 문자·누락·무음 fixture."""
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
from vapasr.data.baseline_eval import (parse_speakers, safe_prefix_end, score_text,
                                      score_session, read_manifest, tn_fingerprint)


class ParserTests(unittest.TestCase):
    def test_continuation_word_and_speaker(self):
        self.assertEqual(parse_speakers(["\n Speaker 0:hel", "lo world", "", "\n Speaker 1:yes"]),
                         [{"speaker":"speaker_0", "text":"hello world"},
                          {"speaker":"speaker_1", "text":"yes"}])

    def test_split_label_and_unassigned(self):
        self.assertEqual(parse_speakers(["oops\n Spe", "aker 2:hi"]),
                         [{"speaker":"__unassigned__", "text":"oops"},
                          {"speaker":"speaker_2", "text":"hi"}])

    def test_silence_and_inline_label(self):
        self.assertEqual(parse_speakers(["", "  "]), [])
        self.assertEqual(parse_speakers(["\n Speaker 0:the string Speaker 1: is quoted"])[0]["text"],
                         "the string Speaker 1: is quoted")

    def test_korean_characters_not_words(self):
        self.assertEqual(score_text("안녕 하세요!", "Korean"), "안 녕 하 세 요")
        self.assertEqual(score_text("가", "ko"), "가")

    def test_english_normalization(self):
        self.assertEqual(score_text("I've got 5 apples.", "English"), "i've got five apples")

    def test_prefix_no_clipped_reference(self):
        us = [{"start":1, "end":12}, {"start":10, "end":18}]
        self.assertEqual(safe_prefix_end(us, 30, 15), 1)
        self.assertEqual(safe_prefix_end(us, 30, 20), 20)

    def test_manifest_guardrails(self):
        row = dict(session_id="s", purpose="dev_smoke", duration_s=60, tn=tn_fingerprint())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text(json.dumps(row)+"\n")
            self.assertEqual(len(read_manifest(path)), 1)
            for invalid in ([row, row], [dict(row, duration_s=481)],
                            [dict(row, purpose="locked_test")], [dict(row, tn={})], []):
                path.write_text("".join(json.dumps(r)+"\n" for r in invalid))
                with self.assertRaises(ValueError): read_manifest(path)


@unittest.skipUnless(importlib.util.find_spec("meeteval"), "meeteval required for scorer integration")
class ScorerTests(unittest.TestCase):
    def test_global_swap(self):
        r = [{"speaker":"a", "text":"hello"}, {"speaker":"b", "text":"yes"}]
        h = [{"speaker":"b", "text":"hello"}, {"speaker":"a", "text":"yes"}]
        m = score_session("s", r, h, "en")
        self.assertEqual(m["cp"]["errors"], 0)
        self.assertEqual(m["orc"]["errors"], 0)

    def test_korean_cer(self):
        m = score_session("s", [{"speaker":"a", "text":"가나 다"}],
                          [{"speaker":"z", "text":"가나 라"}], "ko")
        self.assertEqual(m["cp"]["length"], 3)
        self.assertEqual(m["cp"]["substitutions"], 1)

    def test_mid_session_reassignment(self):
        r = [{"speaker":"a", "text":"one"}, {"speaker":"b", "text":"two"},
             {"speaker":"a", "text":"three"}]
        h = [{"speaker":"x", "text":"one"}, {"speaker":"y", "text":"two"},
             {"speaker":"y", "text":"three"}]
        m = score_session("s", r, h, "en")
        self.assertEqual(m["orc"]["errors"], 0)
        self.assertGreater(m["cp"]["errors"], 0)

    def test_missing_and_silence_insertions(self):
        self.assertEqual(score_session("s", [{"speaker":"a", "text":"one two"}], [], "en")["cp"]["deletions"], 2)
        m = score_session("s", [], [{"speaker":"x", "text":"one two"}], "en")
        self.assertEqual(m["orc"]["insertions"], 2)
        self.assertIsNone(m["orc"]["rate"])


if __name__ == "__main__":
    unittest.main()
