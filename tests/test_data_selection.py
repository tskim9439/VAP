import unittest
import json
from pathlib import Path
import tempfile
from vapasr.data.selection import (agreement, adjudicate, exact_piece, dialogue_rows,
                                  stream_rows, overlap_index, has_overlap)


class SelectionTest(unittest.TestCase):
    def test_no_teacher_no_pass(self):
        self.assertEqual(adjudicate()[0], "PENDING")

    def test_no_gold_from_agreement(self):
        m = agreement("hello", "hello", "hello")
        self.assertEqual(adjudicate(m)[0], "AGREEMENT_CANDIDATE")
        self.assertEqual(adjudicate(m, hard=["missing_audio"])[0], "QUARANTINE")
        self.assertEqual(adjudicate(m, review=["duration_mismatch"])[0], "REVIEW")

    def test_correction_not_automatic(self):
        self.assertEqual(adjudicate(agreement("abc", "xyz", "xyz"))[0], "CORRECTION_REVIEW")
        self.assertEqual(adjudicate(agreement("abc", "", ""))[0], "REVIEW")

    def test_error_rate_can_exceed_one(self):
        self.assertEqual(agreement("a", "bbbb", "bbbb")["qwen_ref"], 4)

    def test_no_neighbouring_crop(self):
        channel = {"pieces": [["/tmp/session_1.wav", 0], ["/tmp/session_3.wav", 4]]}
        u = {"utt_id": "session_000002", "start": 2}
        self.assertIsNone(exact_piece(channel, u, {2})[0])
        self.assertIsNone(exact_piece(channel, u, set())[0])
        channel["pieces"].append(["/tmp/session_2.wav", 2])
        self.assertEqual(exact_piece(channel, u, set())[0]["path"], "/tmp/session_2.wav")

    def test_overlap_not_rejection(self):
        us = [dict(speaker="A", start=0, end=3), dict(speaker="B", start=2, end=4),
              dict(speaker="C", start=4, end=5)]
        ix = overlap_index(us)
        self.assertTrue(has_overlap(us[0], ix))
        self.assertFalse(has_overlap(us[2], ix))

    def test_source_offset_not_synthetic_offset(self):
        d = dict(id="stream", lang="English", corpus="test", split="train",
                 segments=[dict(utt_id="x", text="hi", dur_s=2, offset_s=1,
                                src_offset_s=100, path="x.wav")])
        r = next(stream_rows(d, "test"))
        self.assertEqual(r["audio"]["offset_s"], 100)
        self.assertEqual(r["start_s"], 1)
        self.assertFalse(r["training_eligible"])

    def test_dialogue_missing_preserves_time(self):
        d = dict(conv_id="x", corpus="test", lang="Korean", split="train", duration_s=10,
                 channels={"A": {"pieces": []}}, meta={"missing": [1]},
                 utterances=[dict(utt_id="x_000001", speaker="A", start=2, end=5, text="네")])
        r = next(dialogue_rows(d, "test"))
        self.assertEqual((r["start_s"], r["end_s"]), (2, 5))
        self.assertIsNone(r["audio"])
        self.assertIn("missing_crop_declared", r["reasons"])

    def test_census_holdout_alias_duplicate_and_immutable_source(self):
        from experiments.select_data import P1, P2, census
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "selection"
            (root / "phase2/splits").mkdir(parents=True)
            splits = {n: {"heldout": []} for n in P2}
            splits["aihub71631"]["heldout"] = ["71631:held"]
            (root / "phase2/splits/v1.json").write_text(json.dumps({"corpora": splits}))
            for name in P1:
                p = root / "manifests" / name / "streams.jsonl"
                p.parent.mkdir(parents=True)
                d = dict(id=name, lang="English", corpus="test", split="train",
                         segments=[dict(utt_id="x", text="hi", dur_s=2,
                                        offset_s=1, path="x.wav")])
                p.write_text(json.dumps(d) + "\n" + json.dumps(d) + "\n")
            for name in P2:
                prefix = "71631" if name == "aihub71631" else name
                ds = []
                for stem in ("held", "common"):
                    ds.append(dict(conv_id=f"{prefix}:{stem}", corpus=name, lang="Korean",
                        split="train", duration_s=10, channels={"A": {"path": f"{stem}.wav"}},
                        utterances=[dict(utt_id=stem+"_000001", text="네", speaker="A", start=1, end=3)]))
                (root / f"phase2/{name}.dialogues.jsonl").write_text("\n".join(map(json.dumps, ds)))
            original = (root / "phase2/aihub134-1.dialogues.jsonl").read_bytes()
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                census(root, out, 2)
            summary = json.loads((out / "census.json").read_text())
            counts = summary["sources"]["aihub134-1"]["counts"]
            self.assertEqual(counts["HELDOUT"], 1)
            self.assertEqual(counts["DUPLICATE_LOCATOR_OR_SESSION"], 1)
            self.assertEqual(summary["sources"][P1[0]]["counts"]["DUPLICATE_LOCATOR_OR_SESSION"], 1)
            self.assertEqual(original, (root / "phase2/aihub134-1.dialogues.jsonl").read_bytes())
            pilot = [json.loads(l) for l in (out / "pilot.jsonl").read_text().splitlines()]
            self.assertFalse(any(r["source"] == "aihub134-1" for r in pilot))
            self.assertTrue(all(not r["training_eligible"] for r in pilot))

    def test_strict_audio_channel_crop_and_hash(self):
        import numpy as np
        import soundfile as sf
        from vapasr.data.selection_audio import decode
        with tempfile.TemporaryDirectory() as tmp:
            wav = str(Path(tmp) / "test.wav")
            x = np.ones((16000, 2), dtype=np.float32) * .1
            sf.write(wav, x, 16000)
            row = dict(audio=dict(path=wav+"#ch2", offset_s=0, duration_s=1), duration_s=1)
            with self.assertRaisesRegex(ValueError, "channel_out_of_range"):
                decode(row)
            row["audio"]["path"] = wav+"#ch1"
            a, qa = decode(row)
            b, qb = decode(row)
            self.assertEqual(qa["sha256"], qb["sha256"])
            self.assertEqual(len(a), 16000)
            self.assertEqual(qa["review"], [])
            row["audio"]["duration_s"] = 2
            with self.assertRaisesRegex(ValueError, "truncated_audio_crop"):
                decode(row)
            row["audio"].update(path=wav, duration_s=1)
            self.assertIn("multichannel_without_explicit_channel_mapping", decode(row)[1]["review"])

    def test_zero_audio_is_not_agreement(self):
        import numpy as np
        import soundfile as sf
        from vapasr.data.selection_audio import decode
        with tempfile.TemporaryDirectory() as tmp:
            wav = str(Path(tmp) / "zero.wav")
            sf.write(wav, np.zeros(16000), 16000)
            with self.assertRaisesRegex(ValueError, "all_zero_audio"):
                decode(dict(audio=dict(path=wav), duration_s=1))


if __name__ == "__main__":
    unittest.main()
