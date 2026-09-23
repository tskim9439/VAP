import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from vapasr.data.speechlm_inference import TarReader, adapt, decide


class SpeechLMTest(unittest.TestCase):
    def row(self):
        return dict(source_arrow="source.arrow", source_row_index=1,
                    catalog_id="SLM-SPEECH-000035", language="en", split="train",
                    duration_sec=3., sample_rate=8000, num_channels=1,
                    original_transcript="Don't normalize $5!", audio_tar="audio.tar", audio_member="a.wav")

    def test_verbatim_and_key(self):
        raw = self.row()
        before = copy.deepcopy(raw)
        row = adapt(raw)
        self.assertEqual(row["text"], "Don't normalize $5!")
        self.assertEqual(raw, before)
        self.assertEqual(row["lang"], "English")
        self.assertNotEqual(row["key"], adapt(dict(raw, source_row_index=2))["key"])
        self.assertFalse(row["training_eligible"])

    def test_scope_holds(self):
        for change in (dict(split="test"), dict(num_channels=2), dict(duration_sec=31), dict(language="fr")):
            self.assertEqual(adapt(dict(self.row(), **change))["selection_state"], "HOLD_METADATA")

    def test_pair_gate_and_safety(self):
        row = adapt(self.row())
        row["audio_info"] = dict(review=[])
        teacher = dict(text="Some text", warning=None)
        metrics = dict(qwen_units=20, whisper_units=20, qwen_whisper=.05, qwen_ref=1., whisper_ref=1.)
        self.assertEqual(decide(row, teacher, teacher, metrics), "KEEP_ASR_SILVER")
        self.assertEqual(decide(row, teacher, teacher, dict(metrics, qwen_whisper=.051)), "HOLD_DISAGREEMENT")
        self.assertEqual(decide(row, teacher, teacher, dict(metrics, qwen_units=0)), "HOLD_TEXT")
        self.assertEqual(decide(row, dict(teacher, warning="cap"), teacher, metrics), "HOLD_GENERATION")
        row["audio_info"]["review"] = ["clipping"]
        self.assertEqual(decide(row, teacher, teacher, metrics), "HOLD_AUDIO")

    def test_tar_offset_cache_no_source_write_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            archive = source / "sample.tar"
            def write(payload):
                with tarfile.open(archive, "w") as tf:
                    info = tarfile.TarInfo("dir/a.wav")
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))
            write(b"abc")
            reader = TarReader(root / "index")
            self.assertEqual(reader.read(str(archive), "dir/a.wav"), b"abc")
            self.assertEqual(list(source.iterdir()), [archive])
            self.assertEqual(len(list((root / "index").glob("*.json"))), 1)
            write(b"changed audio")
            self.assertEqual(reader.read(str(archive), "dir/a.wav"), b"changed audio")
            self.assertEqual(len(list((root / "index").glob("*.json"))), 2)
            with self.assertRaises(KeyError):
                reader.read(str(archive), "missing")


if __name__ == "__main__":
    unittest.main()
