import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.selection_align_qwen import align_batch, batches, completed, load_audio, prepare, summarize
from vapasr.data.selection import file_digest
from vapasr.data.selection_alignment import project_items, target_text, text_sha


def item(text, start, end):
    return dict(text=text, start_time=start, end_time=end)


def chars(text):
    return dict(input_ids=list(range(len(text))), offset_mapping=[(i, i + 1) for i in range(len(text))])


class AlignmentTest(unittest.TestCase):
    def row(self):
        text = "Don't change My Style!"
        return dict(key="k", source="db", lang="English", audio={"path": "audio.wav"},
                    text="old normalized label", transcripts={"qwen_raw": text},
                    selection_state="KEEP_ASR_SILVER", auto_selection={"expected_waveform_sha256": "expected"},
                    recommended_training_target=dict(text=text, normalization="none"))

    def test_target_is_verbatim_and_not_old_label(self):
        row = self.row()
        before = copy.deepcopy(row)
        self.assertEqual(target_text(row), row["transcripts"]["qwen_raw"])
        self.assertEqual(before, row)
        self.assertNotEqual(text_sha(target_text(row)), text_sha(target_text(row).lower()))
        for edit in (dict(selection_state="HOLD_DISAGREEMENT"),
                     dict(recommended_training_target=dict(text="normalized", normalization="none")),
                     dict(recommended_training_target=dict(text=target_text(row), normalization="tn"))):
            with self.assertRaises(ValueError):
                target_text(dict(row, **edit))

    def test_case_punctuation_and_anchors(self):
        text = '“Hi, Don\'t!”'
        result = project_items(text, [item("Hi", .2, .4), item("Don't", .5, .9)], chars(text), 1)
        self.assertEqual(result["words"][0]["char_start"], 1)
        self.assertEqual(result["tokens"][0]["timing_kind"], "nonlexical_anchor")
        self.assertEqual(result["tokens"][0]["end_s"], .2)
        self.assertEqual(result["tokens"][-1]["end_s"], .9)
        self.assertFalse(result["tokens"][-1]["lexical_timing_mask"])
        self.assertFalse(result["training_eligible"])
        self.assertFalse(result["turn_supervision"])

    def test_korean_numbers_and_duplicate_bpe_offsets(self):
        text = "100명입니다."
        encoding = chars(text)
        encoding["input_ids"].insert(4, 999)
        encoding["offset_mapping"].insert(4, (3, 4))
        result = project_items(text, [item("100명", .1, .4), item("입니다", .4, .8)], encoding, 1)
        self.assertEqual(result["tokens"][3]["end_s"], .4)
        self.assertEqual(result["tokens"][4]["end_s"], .4)
        self.assertEqual(result["words"][1]["char_start"], 4)

    def test_surface_mismatch_has_no_fallback(self):
        for text, aligned in [("100", "one hundred"), ("Hello", "hello"), ("one two", "one")]:
            with self.assertRaisesRegex(ValueError, "surface"):
                project_items(text, [item(aligned, .1, .3)], chars(text), 1)

    def test_invalid_timestamps(self):
        for start, end in [(-.1, .2), (.3, .2), (0, 1.01), (0, float("nan")), (0, float("inf"))]:
            with self.assertRaises(ValueError):
                project_items("Hi", [item("Hi", start, end)], chars("Hi"), 1)
        with self.assertRaises(ValueError):
            project_items("a b", [item("a", .2, .7), item("b", .1, .6)], chars("a b"), 1)

    def test_missing_or_invalid_offsets(self):
        for encoding in [dict(input_ids=[0], offset_mapping=[(0, 0)]),
                         dict(input_ids=[0], offset_mapping=[(0, 1)]),
                         dict(input_ids=[], offset_mapping=[])]:
            with self.assertRaises(ValueError):
                project_items("Hi", [item("Hi", .1, .3)], encoding, 1)

    def test_decoder_hash_guard(self):
        with patch("vapasr.data.selection_audio.decode", return_value=([0] * 160, dict(sha256="changed"))):
            base, waveform, error = load_audio(self.row())
        self.assertIn("sha256_mismatch", error)
        self.assertIsNone(waveform)
        self.assertEqual(base["target_text"], self.row()["transcripts"]["qwen_raw"])

    def test_batch_duration_and_count(self):
        rows = [dict(lang="English", duration_s=x) for x in [5, 1, 4, 2, 3]]
        result = list(batches(rows, dict(batch=2, batch_sec=6)))
        self.assertEqual(sum(map(len, result)), 5)
        for batch in result:
            self.assertLessEqual(len(batch), 2)
            self.assertLessEqual(sum(r["duration_s"] for r in batch), 6)

    def test_batch_failure_isolated(self):
        class Aligner:
            def align(self, audio, text, language):
                if len(text) > 1:
                    raise RuntimeError("simulated OOM")
                return [text[0]]
        batch = [(dict(target_text=t, lang="English"), [0]) for t in ["a", "b", "c"]]
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        self.assertEqual(align_batch(Aligner(), batch, torch), ["a", "b", "c"])

    def test_resume_integrity_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            dest = out / "results" / "part-00000"
            dest.mkdir(parents=True)
            spec = dict(name=dest.name, input_sha256="input", rows=2)
            (dest / "aligned.jsonl").write_text('{"key":"ok"}\n')
            (dest / "failed.jsonl").write_text('{"key":"fail"}\n')
            result = dict(complete=True, name=dest.name, fingerprint="f", input_sha256="input", rows=2,
                          ok=1, failed=1, outputs={p.name: file_digest(p) for p in dest.glob("*.jsonl")})
            (dest / "summary.json").write_text(json.dumps(result))
            self.assertEqual(completed(out, spec, "f"), result)
            with self.assertRaises(ValueError):
                completed(out, spec, "different")
            (out / "config.json").write_text(json.dumps(dict(fingerprint="f", scope="smoke", shards=[spec])))
            summarize(SimpleNamespace(out=out))
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["status"], "complete_with_failures")
            self.assertEqual(summary["scope"], "smoke")
            self.assertFalse(summary["training_eligible"])
            (dest / "aligned.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Changed output"):
                completed(out, spec, "f")

    def test_prepare_resume_config_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "selection"
            shard = source / "results" / "part-00000"
            shard.mkdir(parents=True)
            detail = dict(complete=True, sources={"db": {"KEEP_ASR_SILVER": 2}},
                          outputs={"keep-asr.jsonl": "input"})
            (shard / "summary.json").write_text(json.dumps(detail))
            summary = dict(complete=True, policy={"version": "asr-pair-select-v2"},
                           sources=detail["sources"], shards=[dict(name=shard.name, outputs=detail["outputs"])])
            (source / "summary.json").write_text(json.dumps(summary))
            a = SimpleNamespace(source=source, out=root / "output", aligner=root / "model",
                                tokenizer=root / "tokenizer", mem_frac=.9, batch=64, batch_sec=480,
                                io_threads=3, max_shards=0, max_rows=0)
            with patch("experiments.selection_align_qwen.tree_hash", return_value={"fixture": "sha"}), \
                    patch("experiments.selection_align_qwen.importlib.util.find_spec", return_value=SimpleNamespace(origin=str(root / "__init__.py"))), \
                    patch("experiments.selection_align_qwen.importlib.metadata.version", return_value="fixture"):
                prepare(a)
                prepare(a)
                config = json.loads((a.out / "config.json").read_text())
                self.assertEqual(config["shards"][0]["rows"], 2)
                self.assertEqual(config["scope"], "full")
                a.batch = 128
                with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                    prepare(a)
                a.out = source / "invalid"
                with self.assertRaisesRegex(ValueError, "independent"):
                    prepare(a)


if __name__ == "__main__":
    unittest.main()
