import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.speechlm_asr_align import Engines, groups, rows_for, done, worker
from vapasr.data.selection import file_digest
from vapasr.data.speechlm_inference import adapt


class PipelineTest(unittest.TestCase):
    def test_batch_limits_and_language(self):
        rows = [dict(lang=lang, duration_s=3, key=str(i)) for i, lang in
                enumerate(["English"] * 5 + ["Korean"] * 3)]
        batches = list(groups(rows, dict(batch=3, batch_sec=7)))
        self.assertEqual(sum(map(len, batches)), 8)
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertLessEqual(sum(r["duration_s"] for r in batch), 7)
            self.assertEqual(len({r["lang"] for r in batch}), 1)

    def test_error_split_preserves_order_and_single_failure(self):
        engine = Engines.__new__(Engines)
        engine.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
        engine.once = lambda name, batch: (None, "oom") if len(batch) > 1 or batch == [2] else ([dict(value=batch[0])], None)
        result = engine.infer("qwen", [1, 2, 3])
        self.assertEqual(result, [dict(value=1), dict(error="oom"), dict(value=3)])

    def test_duplicate_input_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            path = out / "input.jsonl"
            path.write_text('{"key":"same"}\n{"key":"same"}\n')
            job = dict(name="part-0", path="input.jsonl", rows=2, sha256=file_digest(path))
            with self.assertRaises(ValueError):
                rows_for(out, job)
            self.assertIsNone(done(out, dict(fingerprint="f"), job))
            dest = out / "results" / "part-0"
            dest.mkdir(parents=True)
            summary = dict(fingerprint="f", complete=True, rows=2, states={"HOLD":1}, outputs={})
            (dest / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                done(out, dict(fingerprint="f"), job)

    def test_worker_accounting_verbatim_alignment_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for name in ("results", "locks", "work"):
                (out / name).mkdir()
            source = dict(source_arrow="input.arrow", source_row_index=0, catalog_id="test",
                language="en", split="train", duration_sec=1., sample_rate=16000, num_channels=1,
                original_transcript="HELLO", audio_tar="a.tar", audio_member="a.wav")
            rows = [adapt(source), adapt(dict(source, source_row_index=1, split="test"))]
            path = out / "input.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            job = dict(name="part-0", rows=2, path=path.name, sha256=file_digest(path))
            config = dict(fingerprint="f", packages={}, qwen_package={}, models={}, memory_fraction=.9,
                tar_index=str(out / "index"), jobs=[job], batch=2, batch_sec=10, io_workers=1)
            torch = SimpleNamespace(set_num_threads=lambda n: None, cuda=SimpleNamespace(
                device_count=lambda:1, set_per_process_memory_fraction=lambda n:None, max_memory_allocated=lambda:0))
            fake = SimpleNamespace(infer=lambda name, batch: [dict(items=[dict(text="Hello", start_time=.1, end_time=.8)])
                if name == "align" else dict(text="Hello!", warning=None) for _ in batch],
                tokenizer=lambda text, **kw:dict(input_ids=list(range(len(text))), offset_mapping=[(i,i+1) for i in range(len(text))]))
            env = dict(SLURM_PROCID="0", SLURM_NTASKS="8", SLURM_JOB_NUM_NODES="1", SLURM_CPUS_PER_TASK="2")
            with patch.dict("sys.modules", {"torch":torch, "qwen_asr":SimpleNamespace(__file__=str(out / "__init__.py"))}), \
                 patch.dict("os.environ", env), patch("experiments.speechlm_asr_align.read_config", return_value=config), \
                 patch("experiments.speechlm_asr_align.decode", return_value=([0.] * 16000, dict(sha256="pcm", review=[]))), \
                 patch("experiments.speechlm_asr_align.textnorm.score_en", side_effect=lambda s:s.lower()), \
                 patch("experiments.speechlm_asr_align.Engines", return_value=fake) as factory:
                worker(SimpleNamespace(out=out))
                worker(SimpleNamespace(out=out))
                self.assertEqual(factory.call_count, 1)
            result = done(out, config, job)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["aligned"], 1)
            self.assertEqual(result["states"], dict(HOLD_METADATA=1, KEEP_ASR_SILVER=1))
            saved = json.loads((out / "results/part-0/aligned.jsonl").read_text())
            self.assertEqual(saved["target_text"], "Hello!")
            self.assertFalse(saved["training_eligible"])


if __name__ == "__main__":
    unittest.main()
