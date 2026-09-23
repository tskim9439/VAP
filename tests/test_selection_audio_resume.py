import gzip
import json
from pathlib import Path
import tempfile
import unittest
from experiments.selection_audio_census_resume import readable_prefix, recover_prefix


class ResumeTest(unittest.TestCase):
    def make_rows(self):
        return [dict(key=str(i), source="db", status="AUDIO_ERROR", training_eligible=False)
                for i in range(20)]

    def test_complete_and_truncated_gzip(self):
        rows = self.make_rows()
        payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "partial.gz"
            compressed = gzip.compress(payload)
            p.write_bytes(compressed)
            self.assertEqual(list(readable_prefix(p)), rows)
            p.write_bytes(compressed[:-8])
            recovered = list(readable_prefix(p))
            self.assertGreater(len(recovered), 0)
            self.assertEqual(recovered, rows[:len(recovered)])

    def test_resume_consumes_exact_prefix(self):
        rows = self.make_rows()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "partial.gz"
            p.write_bytes(gzip.compress("".join(json.dumps(r)+"\n" for r in rows[:10]).encode()))
            it = iter(rows)
            copied = []
            self.assertEqual(recover_prefix(p, it, "db", copied.append), 10)
            self.assertEqual(copied, rows[:10])
            self.assertEqual(list(it), rows[10:])
            with self.assertRaisesRegex(ValueError, "candidate sequence"):
                recover_prefix(p, iter(rows[1:]), "db", lambda r: None)

    def test_corruption_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.gz"
            p.write_bytes(gzip.compress(b"not json\n"))
            with self.assertRaises(json.JSONDecodeError):
                list(readable_prefix(p))


if __name__ == "__main__":
    unittest.main()
