import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.selection_focus_review import spread, wav_digest


class FocusReviewTest(unittest.TestCase):
    def test_spread_stable_and_unique(self):
        rows = [dict(key=str(i), audio_qc=dict(audio=dict(duration_s=i))) for i in range(20)]
        selected = spread(rows, 8)
        self.assertEqual(selected, spread(list(reversed(rows)), 8))
        self.assertEqual(len({r['key'] for r in selected}), 8)
        self.assertEqual(selected[0], rows[0])
        self.assertEqual(selected[-1], rows[-1])

    def test_float_wave_data_hash_not_header(self):
        payload = struct.pack('<ff', 0.25, -0.5)
        fmt = struct.pack('<HHIIHH', 3, 1, 16000, 64000, 4, 32)
        body = b'WAVEfmt ' + struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', 8) + payload
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'test.wav'
            p.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)
            self.assertEqual(wav_digest(p), hashlib.sha256(payload).hexdigest())
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                wav_digest(p)


if __name__ == '__main__':
    unittest.main()
