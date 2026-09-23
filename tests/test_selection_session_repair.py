import unittest
from experiments.selection_session_repair import rate_check


class ScopeTest(unittest.TestCase):
    def test_explicit_scope(self):
        self.assertIsNone(rate_check("SDRW2200000598.1.1.1",96000,1,b"\0\0"))
        self.assertEqual(rate_check("SDRW2200000999.1",96000,1,b"\0\0"),"outside_reviewed_scope")

    def test_no_ratio_autodetection(self):
        self.assertEqual(rate_check("SDRW2200000598.1",32000,1,b"\0\0"),"duration_inconsistent_with_fixed_48khz")

    def test_byte_and_header_guards(self):
        for n in (0,96001):
            self.assertEqual(rate_check("SDRW2200000598.1",n,1,b"\0\0"),"invalid_pcm_bytes")
        self.assertEqual(rate_check("SDRW2200000598.1",96000,1,b"RIFF"),"headered_audio_not_raw_pcm")

    def test_annotation_guards(self):
        for dur in (float("nan"),float("inf"),-1,0):
            self.assertEqual(rate_check("SDRW2200000598.1",96000,dur,b"\0\0"),"invalid_annotation_duration")


if __name__=="__main__":unittest.main()
