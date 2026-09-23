import unittest
from experiments.selection_vad_audit import pieces


class VADTest(unittest.TestCase):
    def test_gap_is_not_audio(self):
        p,n=pieces('[[1,2],[3,4]]')
        self.assertEqual(n,32000)
        self.assertEqual(p[1]['clip_start_sample'],16000)
        self.assertEqual(p[1]['source_start_s'],3)

    def test_invalid_intervals(self):
        for s in ('[]','[[1,1]]','[[2,1]]','[[-1,1]]','[[1,3],[2,4]]','__import__("os")'):
            with self.assertRaises((ValueError,SyntaxError)):pieces(s)


if __name__=='__main__':unittest.main()
