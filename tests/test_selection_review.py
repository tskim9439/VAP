import unittest
from experiments.selection_review_audit import joined, keep_sample, bucket


class ReviewTest(unittest.TestCase):
    def row(self, i, state="PENDING_AUDIO_TEACHERS"):
        return dict(key=str(i), source="test", selection_state=state, stratum="short")

    def qc(self, i):
        return dict(key=str(i), source="test", training_eligible=False, status="AUDIO_ERROR")

    def test_join_filters_but_does_not_skip_missing(self):
        r, q = self.row(1), self.qc(1)
        self.assertEqual(list(joined([self.row(0, "HELDOUT"), r], [q])), [(r, q)])
        for rows, results in [([r], []), ([], [q]), ([r], [self.qc(2)])]:
            with self.assertRaises(ValueError):
                list(joined(rows, results))

    def test_deterministic_bounded_sample(self):
        heaps = []
        for order in (range(100), reversed(range(100))):
            h = []
            for i in order:
                keep_sample(h, self.row(i), self.qc(i), 8)
            heaps.append(sorted(x[1] for x in h))
        self.assertEqual(len(heaps[0]), 8)
        self.assertEqual(heaps[0], heaps[1])

    def test_approved_qc_rejected(self):
        q = dict(self.qc(1), training_eligible=True)
        with self.assertRaises(ValueError):
            list(joined([self.row(1)], [q]))

    def test_error_stratum(self):
        self.assertEqual(bucket(self.row(1), self.qc(1)), ("all", "AUDIO_ERROR", "none", "short"))


if __name__ == "__main__":
    unittest.main()
