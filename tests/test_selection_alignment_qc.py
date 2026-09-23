import unittest

from vapasr.data.selection_alignment import text_sha
from vapasr.data.selection_alignment_qc import classify_failed, validate_original


def failed(end, duration=1):
    return dict(alignment_ok=False,error="ValueError: invalid_or_out_of_clip_timestamp",duration_s=duration,
                target_text="Hi!",aligner_items=[dict(text="Hi",start_time=.8,end_time=end)],
                schema="qwen-verbatim-align-v1",target_sha256=text_sha("Hi!"),training_eligible=False)


def encoding(): return dict(input_ids=[1,2],offset_mapping=[(0,2),(2,3)])


class AlignmentQCTest(unittest.TestCase):
    def test_policy_boundaries(self):
        r=classify_failed(failed(1.08),encoding());self.assertEqual(r["state"],"ACCEPT_CLAMPED_80MS")
        self.assertEqual(r["corrected"]["aligner_items"][0]["end_time"],1)
        self.assertFalse(r["corrected"]["training_eligible"])
        self.assertEqual(classify_failed(failed(1.081))["state"],"REVIEW_80_320MS")
        self.assertEqual(classify_failed(failed(1.32))["state"],"REVIEW_80_320MS")
        self.assertEqual(classify_failed(failed(1.321))["state"],"REJECT_GT_320MS")

    def test_non_boundary_problem_rejected(self):
        row=failed(1.02);row["aligner_items"][0]["start_time"]=-.1
        self.assertEqual(classify_failed(row)["state"],"REJECT_INVALID")

    def test_original_validation(self):
        row=dict(schema="qwen-verbatim-align-v1",alignment_ok=True,target_text="Hi!",target_sha256=text_sha("Hi!"),
            duration_s=1,aligner_items=[dict(text="Hi",start_time=.2,end_time=.8)],
            words=[dict(text="Hi",char_start=0,char_end=2,start_s=.2,end_s=.8)],
            tokens=[dict(id=1,char_start=0,char_end=2,end_s=.8,timing_kind="word_end_projection",lexical_timing_mask=True),
                    dict(id=2,char_start=2,char_end=3,end_s=.8,timing_kind="nonlexical_anchor",lexical_timing_mask=False)],
            training_eligible=False,timing_scope="decoded_clip_relative")
        self.assertEqual(validate_original(row)["state"],"ACCEPT_ORIGINAL")
        row["words"][0]["end_s"]=.7
        with self.assertRaisesRegex(ValueError,"projection"):
            validate_original(row)


if __name__=="__main__":unittest.main()
