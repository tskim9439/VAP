import copy
import json
from pathlib import Path
import tempfile
import unittest
from vapasr.data.selection_review import binding, load_reviews, review_flags
from vapasr.data.selection import adjudicate


class HumanReviewTest(unittest.TestCase):
    def setUp(self):
        self.row = dict(key='k', source='s', utt_id='u', manifest_sha256='m',
                        audio={'path':'a'}, text='right', raw_text='Right',
                        audio_qc={'audio':{'sha256':'wave'}})
        self.ev = dict(binding=binding(self.row), waveform_sha256='wave',
                       labels=dict(text='fail',timing='pass',speaker='pass',turn='unreviewed'),note='')

    def test_human_fail_overrides_teacher_agreement(self):
        h,r=review_flags(self.row,self.ev)
        self.assertEqual(adjudicate(dict(qwen_ref=0,whisper_ref=0,qwen_whisper=0,ref_units=1),hard=h,review=r)[0], 'QUARANTINE')

    def test_pass_does_not_remove_existing_hard_failure(self):
        self.ev['labels']['text']='pass'
        h,r=review_flags(self.row,self.ev)
        self.assertEqual(h,[])
        self.assertEqual(adjudicate(hard=['audio_bad']+h)[0],'QUARANTINE')

    def test_changed_target_or_source_rejected(self):
        for field in ('text','raw_text','manifest_sha256','source','audio'):
            row=copy.deepcopy(self.row);row[field]='changed'
            with self.assertRaises(ValueError):review_flags(row,self.ev)

    def test_changed_waveform_rejected(self):
        with self.assertRaises(ValueError):
            review_flags(self.row,self.ev,[dict(status='ok',audio={'sha256':'other'})])

    def test_note_keeps_review_needed(self):
        self.ev['labels']['text']='pass';self.ev['note']='ambiguous name'
        self.assertIn('human_note_requires_interpretation',review_flags(self.row,self.ev)[1])

    def test_load_exact_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);m=p/'m.jsonl';h=p/'h.json'
            m.write_text(json.dumps(self.row)+'\n')
            review=dict(training_eligible=False,reviewed_at='test',rows=[dict(key='k',source='s',labels=self.ev['labels'],note='')])
            h.write_text(json.dumps(review));self.assertEqual(set(load_reviews(h,m)),{'k'})
            review['rows']*=2;h.write_text(json.dumps(review))
            with self.assertRaises(ValueError):load_reviews(h,m)
            review['rows']=[];h.write_text(json.dumps(review))
            with self.assertRaises(ValueError):load_reviews(h,m)


if __name__=='__main__':unittest.main()
