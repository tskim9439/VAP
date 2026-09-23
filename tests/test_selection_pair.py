import copy
import unittest
import json
from pathlib import Path
import tempfile
from experiments.selection_reselect_pair import process_shard
from vapasr.data.selection import file_digest
from vapasr.data.selection_pair import reselect


class PairSelectionTest(unittest.TestCase):
    def row(self,rate=.05,state='HOLD_DISAGREEMENT'):
        return dict(selection_state=state,asr_candidate=False,text='old reference',
            transcripts=dict(qwen_raw="Don't change My Style!",whisper_raw='different punctuation'),
            metrics=dict(qwen_whisper=rate,qwen_ref=.8,whisper_ref=.8,ref_units=1,qwen_units=20,whisper_units=20),
            recommended_training_target=None,turn_quality='unverified')

    def test_promote_without_reference_agreement(self):
        r=self.row();before=copy.deepcopy(r);v=reselect(r)
        self.assertEqual(v['selection_state'],'KEEP_ASR_SILVER')
        self.assertEqual(v['retention_basis'],'pair_only')
        self.assertEqual(v['recommended_training_target']['text'],"Don't change My Style!")
        self.assertEqual(v['text'],'old reference');self.assertEqual(r,before)
        self.assertFalse(v['training_eligible']);self.assertEqual(v['turn_quality'],'unverified')

    def test_threshold(self):
        self.assertEqual(reselect(self.row(.050001))['selection_state'],'HOLD_DISAGREEMENT')
        self.assertEqual(reselect(self.row(0))['selection_state'],'KEEP_ASR_SILVER')

    def test_keep_is_preserved(self):
        self.assertEqual(reselect(self.row(0,'KEEP_ASR_SILVER'))['retention_basis'],'reference_and_pair')

    def test_safety_holds_are_not_promoted(self):
        for state in ['HOLD_GENERATION','HOLD_TEXT','HOLD_CHANGED_AUDIO','HOLD_TEACHER','HOLD_METADATA','HOLD_HUMAN']:
            self.assertEqual(reselect(self.row(0,state))['selection_state'],state)

    def test_invalid_metrics(self):
        for rate in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):reselect(self.row(rate))
        with self.assertRaises(ValueError):reselect(self.row(.1,'KEEP_ASR_SILVER'))

    def test_empty_teacher(self):
        r=self.row();r['metrics']['qwen_units']=0
        with self.assertRaises(ValueError):reselect(r)

    def test_shard_resume_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'old';out=Path(d)/'new';name='part-00000'
            sdir=source/'results'/name;sdir.mkdir(parents=True)
            (out/'results').mkdir(parents=True);(out/'work').mkdir()
            r=self.row();r.update(key='x',source='db',duration_s=3)
            decisions=sdir/'decisions.jsonl';decisions.write_text(json.dumps(r)+'\n')
            old=dict(rows=1,sources=dict(db=dict(HOLD_DISAGREEMENT=1)),comparison_tn=dict(version='fixture'),
                outputs={'decisions.jsonl':file_digest(decisions)})
            (sdir/'summary.json').write_text(json.dumps(old))
            job=(source,out,name,old);result=process_shard(job)
            self.assertEqual(result['added'],{'db':1})
            self.assertEqual(process_shard(job),result)
            target=out/'results'/name/'keep-asr.jsonl'
            kept=json.loads(target.read_text());self.assertEqual(kept['recommended_training_target']['text'],r['transcripts']['qwen_raw'])
            self.assertEqual(json.loads(decisions.read_text()),r)
            target.write_text('modified')
            with self.assertRaises(ValueError):process_shard(job)
        r=self.row();r['transcripts']['qwen_raw']=''
        with self.assertRaises(ValueError):reselect(r)


if __name__=='__main__':unittest.main()
