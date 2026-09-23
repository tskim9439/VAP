import json
from pathlib import Path
import tempfile
import unittest
from experiments.selection_auto_run import score
from vapasr.data.selection import file_digest,digest
from vapasr.data.textnorm import fingerprint


class AutoScoreTest(unittest.TestCase):
    def test_join_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inp=root/'input.jsonl'
            row=dict(key='x',source='test-db',lang='English',text='hello there',duration_s=2,
                auto_selection=dict(expected_waveform_sha256='audio'),reasons=[])
            inp.write_text(json.dumps(row)+'\n')
            paths=[]
            for teacher in ['qwen','whisper']:
                path=root/(teacher+'.jsonl');paths.append(path)
                fp=dict(teacher=teacher,input_sha256=file_digest(inp),tn=fingerprint())
                path.with_suffix('.fingerprint.json').write_text(json.dumps(fp))
                path.write_text(json.dumps(dict(key='x',teacher=teacher,fingerprint=digest(fp),
                    status='ok',hyp='Hello there!' if teacher=='qwen' else 'HELLO THERE.',
                    audio=dict(sha256='audio')))+'\n')
            result,_=score(inp,*paths,root/'nested/result')
            self.assertEqual(result['sources']['test-db']['KEEP_ASR_SILVER'],1)
            kept=json.loads((root/'nested/result/keep-asr.jsonl').read_text())
            self.assertFalse(kept['training_eligible'])
            self.assertEqual(kept['text'],'hello there')
            self.assertEqual(kept['transcripts']['qwen_raw'],'Hello there!')
            self.assertEqual(kept['transcripts']['whisper_raw'],'HELLO THERE.')
            self.assertEqual(kept['recommended_training_target']['text'],'Hello there!')
            self.assertEqual(kept['recommended_training_target']['normalization'],'none')
            self.assertEqual(kept['storage_policy']['tn_scope'],'comparison_only')
            paths[1].write_text('')
            with self.assertRaises(ValueError):score(inp,*paths,root/'bad')


if __name__=='__main__':unittest.main()
