import copy
import unittest
from vapasr.data.selection_auto import prepare_row, decide_row


class AutoSelectionTest(unittest.TestCase):
    def setUp(self):
        self.row=dict(key='k',source='librispeech-960',split='train',manifest_sha256='m',
            selection_state='PENDING_AUDIO_TEACHERS',reasons=[],duration_s=2,
            audio=dict(path='a.wav'),text='test',raw_text='test')
        self.qc=dict(status='AUDIO_OK_TEXT_PENDING',audio=dict(sha256='s',duration_s=2,rms=.1,review=[]))

    def test_preserve_target(self):
        old=copy.deepcopy(self.row)
        out,state,_=prepare_row(self.row,self.qc)
        self.assertEqual(state,'READY_TEACHERS');self.assertEqual(self.row,old)
        self.assertEqual(out['text'],old['text']);self.assertFalse(out['training_eligible'])
        self.assertFalse(out['auto_selection']['turn_supervision'])

    def test_scope(self):
        self.row['split']='test'
        self.assertEqual(prepare_row(self.row,self.qc)[1],'EXCLUDE_SCOPE')

    def test_human_fail(self):
        self.assertEqual(prepare_row(self.row,self.qc,human=dict(labels=dict(text='fail')))[1],'HOLD_HUMAN')

    def test_channel_proof(self):
        self.qc['audio']['review']=['multichannel_without_explicit_channel_mapping']
        self.assertEqual(prepare_row(self.row,self.qc)[1],'HOLD_CHANNEL')
        ch=dict(status='IDENTICAL_CHANNELS',waveform_sha256='s',explicit_ch0_candidate='a.wav#ch0')
        self.assertEqual(prepare_row(self.row,self.qc,channel=ch)[0]['audio']['path'],'a.wav#ch0')
        ch['waveform_sha256']='changed'
        with self.assertRaises(ValueError):prepare_row(self.row,self.qc,channel=ch)

    def test_timebase_proof(self):
        self.row['source']='voxpopuli-train'
        self.qc['audio']['review']=['label_audio_duration_mismatch']
        self.assertEqual(prepare_row(self.row,self.qc)[1],'HOLD_AUDIO')
        self.assertEqual(prepare_row(self.row,self.qc,vad_verified=True)[1],'READY_TEACHERS')

    def test_low_level_and_long(self):
        self.qc['audio']['rms']=.001
        self.assertEqual(prepare_row(self.row,self.qc)[1],'HOLD_LOW_LEVEL')
        self.qc['audio']['duration_s']=31
        self.assertEqual(prepare_row(self.row,self.qc)[1],'HOLD_LONG')

    def test_error_not_all_deleted(self):
        self.qc.update(status='AUDIO_ERROR',error='missing_file')
        self.assertEqual(prepare_row(self.row,self.qc)[1],'HOLD_TECHNICAL')
        self.qc['error']='all_zero_audio_with_text'
        self.assertEqual(prepare_row(self.row,self.qc)[1],'EXCLUDE_ASR_PAIR')

    def test_decisions(self):
        row=prepare_row(self.row,self.qc)[0]
        teacher=dict(status='ok',audio=dict(sha256='s'),tn_flags=[],warning=None)
        m=dict(ref_units=10,qwen_units=10,whisper_units=10,qwen_ref=0,whisper_ref=0,qwen_whisper=0)
        self.assertEqual(decide_row(row,teacher,teacher,m)[0],'KEEP_ASR_SILVER')
        m['qwen_ref']=.1
        self.assertEqual(decide_row(row,teacher,teacher,m)[0],'HOLD_DISAGREEMENT')
        self.assertEqual(decide_row(row,None,teacher,m)[0],'HOLD_TEACHER')
        teacher['audio']['sha256']='changed'
        self.assertEqual(decide_row(row,teacher,teacher,m)[0],'HOLD_CHANGED_AUDIO')

    def test_short_exact(self):
        row=prepare_row(self.row,self.qc)[0];t=dict(status='ok',audio=dict(sha256='s'))
        m=dict(ref_units=4,qwen_units=4,whisper_units=4,qwen_ref=.01,whisper_ref=0,qwen_whisper=0)
        self.assertEqual(decide_row(row,t,t,m)[0],'HOLD_DISAGREEMENT')

    def test_repair_identity(self):
        with self.assertRaises(ValueError):
            prepare_row(self.row,self.qc,repaired=dict(source_key='other',manifest_sha256='m'))


if __name__=='__main__':unittest.main()
