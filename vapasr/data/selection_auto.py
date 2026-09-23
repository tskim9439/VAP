"""Conservative automatic ASR selection, not calibrated gold or turn labels."""
import copy
from .selection import digest

POLICY = dict(version='asr-auto-select-v1',max_duration_s=30.,low_rms=.005,
              max_error=.05,short_units=4,auto_relabel=False,
              audit_budget=40,phase2_turn_approval=False)


def prepare_row(row, qc, *, human=None, channel=None, repaired=None, vad_verified=False):
    """Exclude only unusable ASR pairs; unresolved evidence enters HOLD, not deletion."""
    if row['selection_state'] != 'PENDING_AUDIO_TEACHERS' or row['split'] != 'train':
        return None,'EXCLUDE_SCOPE',['not_train_candidate']
    if human:
        if any(v in ('fail','uncertain') for v in human['labels'].values()) or human.get('note','').strip():
            return None,'HOLD_HUMAN',['existing_human_evidence']
    original=copy.deepcopy(row)
    if repaired is not None:
        if repaired['source_key'] != row['key'] or repaired['manifest_sha256'] != row['manifest_sha256']:
            raise ValueError('Repair identity mismatch')
        row=copy.deepcopy(repaired);qc=row['audio_qc']
    else:row=copy.deepcopy(row)
    if row.get('reasons'):
        return None,'HOLD_METADATA',list(row['reasons'])
    if qc['status']=='AUDIO_ERROR':
        error=qc.get('error','')
        impossible=any(s in error for s in ('all_zero_audio_with_text','nonfinite_audio','empty_or_invalid_crop'))
        return None,'EXCLUDE_ASR_PAIR' if impossible else 'HOLD_TECHNICAL',[error]
    info=qc['audio'];flags=list(info.get('review',[]));fixes=[]
    if 'multichannel_without_explicit_channel_mapping' in flags:
        if not channel or channel['status']!='IDENTICAL_CHANNELS':
            return None,'HOLD_CHANNEL',['unresolved_channel']
        if channel['waveform_sha256']!=info['sha256']:raise ValueError('Channel proof mismatch')
        row['audio']['path']=channel['explicit_ch0_candidate']
        flags.remove('multichannel_without_explicit_channel_mapping');fixes.append('identical_channel_explicit_ch0')
    standalone=row['audio'].get('offset_s') is None and row['audio'].get('duration_s') is None
    if standalone and (row['source']=='aihub-bc-train' or (row['source']=='voxpopuli-train' and vad_verified)):
        flags=[f for f in flags if f!='label_audio_duration_mismatch'];fixes.append('verified_file_duration')
    if flags:return None,'HOLD_AUDIO',flags
    if not 0 < info['duration_s'] <= POLICY['max_duration_s']:
        return None,'HOLD_LONG',['outside_v1_teacher_duration']
    if info['rms'] < POLICY['low_rms']:
        return None,'HOLD_LOW_LEVEL',['low_rms_not_silence_label']
    row['duration_s']=info['duration_s']
    row['auto_selection']=dict(policy=digest(POLICY),original_key=original['key'],
        expected_waveform_sha256=info['sha256'],fixes=fixes,original_duration_s=original['duration_s'],
        original_audio=original['audio'],timing_scope='clip_asr_only',
        speaker_supervision=False,turn_supervision=False)
    row['training_eligible']=False
    return row,'READY_TEACHERS',fixes


def decide_row(row, qwen, whisper, metrics):
    if not all(t and t.get('status')=='ok' for t in (qwen,whisper)):
        return 'HOLD_TEACHER','missing_or_failed_teacher'
    expected=row['auto_selection']['expected_waveform_sha256']
    if any(t['audio']['sha256']!=expected for t in (qwen,whisper)):
        return 'HOLD_CHANGED_AUDIO','waveform_changed'
    if row.get('reasons'):return 'HOLD_METADATA','upstream_reasons'
    if any(t.get('tn_flags') or t.get('warning') for t in (qwen,whisper)):
        return 'HOLD_GENERATION','tn_or_generation_warning'
    if not metrics or not min(metrics['ref_units'],metrics['qwen_units'],metrics['whisper_units']):
        return 'HOLD_TEXT','empty_text'
    threshold=0 if metrics['ref_units']<=POLICY['short_units'] else POLICY['max_error']
    if max(metrics['qwen_ref'],metrics['whisper_ref'],metrics['qwen_whisper'])<=threshold:
        return 'KEEP_ASR_SILVER','reference_and_dual_teacher_agree_not_gold'
    return 'HOLD_DISAGREEMENT','retain_for_rule_improvement_not_manual_queue'
