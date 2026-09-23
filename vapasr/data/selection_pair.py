"""Versioned expansion of completed v1 decisions; no ASR or TN rerun."""
import copy
import math

POLICY = dict(version='asr-pair-select-v2', max_pair_error=.05,
              reference_agreement_required=False, preserve_v1_safety_holds=True,
              preferred_target='qwen_raw', tn_scope='comparison_only',
              phase2_turn_approval=False)


def reselect(row):
    result=copy.deepcopy(row)
    old=row['selection_state']
    result['previous_selection_state']=old
    result['selection_policy']=POLICY
    result['training_eligible']=False
    # v1 has already checked source scope, audio identity, human vetoes and
    # generation warnings. Only the lexical-disagreement gate is relaxed.
    if old not in ('KEEP_ASR_SILVER','HOLD_DISAGREEMENT'):
        return result
    metrics=row['metrics']
    rate=metrics['qwen_whisper']
    if not math.isfinite(rate) or rate<0 or min(metrics['qwen_units'],metrics['whisper_units'])<=0:
        raise ValueError('Invalid teacher-pair metrics in eligible v1 row')
    if old=='KEEP_ASR_SILVER' and rate>POLICY['max_pair_error']:
        raise ValueError('Old KEEP contradicts v1 agreement contract')
    if rate<=POLICY['max_pair_error']:
        hyp=row['transcripts']['qwen_raw']
        if not isinstance(hyp,str) or not hyp.strip():raise ValueError('Missing verbatim Qwen target')
        result.update(selection_state='KEEP_ASR_SILVER',asr_candidate=True,
            selection_reason='teacher_pair_agrees_reference_diagnostic_only',
            retention_basis='reference_and_pair' if old=='KEEP_ASR_SILVER' else 'pair_only',
            text_quality='silver',recommended_training_target=dict(text=hyp,normalization='none',
                origin='qwen3_asr_pseudo_label',quality='silver_teacher_pair_agreement',
                display_style_verified=False))
    return result
