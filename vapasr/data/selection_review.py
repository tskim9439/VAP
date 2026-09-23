"""Evidence-bound human review overlay. Never approves or rewrites a target."""
import json
from .selection import digest, file_digest

AXES = {'text', 'timing', 'speaker', 'turn'}
VALUES = {'pass', 'fail', 'uncertain', 'unreviewed'}


def binding(row):
    return digest({k: row[k] for k in ('key', 'source', 'utt_id', 'manifest_sha256',
                                      'audio', 'text', 'raw_text')})


def load_reviews(review_path, manifest_path):
    human = json.loads(review_path.read_text())
    if human.get('training_eligible') is not False:
        raise ValueError('Review is not an approval artifact')
    rows = {}
    with manifest_path.open() as stream:
        for r in map(json.loads, stream):
            if r['key'] in rows:
                raise ValueError('Duplicate review manifest key')
            rows[r['key']] = r
    result = {}
    for h in human['rows']:
        key = h['key']
        if key in result or key not in rows or h['source'] != rows[key]['source']:
            raise ValueError('Review key/source mismatch')
        if set(h['labels']) != AXES or not set(h['labels'].values()) <= VALUES:
            raise ValueError('Invalid review labels')
        if not isinstance(h.get('note', ''), str):
            raise ValueError('Invalid note')
        result[key] = dict(binding=binding(rows[key]),
            waveform_sha256=rows[key]['audio_qc']['audio']['sha256'],
            review_sha256=file_digest(review_path), manifest_sha256=file_digest(manifest_path),
            reviewed_at=human['reviewed_at'], labels=h['labels'], note=h.get('note', ''),
            training_eligible=False)
    if set(result) != set(rows):
        raise ValueError('Review must account for every manifest row')
    return result


def review_flags(row, evidence, teacher_rows=()):
    if binding(row) != evidence['binding']:
        raise ValueError('Reviewed target or source changed; requires new review')
    for t in teacher_rows:
        if t and t['status'] == 'ok' and t['audio']['sha256'] != evidence['waveform_sha256']:
            raise ValueError('Reviewed waveform changed')
    hard = ['human_'+axis+'_fail' for axis, value in sorted(evidence['labels'].items()) if value == 'fail']
    review = ['human_'+axis+'_uncertain' for axis, value in sorted(evidence['labels'].items()) if value == 'uncertain']
    if evidence['note'].strip():
        review.append('human_note_requires_interpretation')
    return hard, review
