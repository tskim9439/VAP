#!/usr/bin/env python3
"""Build a small, reproducible listening subset without changing source evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.selection_review_bundle import render
from vapasr.data.selection import file_digest


def spread(rows, n):
    """Duration quantiles, stable key tie break; diagnostic, not random sampling."""
    ordered = sorted(rows, key=lambda r: (r['audio_qc']['audio']['duration_s'], r['key']))
    if len(ordered) <= n:
        return ordered
    return [ordered[round(i * (len(ordered)-1) / (n-1))] for i in range(n)] if n > 1 else ordered[:1]


def wav_digest(path):
    """Check exported FLOAT mono/16k WAV and hash its actual waveform bytes."""
    data = path.read_bytes()
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('Not RIFF WAVE')
    pos = 12
    fmt = payload = None
    while pos + 8 <= len(data):
        tag, size = struct.unpack_from('<4sI', data, pos)
        body = data[pos+8:pos+8+size]
        if len(body) != size:
            raise ValueError('Truncated WAV')
        if tag == b'fmt ':
            fmt = struct.unpack_from('<HHIIHH', body)
        if tag == b'data':
            payload = body
        pos += 8 + size + (size % 2)
    if fmt is None or (fmt[0], fmt[1], fmt[2], fmt[5]) != (3, 1, 16000, 32) or not payload:
        raise ValueError('Unexpected exported WAV format')
    return hashlib.sha256(payload).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    original = a.bundle / 'review.jsonl'
    rows = list(map(json.loads, original.open()))
    exports = {r['key']: r for r in map(json.loads, (a.bundle / 'exports.jsonl').open())}
    rows = [r for r in rows if exports[r['key']]['files'] and 'error' not in exports[r['key']]]
    groups = [
        ('NIKL 2024 경계', 8, [r for r in rows if r['source'] == 'nikl-1000' and r['utt_id'].startswith('SDRW24')]),
        ('NIKL 2025 경계', 8, [r for r in rows if r['source'] == 'nikl-1000' and r['utt_id'].startswith('SDRW25')]),
        ('VoxPopuli 전사·잘림', 8, [r for r in rows if r['source'] == 'voxpopuli-train']),
        ('방송 다른 채널', 4, [r for r in rows if r['source'] == 'aihub-bc-train' and exports[r['key']].get('channel_0_1_max_abs_diff', 0) > 0]),
        ('방송 동일 채널', 2, [r for r in rows if r['source'] == 'aihub-bc-train' and exports[r['key']].get('channel_0_1_max_abs_diff') == 0]),
        ('AMI 동일 채널 대조', 2, [r for r in rows if r['source'] == 'ami' and exports[r['key']].get('channel_0_1_max_abs_diff') == 0]),
    ]
    selected = []
    for name, n, pool in groups:
        chosen = spread(pool, n)
        if len(chosen) != n:
            raise ValueError(f'Insufficient rows in {name}: {len(chosen)}/{n}')
        selected.extend(chosen)
    if len({r['key'] for r in selected}) != len(selected):
        raise ValueError('Duplicate subset key')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'audio').mkdir()
    copied = []
    with (a.out / 'exports.jsonl').open('x') as ef, (a.out / 'review.jsonl').open('x') as rf:
        for row in selected:
            e = exports[row['key']]
            for i, audio in enumerate(e['files']):
                rel = Path(audio['path'])
                if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'audio':
                    raise ValueError('Unsafe audio path')
                src = a.bundle / rel
                sha = wav_digest(src)
                if i == 0 and sha != e['audio']['sha256']:
                    raise ValueError('Original waveform hash mismatch')
                shutil.copyfile(src, a.out / rel)
                if file_digest(src) != file_digest(a.out / rel):
                    raise ValueError('Audio copy mismatch')
                copied.append(dict(path=str(rel), waveform_sha256=sha))
            ef.write(json.dumps(e, ensure_ascii=False)+'\n')
            rf.write(json.dumps(row, ensure_ascii=False)+'\n')
    teachers = [(name, a.bundle / f'{slug}.jsonl') for name, slug in
                [('Qwen', 'qwen'), ('Whisper-v2', 'whisper-v2'), ('Whisper-v3', 'whisper-v3')]]
    render(selected, a.out, teachers, input_path=original)
    page = (a.out / 'index.html').read_text()
    instructions = '''<section style="background:#fff4d6;padding:16px"><h2>이번 검수: 미해결 사례 32개</h2>
<p>NIKL 2024/25 각 8개: 전사가 실제 발음과 맞는지, 첫·끝 음절이 잘렸는지 확인합니다.
0.4초 길이 차이는 알려져 있지만, 청취만으로 앞뒤 padding이나 원대화 기준점을 확정하지 않습니다.</p>
<p>VoxPopuli 8개: VAD 연결 파일입니다. 전사 누락·잘림을 확인하되 원래 휴지·턴이 보존됐다고 판단하지 않습니다.</p>
<p>방송 6개·AMI 2개: 두 채널을 듣고 내용·화자가 같은지, 어느 채널이 더 명료한지 메모해 주세요.
채널 차이가 화자 분리라는 뜻은 아닙니다.</p>
<p><b>전사 / 경계·잘림 / 채널·화자만 평가</b>하고 턴 근거는 미검수로 두세요.
말을 자연스럽게 고친 교사 문장보다 실제 들리는 발음·반복을 우선합니다.
새로고침 전 검수 JSON을 내려받아 주세요. 이 표본은 전체 DB 승인이나 품질 비율 추정용이 아닙니다.</p></section>'''
    page = page.replace('<header>', instructions+'<header>', 1).replace("a.download='human-review.json'", "a.download='human-review-v03.json'")
    (a.out / 'index.html').write_text(page)
    summary = dict(complete=True, rows=len(selected), audio_files=len(copied),
        groups={name:n for name,n,_ in groups}, sources=Counter(r['source'] for r in selected),
        selection='duration_quantiles_within_purpose_group', original_input_sha256=file_digest(original),
        subset_sha256=file_digest(a.out/'review.jsonl'), exports_sha256=file_digest(a.out/'exports.jsonl'),
        script_sha256=file_digest(Path(__file__).resolve()), audio=copied,
        training_eligible=0, purpose='additional_listening_not_training_approval')
    (a.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='audio'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
