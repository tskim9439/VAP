#!/usr/bin/env python3
"""Compare VoxPopuli files against official VAD concatenation, not envelope span."""
import argparse
import ast
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest
from experiments.selection_review_audit import joined


def pieces(value, sr=16000):
    raw=ast.literal_eval(value)
    if not isinstance(raw,list) or not raw:raise ValueError("empty/invalid VAD")
    result=[];pos=0;last=0
    for pair in raw:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError("invalid VAD interval")
        start,end=map(float,pair)
        if not all(map(math.isfinite,(start,end))) or start<last or end<=start:
            raise ValueError("nonmonotonic/invalid VAD")
        frames=int(end*sr)-int(start*sr)
        if frames<=0:raise ValueError("empty VAD frames")
        result.append(dict(source_start_s=start,source_end_s=end,
            clip_start_sample=pos,clip_end_sample=pos+frames))
        pos+=frames;last=end
    return result,pos


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection",required=True,type=Path)
    ap.add_argument("--qc",required=True,type=Path)
    ap.add_argument("--annotations",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args();csv.field_size_limit(1<<25)
    ann={}
    with a.annotations.open() as f:
        for r in csv.DictReader(f,delimiter="|"):
            if r['split']!='train':continue
            key=r['session_id']+'-'+r['id_']
            if key in ann:raise ValueError("Duplicate annotation ID")
            ann[key]=r
    marker=json.loads((a.qc/'voxpopuli-train.done.json').read_text())
    if file_digest(marker['output'])!=marker['output_sha256']:raise ValueError("QC checksum mismatch")
    cp=a.selection/'voxpopuli-train.candidates.jsonl.gz'
    a.out.mkdir(parents=True,exist_ok=False)
    counts=Counter();deltas=Counter();qcounts=Counter();intervals=Counter()
    with gzip.open(cp,'rt') as cf,gzip.open(marker['output'],'rt') as qf,(a.out/'vad-timebase.jsonl').open('x') as out:
        for row,qc in joined(map(json.loads,cf),map(json.loads,qf)):
            qcounts[qc['status']]+=1
            record=dict(key=row['key'],source=row['source'],utt_id=row['utt_id'],training_eligible=False,
                        declared_duration_s=row['duration_s'],audio=row['audio'])
            try:
                r=ann[row['utt_id']];info=qc['audio']
                if info['original_sample_rate']!=16000:raise ValueError("Unexpected file sample rate")
                segments,frames=pieces(r['vad'])
                actual=round(info['duration_s']*16000);delta=actual-frames
                matched=abs(delta)<=2
                record.update(status='VAD_CONCAT_MATCH' if matched else 'HOLD_VAD_MISMATCH',
                    sample_rate=16000,vad_segments=segments,expected_vad_frames=frames,
                    actual_frames=actual,frame_difference=delta,file_duration_s=info['duration_s'],
                    annotation_start_s=float(r['start_time']),annotation_end_s=float(r['end_time']),
                    waveform_sha256=info['sha256'],timebase='vad_concatenation',
                    single_affine_offset_valid=len(segments)==1 and matched,
                    proposed_duration_s=info['duration_s'] if matched else None,
                    original_dialogue_turn_supervision=False)
                deltas[str(delta)]+=1;intervals[str(len(segments))]+=1
            except Exception as e:record.update(status='HOLD',error=f'{type(e).__name__}: {e}')
            counts[record['status']]+=1;out.write(json.dumps(record,ensure_ascii=False)+'\n')
    if qcounts!=Counter(marker['counts']):raise ValueError('QC count mismatch')
    summary=dict(complete=True,counts=counts,frame_difference=deltas,segment_count=intervals,
        annotations_sha256=file_digest(a.annotations),candidates_sha256=file_digest(cp),
        qc_sha256=marker['output_sha256'],script_sha256=file_digest(Path(__file__).resolve()),
        official_code='https://github.com/facebookresearch/voxpopuli/blob/main/voxpopuli/get_asr_data.py',
        training_eligible=0,applied_changes=0)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
