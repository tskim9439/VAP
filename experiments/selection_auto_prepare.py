#!/usr/bin/env python3
"""Replay complete QC once; prepare an automatic smoke cohort or full shards."""
import argparse
from collections import Counter
import gzip
import heapq
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest,digest
from vapasr.data.selection_auto import prepare_row,POLICY
from vapasr.data.selection_review import load_reviews,binding
from experiments.selection_review_audit import joined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-source',type=int,default=128,help='0 writes all READY rows as bounded shards')
    p.add_argument('--shard-size',type=int,default=2048)
    a=p.parse_args()
    if a.per_source<0 or a.shard_size<1:raise ValueError('Invalid sizing')
    base=a.root/'v0.1-20260918';qcroot=base/'audio-census-w32'
    census=json.loads((base/'census.json').read_text());assert census['complete']
    repairs=a.root/'repair-v0.3-20260919';rs=json.loads((repairs/'summary.json').read_text())
    assert rs['complete'] and rs['original_manifest_unchanged']
    assert file_digest(repairs/'repaired.jsonl')==rs['repaired_sha256']
    repaired={r['source_key']:r for r in map(json.loads,(repairs/'repaired.jsonl').open())}
    channelroot=a.root/'channels-v0.3-20260919';chsummary=json.loads((channelroot/'summary.json').read_text())
    assert chsummary['complete'] and file_digest(channelroot/'channels.jsonl')==chsummary['output_sha256']
    channels={r['key']:r for r in map(json.loads,(channelroot/'channels.jsonl').open())}
    vad=json.loads((a.root/'voxpopuli-timebase-v0.3-20260919/summary.json').read_text())
    assert vad['complete'] and set(vad['counts'])=={'VAD_CONCAT_MATCH'}
    reviewroot=a.root/'review-v03-input-20260920'
    human=load_reviews(reviewroot/'human-review-20260919-232823.json',reviewroot/'review.jsonl')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'inputs').mkdir()
    (a.out/'policy.json').write_text(json.dumps(POLICY,indent=2))
    summary=dict(complete=False,mode='full' if not a.per_source else 'automatic_smoke',
        per_source=a.per_source,policy=POLICY,census_sha256=file_digest(base/'census.json'),
        human_review_sha256=file_digest(reviewroot/'human-review-20260919-232823.json'),
        repair_evidence_sha256=rs['evidence_sha256'],channel_sha256=chsummary['output_sha256'],
        prepare_code_sha256=file_digest(Path(__file__).resolve()),
        policy_code_sha256=file_digest(Path(__file__).resolve().parents[1]/'vapasr/data/selection_auto.py'),
        sources={},shards=[],training_eligible=0)
    buffer=[];sample=[]
    def write_shard():
        if not buffer:return
        name=f'inputs/part-{len(summary["shards"]):05d}.jsonl';dest=a.out/name
        with dest.open('x') as f:
            for r in buffer:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        summary['shards'].append(dict(path=name,rows=len(buffer),sha256=file_digest(dest),
            audio_h=sum(r['duration_s'] for r in buffer)/3600))
        buffer.clear()
    with gzip.open(a.out/'preflight.jsonl.gz','wt',compresslevel=1) as journal:
        for source,meta in census['sources'].items():
            marker=json.loads((qcroot/(source+'.done.json')).read_text())
            if marker.get('status')=='SOURCE_HOLD':
                summary['sources'][source]=dict(state='HOLD_SOURCE',note='existing_source_integrity_hold');continue
            cp=base/(source+'.candidates.jsonl.gz');qp=Path(marker['output'])
            csha=file_digest(cp)
            if file_digest(qp)!=marker['output_sha256']:raise ValueError('QC hash mismatch')
            if file_digest(meta['path'])!=meta['sha256']:raise ValueError('Upstream manifest changed')
            if source in chsummary['fingerprints']:
                cf=chsummary['fingerprints'][source]
                if cf['candidates_sha256']!=csha or cf['qc_sha256']!=marker['output_sha256']:raise ValueError('Channel dataset changed')
            verified_vad=source=='voxpopuli-train' and vad['candidates_sha256']==csha and vad['qc_sha256']==marker['output_sha256']
            if source=='voxpopuli-train' and not verified_vad:raise ValueError('VAD proof changed')
            counts=Counter();hours=Counter();qc_counts=Counter();heap=[]
            with gzip.open(cp,'rt') as cf,gzip.open(qp,'rt') as qf:
                for row,qc in joined(map(json.loads,cf),map(json.loads,qf)):
                    if row['manifest_sha256']!=meta['sha256']:raise ValueError('Candidate manifest mismatch')
                    qc_counts[qc['status']]+=1;h=human.get(row['key'])
                    if h and binding(row)!=h['binding']:raise ValueError('Human evidence changed')
                    out,state,reasons=prepare_row(row,qc,human=h,channel=channels.get(row['key']),
                        repaired=repaired.get(row['key']),vad_verified=verified_vad)
                    counts[state]+=1;hours[state]+=qc.get('audio',{}).get('duration_s',0)/3600
                    journal.write(json.dumps(dict(key=row['key'],source=source,state=state,reasons=reasons))+'\n')
                    if out:
                        if a.per_source:
                            rank=int(digest(['auto-smoke-v1',row['key']]),16)
                            item=(-rank,row['key'],out)
                            if len(heap)<a.per_source:heapq.heappush(heap,item)
                            elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
                        else:
                            buffer.append(out)
                            if len(buffer)>=a.shard_size:write_shard()
            if qc_counts!=Counter(marker['counts']):raise ValueError('QC count mismatch')
            sample.extend(x[2] for x in heap)
            summary['sources'][source]=dict(counts=counts,audio_h=hours,selected_smoke=len(heap),
                candidates_sha256=csha,qc_sha256=marker['output_sha256'])
            print(json.dumps(dict(source=source,counts=counts,sampled=len(heap))),flush=True)
    if a.per_source:
        # Interleave sources before dividing the smoke cohort into two balanced shards.
        sample.sort(key=lambda r:digest(['mix',r['key']]))
        mid=(len(sample)+1)//2
        for chunk in (sample[:mid],sample[mid:]):buffer.extend(chunk);write_shard()
    else:write_shard()
    summary['complete']=True;summary['selected_rows']=sum(s['rows'] for s in summary['shards'])
    summary['preflight_sha256']=file_digest(a.out/'preflight.jsonl.gz')
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('COMPLETE prepared '+str(summary['selected_rows'])+' automatic screening rows',flush=True)


if __name__=='__main__':main()
