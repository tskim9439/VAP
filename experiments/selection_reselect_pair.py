#!/usr/bin/env python3
"""Reselect completed, hash-bound v1 outputs on CPU into a separate v2 root."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest
from vapasr.data.selection_pair import POLICY,reselect


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)


def process_shard(job):
    source,out,name,expected=job
    dest=out/'results'/name
    old=json.loads((source/'results'/name/'summary.json').read_text())
    if old!=expected:raise ValueError('Old shard summary changed: '+name)
    input_path=source/'results'/name/'decisions.jsonl'
    source_sha=old['outputs']['decisions.jsonl']
    if dest.exists():
        result=json.loads((dest/'summary.json').read_text())
        if result['source_sha256']!=source_sha or result['policy']!=POLICY or not result['complete']:
            raise ValueError('Invalid resume shard: '+name)
        for filename,sha in result['outputs'].items():
            if file_digest(dest/filename)!=sha:raise ValueError('Resume output changed')
        return result
    stage=out/'work'/name;stage.mkdir(parents=True,exist_ok=True)
    counts=defaultdict(Counter);hours=defaultdict(Counter);added=Counter();added_h=Counter();previous=Counter()
    rows=0;seen=set();hasher=hashlib.sha256()
    with input_path.open('rb') as src, (stage/'keep-asr.jsonl').open('w') as keep, (stage/'hold.jsonl').open('w') as hold, (stage/'decisions.jsonl').open('w') as audit:
        for line in src:
            hasher.update(line);r=json.loads(line)
            if r['key'] in seen:raise ValueError('Duplicate key in v1 shard')
            seen.add(r['key']);previous[r['selection_state']]+=1
            v=reselect(r);state=v['selection_state'];db=v['source'];duration=v['duration_s']/3600
            counts[db][state]+=1;hours[db][state]+=duration;rows+=1
            is_keep=state=='KEEP_ASR_SILVER'
            if is_keep and r['selection_state']!='KEEP_ASR_SILVER':added[db]+=1;added_h[db]+=duration
            (keep if is_keep else hold).write(json.dumps(v,ensure_ascii=False)+'\n')
            audit.write(json.dumps(dict(key=v['key'],source=db,previous=r['selection_state'],
                state=state,reason=v.get('selection_reason'),metrics=v.get('metrics')),ensure_ascii=False)+'\n')
    expected_counts=Counter()
    for c in old['sources'].values():expected_counts.update(c)
    if rows!=old['rows'] or previous!=expected_counts or hasher.hexdigest()!=source_sha:
        raise ValueError('Old decision content/count changed: '+name)
    result=dict(complete=True,name=name,rows=rows,policy=POLICY,source_sha256=source_sha,
        sources=counts,audio_h=hours,added=added,added_audio_h=added_h,previous=previous,
        training_eligible=0,comparison_tn=old['comparison_tn'],
        outputs={f.name:file_digest(f) for f in stage.glob('*.jsonl')})
    save(stage/'summary.json',result);stage.rename(dest)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();source=a.source.resolve();out=a.out.resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Use an independent output root')
    if not 1<=a.workers<=32:raise ValueError('Worker limit 1..32')
    aggregate=json.loads((source/'automatic-summary.json').read_text())
    prep=json.loads((source/'summary.json').read_text())
    if not aggregate['complete'] or aggregate['rows']!=prep['selected_rows']:raise ValueError('Source run incomplete')
    if aggregate['policy']['version']!='asr-auto-select-v1':raise ValueError('Expected v1 run')
    root=Path(__file__).resolve().parents[1]
    config=dict(policy=POLICY,source=str(source),source_summary_sha256=file_digest(source/'automatic-summary.json'),
        prepare_sha256=file_digest(source/'summary.json'),code_sha256=file_digest(Path(__file__).resolve()),
        policy_code_sha256=file_digest(root/'vapasr/data/selection_pair.py'))
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=config:raise ValueError('Resume fingerprint changed')
    elif any(p.name!='run.lock' for p in out.iterdir()):raise ValueError('Unfingerprinted existing output')
    save(out/'config.json',config)
    (out/'results').mkdir(exist_ok=True);(out/'work').mkdir(exist_ok=True)
    by_sha={s['input_sha256']:s for s in aggregate['shards']}
    if len(by_sha)!=len(prep['shards']):raise ValueError('Source shard mismatch')
    jobs=[(source,out,Path(s['path']).stem,by_sha[s['sha256']]) for s in prep['shards']]
    results={};sources=defaultdict(Counter);hours=defaultdict(Counter);added=Counter();added_h=Counter()
    started=time.time()
    def publish(complete=False):
        save(out/'summary.json',dict(complete=complete,policy=POLICY,source=str(source),
            expected_rows=aggregate['rows'],rows=sum(r['rows'] for r in results.values()),
            completed_shards=len(results),total_shards=len(jobs),sources=sources,audio_h=hours,
            added=added,added_audio_h=added_h,training_eligible=0,
            elapsed_s=time.time()-started,workers=a.workers,gpus=0,
            shards=[dict(name=k,rows=v['rows'],outputs=v['outputs']) for k,v in sorted(results.items())]))
    publish()
    try:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for future in as_completed([pool.submit(process_shard,j) for j in jobs]):
                r=future.result();results[r['name']]=r
                for db,c in r['sources'].items():sources[db].update(c)
                for db,h in r['audio_h'].items():hours[db].update(h)
                added.update(r['added']);added_h.update(r['added_audio_h'])
                publish();print(json.dumps(dict(shard=r['name'],completed=len(results),total=len(jobs),added=sum(r['added'].values()))),flush=True)
        if sum(r['rows'] for r in results.values())!=aggregate['rows']:raise ValueError('Output row count mismatch')
        old_kept=sum(c.get('KEEP_ASR_SILVER',0) for c in aggregate['sources'].values())
        kept=sum(c.get('KEEP_ASR_SILVER',0) for c in sources.values())
        if kept!=old_kept+sum(added.values()):raise ValueError('Retention accounting mismatch')
        publish(True);save(out/'run-status.json',dict(state='complete',rows=aggregate['rows'],kept=kept,added=sum(added.values())))
        print(f'COMPLETE kept={kept} added={sum(added.values())}',flush=True)
    except BaseException as e:
        save(out/'run-status.json',dict(state='failed',error=repr(e)));raise


if __name__=='__main__':main()
