#!/usr/bin/env python3
"""Bounded, resumable dual-teacher queue and automatic ASR-only selection."""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest, digest, agreement
from vapasr.data.selection_auto import POLICY, decide_row

STORAGE_POLICY = dict(version='verbatim-asr-v1', tn_scope='comparison_only',
    preferred_target='qwen_raw', target_origin='teacher_pseudo_label', overwrite_source=False)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def score(inp, qpath, wpath, out):
    from vapasr.data.textnorm import score_en, score_ko, fingerprint
    with inp.open() as f: rows = [json.loads(x) for x in f]
    keys = {r['key'] for r in rows}
    if len(keys) != len(rows): raise ValueError('Duplicate input')
    teachers = []
    for name, path in [('qwen', qpath), ('whisper', wpath)]:
        fp = json.loads(path.with_suffix('.fingerprint.json').read_text())
        if fp['teacher'] != name or fp['input_sha256'] != file_digest(inp) or fp['tn'] != fingerprint():
            raise ValueError('Teacher provenance mismatch')
        with path.open() as f: records = [json.loads(x) for x in f]
        mapping = {r['key']: r for r in records}
        if len(records) != len(mapping) or set(mapping) != keys:
            raise ValueError('Incomplete or duplicate teacher coverage')
        if any(r['fingerprint'] != digest(fp) or r['teacher'] != name for r in records):
            raise ValueError('Row fingerprint mismatch')
        teachers.append(mapping)
    out.mkdir(parents=True,exist_ok=True)
    counts, hours = defaultdict(Counter), defaultdict(Counter)
    audits = []
    with (out/'decisions.jsonl').open('w') as decisions, (out/'keep-asr.jsonl').open('w') as keep, (out/'hold.jsonl').open('w') as hold:
        for row in rows:
            qr, wr = (t[row['key']] for t in teachers)
            metrics = None
            if qr['status'] == wr['status'] == 'ok':
                norm = (lambda x: list(score_ko(x, False))) if row['lang'] == 'Korean' else (lambda x: score_en(x).split())
                metrics = agreement(norm(row['text']), norm(qr['hyp']), norm(wr['hyp']))
            state, reason = decide_row(row, qr, wr, metrics)
            # Keep display conventions exactly as produced. The legacy text
            # field stays as provenance, never silently becomes a pseudo-label.
            row['transcripts']=dict(source_raw=row.get('raw_text'),
                source_manifest_target=row['text'],qwen_raw=qr.get('hyp'),whisper_raw=wr.get('hyp'))
            row['storage_policy']=STORAGE_POLICY
            row['recommended_training_target']=(dict(text=qr['hyp'],normalization='none',
                origin='qwen3_asr_pseudo_label',quality='silver_lexical_agreement',
                display_style_verified=False) if state=='KEEP_ASR_SILVER' else None)
            row.update(selection_state=state, selection_reason=reason, metrics=metrics,
                asr_candidate=state=='KEEP_ASR_SILVER', training_eligible=False,
                text_quality='silver' if state=='KEEP_ASR_SILVER' else 'unresolved',
                timing_quality='unverified', speaker_quality='unverified', turn_quality='unverified')
            counts[row['source']][state] += 1
            hours[row['source']][state] += row['duration_s']/3600
            line = json.dumps(row, ensure_ascii=False)+'\n'
            decisions.write(line)
            (keep if row['asr_candidate'] else hold).write(line)
            audits.append(dict(key=row['key'],source=row['source'],state=state))
    result = dict(complete=True, rows=len(rows), sources=counts, audio_h=hours,
        policy=POLICY,storage_policy=STORAGE_POLICY,comparison_tn=fingerprint(),
        implementation_sha256=file_digest(Path(__file__).resolve()),
        input_sha256=file_digest(inp), qwen_sha256=file_digest(qpath),
        whisper_sha256=file_digest(wpath), training_eligible=0,
        outputs={p.name:file_digest(p) for p in out.glob('*.jsonl')})
    save(out/'summary.json',result)
    return result, audits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--gpus',default='1,2,3,4')
    p.add_argument('--qwen-model',type=Path,default=Path('/soundai/Model/Qwen3-ASR-0.6B'))
    p.add_argument('--whisper-model',type=Path,default=Path('/soundai/users/tskim/VAPKT-data/baselines/whisper-large-v2'))
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--qwen-batch',type=int)
    p.add_argument('--whisper-batch',type=int)
    p.add_argument('--wait-preparation-pid',type=int,help='Wait up to two hours for this preparation process')
    a=p.parse_args(); base=a.prepared.resolve()
    gpus=[int(x) for x in a.gpus.split(',')]
    if not 1<=len(gpus)<=4 or len(set(gpus))!=len(gpus):raise ValueError('Maximum four unique GPUs')
    batches=dict(qwen=a.qwen_batch or a.batch,whisper=a.whisper_batch or a.batch)
    if min(batches.values())<1:raise ValueError('Invalid batch size')
    lock=(base/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (base/'summary.json').exists() and a.wait_preparation_pid:
        print(f'WAIT preparation pid={a.wait_preparation_pid}',flush=True)
        deadline=time.monotonic()+7200
        while not (base/'summary.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Preparation wait exceeded two hours')
            os.kill(a.wait_preparation_pid,0)
            save(base/'run-status.json',dict(state='waiting_preparation',pid=a.wait_preparation_pid))
            time.sleep(10)
    prep=json.loads((base/'summary.json').read_text())
    if not prep['complete'] or prep['policy']!=POLICY:raise ValueError('Preparation incomplete or policy changed')
    root=Path(__file__).resolve().parents[1]
    if prep['policy_code_sha256']!=file_digest(root/'vapasr/data/selection_auto.py'):
        raise ValueError('Policy implementation changed')
    snapshot=base/'code-snapshot'
    if not snapshot.exists():
        staging=base/'code-snapshot-building';staging.mkdir(exist_ok=True)
        for directory in ['vapasr','experiments']:
            for source in (root/directory).rglob('*.py'):
                dest=staging/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
        hashes={str(f.relative_to(staging)):file_digest(f) for f in staging.rglob('*.py')}
        save(staging/'hashes.json',hashes);staging.rename(snapshot)
    for name, sha in json.loads((snapshot/'hashes.json').read_text()).items():
        if file_digest(snapshot/name)!=sha:raise ValueError('Snapshot changed')
    # Score through the same frozen implementation after a resume, not mutable source code.
    if Path(__file__).resolve()!=snapshot/'experiments/selection_auto_run.py':
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        os.execv(sys.executable,[sys.executable,str(snapshot/'experiments/selection_auto_run.py'),*sys.argv[1:]])
    jobs=[]
    for s in prep['shards']:
        inp=base/s['path']
        if file_digest(inp)!=s['sha256']:raise ValueError('Input shard changed')
        for teacher,model in [('qwen',a.qwen_model),('whisper',a.whisper_model)]:
            dest=base/'teachers'/inp.stem/(teacher+'.jsonl');dest.parent.mkdir(parents=True,exist_ok=True)
            jobs.append(dict(input=inp,output=dest,teacher=teacher,model=model))
    config=dict(prepared_sha256=file_digest(base/'summary.json'),batches=batches,
        qwen_model=str(a.qwen_model),whisper_model=str(a.whisper_model),decode_workers=8,
        storage_policy=STORAGE_POLICY)
    configpath=base/'run-config.json'
    if configpath.exists() and json.loads(configpath.read_text())!=config:raise ValueError('Run recipe changed')
    save(configpath,config)
    pending=list(jobs);active={};completed=0;started=time.time();failures=[]
    finished=set();scored=set();totals=defaultdict(Counter);hours=defaultdict(Counter);audit=[];shards=[]
    def publish(complete=False):
        save(base/'automatic-summary.json',dict(complete=complete,rows=sum(s['rows'] for s in shards),
            expected_rows=prep['selected_rows'],policy=POLICY,storage_policy=STORAGE_POLICY,
            sources=totals,audio_h=hours,shards=shards,
            elapsed_s=time.time()-started,training_eligible=0,scope=prep['mode']))
    try:
        while pending or active:
            for gpu,(proc,log,job) in list(active.items()):
                rc=proc.poll()
                if rc is not None:
                    log.close();del active[gpu]
                    if rc:failures.append(dict(output=str(job['output']),returncode=rc))
                    else:
                        completed+=1;finished.add((job['input'].stem,job['teacher']))
            if failures:raise RuntimeError('Teacher process failed; see run-status.json')
            memory={int(i):int(m) for i,m in (x.split(',') for x in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines())}
            for gpu in gpus:
                if not pending:break
                if gpu in active or memory.get(gpu,999999)>1024:continue
                job=pending.pop(0);log=job['output'].with_suffix('.log').open('a')
                cmd=[sys.executable,str(snapshot/'experiments/selection_teachers.py'),
                    '--input',str(job['input']),'--output',str(job['output']),
                    '--teacher',job['teacher'],'--model',str(job['model']),
                    '--gpu',str(gpu),'--batch',str(batches[job['teacher']]),'--decode-workers','8']
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=snapshot)
                active[gpu]=(proc,log,job)
                print(f'START gpu={gpu} pid={proc.pid} {job["output"]}',flush=True)
            save(base/'run-status.json',dict(state='running',completed_jobs=completed,total_jobs=len(jobs),
                elapsed_s=time.time()-started,active={g:dict(pid=p.pid,output=str(j['output'])) for g,(p,l,j) in active.items()},failures=failures))
            for s in prep['shards']:
                inp=base/s['path'];name=inp.stem
                if name in scored or not all((name,t) in finished for t in ('qwen','whisper')):continue
                d=base/'teachers'/name
                result,candidates=score(inp,d/'qwen.jsonl',d/'whisper.jsonl',base/'results'/name)
                scored.add(name);shards.append(result)
                for source,c in result['sources'].items():totals[source].update(c)
                for source,c in result['audio_h'].items():hours[source].update(c)
                audit.extend(candidates);audit.sort(key=lambda r:digest(['audit-v1',r['key']]))
                audit=([r for r in audit if r['state']=='KEEP_ASR_SILVER'][:32]+
                       [r for r in audit if r['state']!='KEEP_ASR_SILVER'][:8])
                publish();print(f'SCORED {name} rows={result["rows"]}',flush=True)
            if active or pending:time.sleep(5)
        # Globally bounded audit, never a per-row manual-resolution queue.
        audit.sort(key=lambda r:digest(['audit-v1',r['key']]))
        selected=[r for r in audit if r['state']=='KEEP_ASR_SILVER'][:32]
        selected += [r for r in audit if r['state']!='KEEP_ASR_SILVER'][:8]
        save(base/'optional-audit.json',dict(required_to_continue=False,budget=40,rows=selected,
            note='Small diagnostic audit; not calibrated precision or database-wide certification'))
        if len(scored)!=len(prep['shards']):raise ValueError('Missing scored shards')
        publish(complete=True)
        save(base/'run-status.json',dict(state='complete',completed_jobs=completed,total_jobs=len(jobs)))
        print('COMPLETE '+json.dumps(totals),flush=True)
    except BaseException as e:
        for proc,log,job in active.values():
            proc.terminate()
        for proc,log,job in active.values():
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        save(base/'run-status.json',dict(state='failed',error=repr(e),failures=failures))
        raise


if __name__=='__main__':main()
