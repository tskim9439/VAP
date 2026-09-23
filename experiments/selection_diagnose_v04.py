#!/usr/bin/env python3
"""Scoped CPU evidence collection. No target edits or automatic approvals."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest, digest
from vapasr.data.selection_audio import decode
from vapasr.data.selection_review import load_reviews


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    import numpy as np
    import soundfile as sf
    from vapasr.data.textnorm import target_ko, score_en, fingerprint
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review-manifest',type=Path,required=True)
    ap.add_argument('--human-review',type=Path,required=True)
    ap.add_argument('--teachers',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    overlay=load_reviews(a.human_review,a.review_manifest)
    rows={r['key']:r for r in map(json.loads,a.review_manifest.open())}
    teacher={}
    for name in ('qwen','whisper-v2','whisper-v3'):
        p=a.teachers/(name+'.jsonl');fp=json.loads(p.with_suffix('.fingerprint.json').read_text())
        if fp['input_sha256']!=file_digest(a.teachers/'review.jsonl'):raise ValueError('Teacher input mismatch')
        records={r['key']:r for r in map(json.loads,p.open())}
        for key in rows:
            t=records[key]
            if t['fingerprint']!=digest(fp) or t['status']!='ok' or t['audio']['sha256']!=rows[key]['audio_qc']['audio']['sha256']:
                raise ValueError('Teacher evidence mismatch')
        teacher[name]=records
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'audio').mkdir()
    save(a.out/'human-overlay.json',overlay)
    cards=[];assets=[]

    def export(x,name,label,**meta):
        x=np.ascontiguousarray(x,dtype='<f4')
        if len(x)==0 or not np.isfinite(x).all():raise ValueError('Invalid diagnostic waveform')
        path='audio/'+name+'.wav';sf.write(a.out/path,x,16000,subtype='FLOAT')
        rec=dict(path=path,label=label,duration_s=len(x)/16000,rms=float(np.sqrt(np.mean(x.astype(float)**2))),
                 peak=float(np.max(np.abs(x))),waveform_sha256=hashlib.sha256(x.tobytes()).hexdigest(),**meta)
        assets.append(rec);return rec

    def card(row,files,evidence):
        cards.append(dict(key=row['key'],source=row['source'],utt_id=row['utt_id'],files=files,
            target=row['text'],raw_text=row['raw_text'],human=overlay[row['key']],evidence=evidence,
            teachers={n:records[row['key']]['hyp'] for n,records in teacher.items()}))

    def original(row):
        x,info=decode(row)
        if info['sha256']!=row['audio_qc']['audio']['sha256']:raise ValueError('QC waveform changed')
        return x

    print('BROADCAST start',flush=True)
    bc=next(r for r in rows.values() if r['utt_id']=='vr_m_001_370_049_0348');stem=bc['utt_id']
    root=Path('/soundai/DB/raw/aihub');roots=sorted(root.glob('031*'))+sorted(root.glob('033*'))
    # Match all transcription genres/pairs; stem identity alone is not trusted.
    zips=[z for r in roots for z in sorted(r.glob('*/Training/*/TL_*음성전사_*.zip'))]
    def inspect_label(z):
        found=[]
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                if Path(name).name==stem+'.json':
                    data=f.read(name);d=json.loads(data)
                    found.append(dict(zip=str(z),member=name,json_sha256=hashlib.sha256(data).hexdigest(),
                        pron=d.get('발음전사'),spell=d.get('철자전사'),work=d.get('전사작업본'),
                        reproduced_target=target_ko(d.get('발음전사',''),'aihubbc')))
        return found
    with ThreadPoolExecutor(max_workers=8) as pool:
        variants=[v for group in pool.map(inspect_label,zips) for v in group]
    azips=[z for r in roots for z in sorted(r.glob('*/Training/*/TS_*한국어음성_*.zip'))]
    def inspect_audio(z):
        found=[]
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                if Path(name).name==stem+'.wav':
                    data=f.read(name)
                    with sf.SoundFile(io.BytesIO(data)) as wav:
                        x=wav.read(dtype='float32',always_2d=True)
                        found.append(dict(zip=str(z),member=name,bytes_sha256=hashlib.sha256(data).hexdigest(),
                            native_waveform_sha256=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
                            sr=wav.samplerate,channels=wav.channels,frames=len(x)))
        return found
    with ThreadPoolExecutor(max_workers=8) as pool:
        av=[v for group in pool.map(inspect_audio,azips) for v in group]
    if not variants or not av:raise ValueError('Missing broadcast source variants')
    mp=Path('/soundai/users/tskim/VAPKT-data/data/manifests/aihub-bc-train')
    if file_digest(mp/'streams.jsonl')!=bc['manifest_sha256']:raise ValueError('Broadcast manifest changed')
    matches=[]
    with (mp/'streams.jsonl').open() as f:
        for d in map(json.loads,f):
            matches.extend(dict(parent_id=d['id'],segment=s) for s in d['segments'] if s['utt_id']==stem)
    bc_result=dict(labels=variants,audio_variants=av,scanned_label_zips=len(zips),scanned_audio_zips=len(azips),
        manifest_matches=matches,current_tn=fingerprint(),manifest_stats=json.loads((mp/'stats.json').read_text()),
        all_targets_match=all(v['reproduced_target']==bc['text'] for v in variants),
        all_spell_match=all(v['spell']==bc['raw_text'] for v in variants),
        audio_variant_hashes=len({v['native_waveform_sha256'] for v in av}),training_eligible=False)
    save(a.out/'broadcast.json',bc_result)
    card(bc,[export(original(bc),'broadcast-original','기존 원음 그대로')],bc_result)
    print('BROADCAST complete '+str(len(variants))+' labels / '+str(len(av))+' audio',flush=True)

    print('AMI start',flush=True)
    ami=next(r for r in rows.values() if r['source']=='ami' and r['utt_id']=='A_00018')
    ann=Path('/soundai/DB/raw/ami/annotations/ami_public_manual_1.6.2.zip')
    with zipfile.ZipFile(ann) as z:
        metadata=z.read('corpusResources/meetings.xml')
        m=next(el for el in ET.fromstring(metadata) if el.get('observation')=='ES2010d')
        agents={el.get('nxt_agent'):int(el.get('channel')) for el in m}
        segdata=z.read('segments/ES2010d.A.segments.xml');seg=list(ET.fromstring(segdata))[18]
        wordsdata=z.read('words/ES2010d.A.words.xml')
        words=[dict(text=w.text,**w.attrib) for w in ET.fromstring(wordsdata)
               if w.get('starttime') and 231<float(w.get('starttime'))<238]
    start=ami['audio']['offset_s'];duration=ami['audio']['duration_s']
    if abs(float(seg.get('transcriber_start'))-start)>.001 or abs(float(seg.get('transcriber_end'))-start-duration)>.001:
        raise ValueError('AMI segment boundary mismatch')
    if Path(ami['audio']['path']).name!=f'ES2010d.Headset-{agents["A"]}.wav':raise ValueError('AMI channel mapping mismatch')
    x=original(ami);files=[export(x,'ami-original','기존 crop 원음')]
    peak=float(np.max(np.abs(x)));gain_db=min(12.,20*math.log10(10**(-1/20)/peak)) if peak else 0.
    files.append(export(x*10**(gain_db/20),'ami-gain','진단용 증폭 crop',gain_db=gain_db))
    for agent,ch in sorted(agents.items()):
        path=Path(ami['audio']['path']).parent/f'ES2010d.Headset-{ch}.wav'
        if not path.exists():continue
        r=dict(ami,audio=dict(path=str(path)+'#ch0',offset_s=start-2,duration_s=duration+4),duration_s=duration+4)
        y,info=decode(r)
        files.append(export(y,'ami-context-'+agent,f'앞뒤 2초 문맥: agent {agent}, Headset-{ch}',
            source_path=str(path),source_start_s=start-2,focus_start_s=2.,focus_end_s=2+duration))
    ami_result=dict(agents=agents,segment=seg.attrib,nearby_words=words,channel_mapping_matches=True,
        crop_qc_hash_matches=True,annotation_zip_sha256=file_digest(ann),
        meetings_xml_sha256=hashlib.sha256(metadata).hexdigest(),segments_sha256=hashlib.sha256(segdata).hexdigest(),
        words_sha256=hashlib.sha256(wordsdata).hexdigest(),files=files,training_eligible=False)
    save(a.out/'ami.json',ami_result);card(ami,files,ami_result)
    print('AMI complete',flush=True)

    print('VOXPOPULI start',flush=True)
    apath=Path('/soundai/users/tskim/VAPKT-data/data/labels/voxpopuli/asr_en.tsv')
    selected=[r for r in rows.values() if r['source']=='voxpopuli-train' and overlay[r['key']]['labels']['text']=='fail']
    ids={r['utt_id'] for r in selected};annotations={}
    csv.field_size_limit(1<<25)
    with apath.open() as f:
        for r in csv.DictReader(f,delimiter='|'):
            uid=r['session_id']+'-'+r['id_']
            if uid in ids:annotations[uid]=r
    if set(annotations)!=ids:raise ValueError('VoxPopuli annotation missing')
    search_roots=[Path('/soundai/users/tskim/VAPKT-data/data/audio/voxpopuli'),
        Path('/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/EN/TRAIN/OPEN/voxpopuli'),
        Path('/soundai/DB/raw/voxpopuli')]
    sessions={r['session_id'] for r in annotations.values()};found=[];search_errors=[];examined=0
    import os
    for root in search_roots:
        if not root.exists():search_errors.append(dict(root=str(root),status='absent'));continue
        for base,dirs,names in os.walk(root,onerror=lambda e:search_errors.append(dict(error=str(e)))):
            if len(Path(base).relative_to(root).parts)>=4:dirs[:]=[]
            for name in names:
                examined+=1
                if Path(name).stem in sessions and Path(name).suffix.lower() in ('.ogg','.wav','.flac','.mp3'):
                    found.append(str(Path(base)/name))
    vox=[]
    for r in selected:
        an=annotations[r['utt_id']];x=original(r);segments=ast.literal_eval(an['vad'])
        evidence=dict(annotation={k:an[k] for k in ('session_id','id_','original_text','normed_text','start_time','end_time','vad','speaker_id','split')},
            vad_segments=segments,expected_frames=sum(int(e*16000)-int(s*16000) for s,e in segments),
            actual_frames=len(x),annotation_sha256=file_digest(apath),
            same_lexical={n:score_en(records[r['key']]['hyp'])==score_en(r['text']) for n,records in teacher.items()},
            normalized_ref=score_en(r['text']),normalized_teachers={n:score_en(records[r['key']]['hyp']) for n,records in teacher.items()},
            original_session_candidates=[p for p in found if Path(p).stem==an['session_id']],training_eligible=False)
        files=[export(x,'vox-'+r['key'][:12],'배포 crop 원음')]
        card(r,files,evidence);vox.append(dict(key=r['key'],utt_id=r['utt_id'],**evidence))
    save(a.out/'voxpopuli.json',dict(rows=vox,search_roots=list(map(str,search_roots)),
        search_max_depth=4,examined_files=examined,search_errors=search_errors,
        original_session_files=found,scope='known dataset roots only; not proof of global absence'))
    save(a.out/'cards.json',cards)
    # Audio first, hypotheses hidden to reduce priming. No automatic label approval.
    esc=lambda x:html.escape(str(x));body=[]
    for c in cards:
        audio=''.join(f'<p>{esc(f["label"])} · {f["duration_s"]:.3f}s</p><audio controls preload="none" src="{esc(f["path"])}"></audio>' for f in c['files'])
        body.append(f'<article data-key="{c["key"]}"><h2>{esc(c["source"])} / {esc(c["utt_id"])}</h2>{audio}<p>먼저 실제 들리는 내용을 적어 주세요. 문맥에서 대상 구간은 AMI의 경우 2.000초부터입니다.</p><textarea class="heard" placeholder="들은 내용 / 들리지 않음 / 판단 불가"></textarea><details><summary>기존 타깃·교사·진단 근거 보기</summary><pre>{esc(json.dumps({k:v for k,v in c.items() if k!="files"},ensure_ascii=False,indent=2))}</pre></details><select><option value="unreviewed">미검수</option><option value="resolved">문제 원인/정정문 확인</option><option value="uncertain">판단 보류</option></select><textarea class="note" placeholder="원인, 정확한 정정문, 채널·위치에 대한 메모"></textarea></article>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>후속 원인 진단</title><style>body{max-width:980px;margin:24px auto;font-family:system-ui;padding:16px}article{border:1px solid #aaa;border-radius:8px;padding:18px;margin:22px 0}audio,textarea{width:100%}textarea{height:60px;margin:8px 0}pre{white-space:pre-wrap;word-break:break-all}button{padding:12px}</style><h1>후속 원인 진단: 8개 사례</h1><p>원본·전사는 변경하지 않았습니다. 먼저 후보 문장을 보지 않고 듣고, 이후 근거를 펼쳐 주세요. 증폭 사본은 진단용입니다. 다른 Headset은 다른 화자/누설음을 포함할 수 있습니다. VoxPopuli는 배포 crop이며 원래 문맥 복원이 아닙니다. 표시 선호는 실제 단어 정정과 구분해 메모해 주세요. 기존 실패 판정은 자동 해제하지 않습니다.</p><button onclick="download()">진단 JSON 저장</button><p>입력은 새로고침하면 사라집니다. 저장 후 파일을 전달해 주세요.</p>'''+''.join(body)+'''<script>function download(){let rows=[...document.querySelectorAll('article')].map(c=>({key:c.dataset.key,resolution:c.querySelector('select').value,heard_text:c.querySelector('.heard').value,note:c.querySelector('.note').value}));let data={reviewed_at:new Date().toISOString(),training_eligible:false,purpose:'diagnostic-followup-v04',rows};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='human-diagnostic-v04.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}</script></html>'''
    (a.out/'index.html').write_text(page)
    save(a.out/'summary.json',dict(complete=True,cards=len(cards),assets=assets,training_eligible=0,
        human_sha256=file_digest(a.human_review),review_manifest_sha256=file_digest(a.review_manifest),
        script_sha256=file_digest(Path(__file__).resolve()),gpu_used=0,
        overlay_sha256=file_digest(a.out/'human-overlay.json')))
    print('COMPLETE '+str(len(cards))+' cases; training not approved',flush=True)


if __name__=='__main__':main()
