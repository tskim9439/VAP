#!/usr/bin/env python3
"""Qualitative held-out multi-speaker evaluation for a mono VAP-ASR checkpoint.

Selects deterministic overlap/non-overlap windows from Phase-2 dialogue corpora,
renders reference-speaker and model-emission timelines, and writes a listening
HTML bundle.  This intentionally reports pooled ASR only: a lanes=0 model does
not perform diarization.
"""
from __future__ import annotations
import argparse, collections, html, json, os, random, re
from pathlib import Path
import numpy as np, soundfile as sf, torch

from vapasr.data.dialogue import Dialogue
from vapasr.data.dialogue_dataset import DialogueWindowDataset
from vapasr.data.dialogue_mix import crop_audio
from vapasr.data.textnorm import score_en, score_ko
from vapasr.hf.infer import load_model, transcribe

def ed(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        q=[i]+[0]*len(b)
        for j,y in enumerate(b,1): q[j]=min(p[j]+1,q[j-1]+1,p[j-1]+(x!=y))
        p=q
    return p[-1]

def proxy(d,tok):
    for u in d.utterances:
        if u.tokens or not u.text: continue
        ids=tok(' '+u.text,add_special_tokens=False)['input_ids']
        u.tokens=[(t,u.start+(u.end-u.start)*(i+1)/len(ids)) for i,t in enumerate(ids)] if ids else []

def overlap_seconds(eps):
    events=[]
    for e in eps: events += [(e.start,1,e.speaker),(e.end,-1,e.speaker)]
    events.sort(key=lambda x:(x[0],-x[1])); active=collections.Counter(); last=None; total=0.
    for t,delta,spk in events:
        if last is not None and len([s for s,n in active.items() if n>0])>=2: total += max(0.,t-last)
        active[spk]+=delta; last=t
    return total

def esc(x): return html.escape(str(x),quote=True)
COL=['#0969da','#bf8700','#8250df','#1a7f37','#cf222e','#57606a']
def svg(path,wav,eps,emissions,dur,title,tok):
    W,H,x0,pw=1500,500,125,1340; q=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><text x="750" y="26" text-anchor="middle" font-family="sans-serif" font-size="17">{esc(title)}</text>']
    bins=min(pw,max(1,len(wav)//2)); ee=np.linspace(0,len(wav),bins+1).astype(int); amp=max(1e-4,float(np.max(np.abs(wav))))
    for j,(a,b) in enumerate(zip(ee[:-1],ee[1:])):
        z=wav[a:max(a+1,b)]; x=x0+j*pw/max(1,bins-1); ya=105-float(z.max())/amp*58; yb=105-float(z.min())/amp*58
        q.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ya:.1f}" y2="{yb:.1f}" stroke="#315a7d"/>')
    q.append('<text x="10" y="110" font-family="sans-serif">wave</text>')
    lanes=sorted({e.lane for e in eps if e.lane}); base=185
    for lane in lanes:
        y=base+(lane-1)*42; q += [f'<text x="10" y="{y+5}" font-family="sans-serif">ref lane {lane}</text>',f'<line x1="{x0}" x2="{x0+pw}" y1="{y}" y2="{y}" stroke="#ddd"/>']
    for e in eps:
        if not e.lane: continue
        y=base+(e.lane-1)*42; x=x0+pw*e.start/dur; ww=max(2,pw*(e.end-e.start)/dur); txt=tok.decode([t for t,_ in e.tokens],skip_special_tokens=True).strip()
        q.append(f'<rect x="{x:.1f}" y="{y-13}" width="{ww:.1f}" height="26" rx="4" fill="{COL[(e.lane-1)%len(COL)]}" opacity=".8"><title>{esc(txt)}</title></rect>')
    y=base+max(lanes or [1])*42+25; q += [f'<text x="10" y="{y+5}" font-family="sans-serif">model emit</text>',f'<line x1="{x0}" x2="{x0+pw}" y1="{y}" y2="{y}" stroke="#ddd"/>']
    for em in emissions:
        x=x0+pw*em['time_s']/dur; q.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y-15}" y2="{y+15}" stroke="#cf222e" stroke-width="2"/>')
    for sec in range(0,int(dur)+1,5):
        x=x0+pw*sec/dur; q += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="145" y2="{y+30}" stroke="#d8dee4"/>',f'<text x="{x:.1f}" y="{y+55}" text-anchor="middle" font-family="sans-serif" font-size="11">{sec}s</text>']
    q.append('</svg>'); path.write_text(''.join(q),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--sets',default='nikl2020:6,chime6:6'); ap.add_argument('--seed',type=int,default=33000); ap.add_argument('--mono-cache',default=None); a=ap.parse_args()
    out=Path(a.out); (out/'audio').mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(exist_ok=True)
    model,tok=load_model(a.checkpoint,device='cuda',dtype=torch.bfloat16); assert model.config.lanes==0
    rows=[]
    for spec in a.sets.split(','):
        corpus,n=spec.split(':'); n=int(n); src=Path(a.data)/f'{corpus}.refined.dialogues.jsonl'; src=src if src.exists() else Path(a.data)/f'{corpus}.dialogues.jsonl'
        dl=[Dialogue.from_json(x) for x in open(src,encoding='utf-8') if x.strip()]
        for d in dl: proxy(d,tok)
        prox=out/f'_{corpus}.proxy.dialogues.jsonl'
        with open(prox,'w',encoding='utf-8') as f:
            for d in dl: f.write(d.to_json()+'\n')
        ds=DialogueWindowDataset([str(prox)],tok,R=6,window_s=(20.,40.),hop_s=10.,delays=(2,),seed=7,allow_unaligned=True,mono_cache_dir=a.mono_cache,min_text_tokens=8)
        cand=[]
        for i,(cid,t0,L) in enumerate(ds.items):
            eps,_=ds.window_episodes(ds.dlgs[cid],t0,L); eps=[e for e in eps if e.lane]
            if len({e.speaker for e in eps})<2: continue
            ov=overlap_seconds(eps); cand.append((i,eps,ov))
        rng=random.Random(f'{a.seed}:{corpus}'); rng.shuffle(cand)
        over=[x for x in cand if x[2]>=.20]; non=[x for x in cand if x[2]<.20]; chosen=over[:n//2]+non[:n-n//2]
        used={x[0] for x in chosen}; chosen += [x for x in cand if x[0] not in used][:n-len(chosen)]
        for j,(i,eps,ov) in enumerate(chosen):
            cid,t0,L=ds.items[i]; d=ds.dlgs[cid]; wav=np.asarray(crop_audio(ds.mono(cid),t0,t0+L),dtype=np.float32)
            res=transcribe(model,tok,wav,lang=d.lang,delay=2); hyp=res.text(tok,normalize=False)
            ref=' '.join(tok.decode([t for e in sorted(eps,key=lambda e:e.start) for t,_ in e.tokens],skip_special_tokens=True).split())
            if d.lang=='Korean': ru=list(re.sub(r'\s+','',score_ko(ref,True))); hu=list(re.sub(r'\s+','',score_ko(hyp,True))); unit='CER'
            else: ru=score_en(ref).split(); hu=score_en(hyp).split(); unit='WER'
            name=f'{corpus}-{j:02d}'; sf.write(out/'audio'/f'{name}.wav',wav,16000,subtype='PCM_16')
            emits=[dict(k=c.k,time_s=round(min(L,(c.k+1)*.08),3),pieces=c.pieces) for c in res.chunks if c.ids]
            svg(out/'plots'/f'{name}.svg',wav,eps,emits,L,f'checkpoint-35000 | {name} | {unit} {ed(ru,hu)/max(1,len(ru)):.1%}',tok)
            episodes=[dict(lane=e.lane,speaker=e.speaker,start=round(e.start,3),end=round(e.end,3),text=tok.decode([t for t,_ in e.tokens],skip_special_tokens=True).strip()) for e in eps]
            row=dict(name=name,corpus=corpus,conv=cid,t0=t0,duration_s=L,lang=d.lang,speakers=len({e.speaker for e in eps}),overlap_s=round(ov,3),reference=ref,hypothesis=hyp,unit=unit,error_rate=ed(ru,hu)/max(1,len(ru)),stats=res.stats(),episodes=episodes,emissions=emits,audio=f'audio/{name}.wav',plot=f'plots/{name}.svg')
            rows.append(row); print('DONE',name,cid,t0,'spk',row['speakers'],'overlap',ov,unit,row['error_rate'],flush=True)
    json.dump(dict(checkpoint=a.checkpoint,seed=a.seed,selection='per corpus: half overlap>=0.20s, half non-overlap; deterministic shuffle',rows=rows),open(out/'results.json','w'),ensure_ascii=False,indent=2)
    cards=[]
    for r in rows:
        eps=''.join(f'<li><b>lane {x["lane"]}</b> {x["start"]:.2f}–{x["end"]:.2f}s: {esc(x["text"])}</li>' for x in r['episodes'])
        cards.append(f'''<article><h2>{esc(r['name'])} · {esc(r['lang'])} · {r['speakers']} speakers · overlap {r['overlap_s']:.2f}s</h2><p><b>{r['unit']} {r['error_rate']:.1%}</b> · {r['duration_s']:.1f}s · RTF {r['stats']['realtime_factor']} · forced {r['stats']['forced']}</p><audio controls preload="metadata" src="{r['audio']}"></audio><img src="{r['plot']}" loading="lazy"><details><summary>참조 화자 구간</summary><ul>{eps}</ul></details><dl><dt>통합 참조</dt><dd>{esc(r['reference'])}</dd><dt>checkpoint-35000 출력</dt><dd>{esc(r['hypothesis'])}</dd></dl></article>''')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>checkpoint-35000 다화자 추론</title><style>body{{font:15px system-ui;max-width:1480px;margin:25px auto;padding:0 16px;background:#f6f8fa;color:#1f2328}}article{{background:white;border:1px solid #d0d7de;border-radius:12px;padding:18px;margin:20px 0}}audio,img{{width:100%}}img{{border:1px solid #d8dee4;margin-top:10px}}dt{{font-weight:700;margin-top:9px}}dd{{margin:4px 0;padding:9px;background:#f6f8fa;border-radius:6px}}</style><h1>checkpoint-35000 다화자 held-out 샘플</h1><p>이 모델은 lanes=0인 mono ASR이다. 화자별 전사가 아니라 모든 화자의 시간순 통합 전사를 평가한다. 샘플은 결과를 보기 전에 overlap/non-overlap으로 고정 선별했다.</p>{''.join(cards)}</html>'''
    (out/'index.html').write_text(page,encoding='utf-8'); print('OUTPUT',out,flush=True)
if __name__=='__main__': main()
