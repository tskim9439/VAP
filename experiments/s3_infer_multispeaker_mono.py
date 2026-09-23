#!/usr/bin/env python3
"""Run a Phase-1/mono checkpoint on fixed multi-speaker WAVs for comparison."""
import argparse, json, os, re
from pathlib import Path

import torch
from vapasr.hf.infer import load_audio, load_model, transcribe
from vapasr.data.textnorm import score_en, score_ko


def distance(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        q = [i] + [0] * len(b)
        for j, y in enumerate(b, 1): q[j] = min(p[j] + 1, q[j-1] + 1, p[j-1] + (x != y))
        p = q
    return p[-1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--sample',action='append',required=True,help='name|language|wav|window-json')
    a=ap.parse_args(); model,tok=load_model(a.checkpoint,device='cuda',dtype=torch.bfloat16); rows=[]
    for spec in a.sample:
        name,lang,wav_path,json_path=spec.split('|',3); meta=json.load(open(json_path)); wav=load_audio(wav_path)
        res=transcribe(model,tok,wav,lang=lang,delay=2); hyp=res.text(tok,normalize=False)
        ref=' '.join(e['text'] for e in sorted(meta['ref']['episodes'],key=lambda e:(e['start'],e['lane'])))
        if lang=='Korean':
            r=list(re.sub(r'\s+','',score_ko(ref,True))); h=list(re.sub(r'\s+','',score_ko(hyp,True))); unit='CER'
        else: r=score_en(ref).split(); h=score_en(hyp).split(); unit='WER'
        rows.append(dict(name=name,lang=lang,audio=wav_path,duration_s=round(len(wav)/16000,3),reference=ref,
                         hypothesis=hyp,unit=unit,error_rate=distance(r,h)/max(1,len(r)),stats=res.stats(),
                         emissions=[dict(k=c.k,time_s=round(min(len(wav)/16000,(c.k+1)*.08),3),pieces=c.pieces)
                                    for c in res.chunks if c.ids]))
        print('DONE',name,unit,rows[-1]['error_rate'],flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(dict(checkpoint=a.checkpoint,rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
