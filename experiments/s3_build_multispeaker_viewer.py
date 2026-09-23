#!/usr/bin/env python3
"""Build a self-contained-ish local HTML comparison from mono and D1b JSON."""
import html, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'raw/sources/experiments/2026-09-22-mono-vs-phase2-multispeaker'
OLD=ROOT/'raw/sources/experiments/2026-09-18-phase2-d1b-sample-inference'
mono=json.load(open(OUT/'mono.json'))
colors={1:'#0969da',2:'#bf8700',3:'#8250df',4:'#1a7f37',5:'#cf222e',6:'#57606a'}

def e(s): return html.escape(str(s))
def timeline(name,row,w):
    dur=float(w['L']); W=1200; x0=125; pw=1045; H=330
    q=[f'<svg class="timeline" data-name="{name}" viewBox="0 0 {W} {H}">', '<rect width="100%" height="100%" fill="#fff"/>']
    for sec in range(0,int(dur)+1,5):
        x=x0+pw*sec/dur; q += [f'<line x1="{x}" x2="{x}" y1="25" y2="290" stroke="#d8dee4"/>',f'<text x="{x}" y="315" text-anchor="middle">{sec}s</text>']
    rows=[('REF lane 1',55),('REF lane 2',90),('D1b lane 1',145),('D1b lane 2',180),('D1b lane 3',215),('mono emit',270)]
    for label,y in rows: q += [f'<text x="5" y="{y+5}">{label}</text>',f'<line x1="{x0}" x2="{x0+pw}" y1="{y}" y2="{y}" stroke="#eee"/>']
    for s in w['ref']['episodes']:
        y=55+(s['lane']-1)*35; x=x0+pw*s['start']/dur; ww=max(2,pw*(s['end']-s['start'])/dur)
        q.append(f'<rect x="{x}" y="{y-11}" width="{ww}" height="22" rx="4" fill="{colors.get(s["lane"],"#777")}" opacity=".75"><title>{e(s["text"])}</title></rect>')
    for s in w['hyp']['segments']:
        lane=int(s['lane']);
        if lane>3: continue
        start=float(s.get('start') or 0); end=float(s.get('end') or start+.08); y=145+(lane-1)*35; x=x0+pw*start/dur; ww=max(2,pw*(end-start)/dur)
        q.append(f'<rect x="{x}" y="{y-11}" width="{ww}" height="22" rx="4" fill="{colors.get(lane,"#777")}" opacity=".75"><title>{e(s.get("text",""))}</title></rect>')
    for em in row['emissions']:
        x=x0+pw*float(em['time_s'])/dur; q.append(f'<line x1="{x}" x2="{x}" y1="258" y2="282" stroke="#cf222e" stroke-width="2"><title>{e("".join(em["pieces"]))}</title></line>')
    q.append(f'<line class="playhead" data-name="{name}" x1="{x0}" x2="{x0}" y1="25" y2="290" stroke="#111" stroke-width="2"/>')
    q.append('</svg>'); return ''.join(q)

cards=[]
for row in mono['rows']:
    name=row['name']; w=json.load(open(OLD/'windows'/f'{name}.json')); m=w['metrics']
    ref=''.join(f'<li><b>lane {s["lane"]}</b> {s["start"]:.2f}–{s["end"]:.2f}s: {e(s["text"])}</li>' for s in w['ref']['episodes'])
    hyp=''.join(f'<li><b>lane {s["lane"]}</b> {float(s.get("start") or 0):.2f}–{float(s.get("end") or 0):.2f}s: {e(s.get("text",""))}</li>' for s in w['hyp']['segments'] if s.get('text'))
    audio=f'../2026-09-18-phase2-d1b-sample-inference/audio/{name}.ogg'
    cards.append(f'''<article><h2>{e(name)} · {e(row['lang'])} · {row['duration_s']:.1f}s</h2>
<audio id="audio-{name}" controls preload="metadata" src="{audio}"></audio>{timeline(name,row,w)}
<div class="grid"><section><h3>참조 화자 구간</h3><ul>{ref}</ul></section><section><h3>D1b Phase 2 출력</h3>
<p><b>pooled {e(m['unit'].upper())} {m['pooled_rate']:.1%}</b> · lane {e(m['unit'].upper())} {m['lane_rate']:.1%} · ONSET P/R {m['onset']['hit']/max(1,m['onset']['hyp']):.1%}/{m['onset']['hit']/max(1,m['onset']['ref']):.1%}</p><ul>{hyp}</ul></section></div>
<section class="mono"><h3>34.5k mono 모델 출력</h3><p><b>{e(row['unit'])} {row['error_rate']:.1%}</b> · 화자 구분 없음 · RTF {row['stats']['realtime_factor']}</p><p>{e(row['hypothesis'])}</p></section></article>''')

page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>다화자 샘플 비교</title><style>
body{{font:15px system-ui;max-width:1280px;margin:25px auto;padding:0 15px;background:#f6f8fa;color:#1f2328}}article{{background:white;border:1px solid #d0d7de;border-radius:12px;padding:20px;margin:20px 0}}audio{{width:100%}}.timeline{{width:100%;height:auto;margin-top:10px;border:1px solid #d8dee4}}.timeline text{{font:12px system-ui}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}li{{margin:5px 0}}.mono{{background:#fff8c5;padding:10px 14px;border-radius:8px}}code{{background:#eee;padding:2px 4px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>동일 다화자 오디오: mono 34.5k vs Phase 2 D1b</h1><p><code>checkpoint-34500</code>은 lanes=0이라 화자를 분리하지 않는다. <code>p2-D1b-restart/final</code>만 lane·ONSET·EOT를 출력한다. 검정 세로선은 재생 위치이며 막대에 마우스를 올리면 해당 구간 전사가 보인다.</p>{''.join(cards)}
<p><a href="../2026-09-18-phase2-d1b-sample-inference/viewer.html">기존 D1b 상세 동기화 뷰어 열기</a></p>
<script>document.querySelectorAll('audio').forEach(a=>{{let n=a.id.slice(6), lines=document.querySelectorAll('.playhead[data-name="'+n+'"]'), svg=document.querySelector('svg[data-name="'+n+'"]'); a.addEventListener('timeupdate',()=>{{let x=125+1045*a.currentTime/Math.max(.001,a.duration);lines.forEach(l=>{{l.setAttribute('x1',x);l.setAttribute('x2',x)}})}})}})</script></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
(OUT/'README.md').write_text('# 다화자 동일 샘플 비교\n\n- mono: checkpoint-34500 (lanes=0)\n- Phase 2: p2-D1b-restart/final (lanes=6)\n- 샘플: held-out NIKL 2020, CHiME-6\n- 통합 뷰어: `index.html`\n',encoding='utf-8')
print(OUT/'index.html')
