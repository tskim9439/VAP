#!/usr/bin/env python3
"""Render a deterministic six-sample qualitative report for an HF VAP-ASR checkpoint.

The output directory is self contained: WAV files, timeline PNGs, results.json,
summary.md, and an index.html with audio controls.  Selection is deterministic
and duration based, so it cannot silently cherry-pick low-error examples.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vapasr.data.textnorm import score_en, score_ko
from vapasr.hf.infer import load_model, transcribe
from vapasr.uslm.mono_data import MonoStreamDataset


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return substitutions, deletions, insertions for one optimal alignment."""
    n, m = len(ref), len(hyp)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1): d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                a = d[i - 1][j - 1]; sub = (a[0] + 1, a[1] + 1, a[2], a[3])
                a = d[i - 1][j]; delete = (a[0] + 1, a[1], a[2] + 1, a[3])
                a = d[i][j - 1]; insert = (a[0] + 1, a[1], a[2], a[3] + 1)
                d[i][j] = min(sub, delete, insert, key=lambda x: (x[0], x[3], x[2], x[1]))
    return d[n][m][1:]


def metric(ref: str, hyp: str, lang: str) -> dict:
    if lang == "Korean":
        r = list(re.sub(r"\s+", "", score_ko(ref, True)))
        h = list(re.sub(r"\s+", "", score_ko(hyp, True)))
        name = "CER"
    else:
        r = score_en(ref).split(); h = score_en(hyp).split(); name = "WER"
    s, d, i = edit_counts(r, h)
    return {"name": name, "value": (s + d + i) / max(1, len(r)),
            "S": s, "D": d, "I": i, "N": len(r)}


def choose(ds: MonoStreamDataset, targets: list[float]) -> list[int]:
    """Pick unique items nearest fixed target durations; IDs break ties."""
    available = set(range(len(ds))); out = []
    for target in targets:
        i = min(available, key=lambda j: (abs(ds.items[j]["duration_s"] - target), ds.items[j]["id"]))
        available.remove(i); out.append(i)
    return out


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _heat(v: float) -> str:
    """Small dependency-free inferno-like colour ramp."""
    v = min(1., max(0., v))
    stops = ((0., (8, 5, 30)), (.35, (105, 18, 110)), (.7, (220, 68, 55)), (1., (252, 235, 110)))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            q = (v-a)/(b-a); c = tuple(round(x+(y-x)*q) for x, y in zip(ca, cb)); return "#%02x%02x%02x" % c
    return "#fceb6e"


def render_plot(path: Path, wav: np.ndarray, ref_tokens, result, tok, title: str) -> None:
    """Write standalone SVG: waveform, log-STFT heatmap, reference/emission ticks."""
    dur = len(wav) / 16000; W, H = 1500, 760; x0, x1 = 82, 1475; pw = x1-x0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{W/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="17">{_esc(title)}</text>']
    # waveform envelope, 2 points per horizontal pixel
    bins = min(pw, max(1, len(wav)//2)); edges = np.linspace(0, len(wav), bins+1).astype(int)
    ymin, ymax = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        z = wav[a:max(a+1,b)]; ymin.append(float(z.min())); ymax.append(float(z.max()))
    scale = max(1e-4, max(max(map(abs, ymin)), max(map(abs, ymax))))
    for j, (lo, hi) in enumerate(zip(ymin, ymax)):
        x = x0 + j*pw/max(1,bins-1); ya = 115-hi/scale*68; yb = 115-lo/scale*68
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ya:.1f}" y2="{yb:.1f}" stroke="#315a7d" stroke-width="1"/>')
    parts += ['<text x="8" y="120" font-family="sans-serif" font-size="13">wave</text>',
              f'<line x1="{x0}" x2="{x1}" y1="115" y2="115" stroke="#bbb"/>']
    # compact STFT rendered as SVG cells (no plotting dependency)
    nfft, hop, cols, bands = 512, 256, 260, 72
    starts = np.arange(0, max(1, len(wav)-nfft+1), hop)
    if not len(starts): starts = np.array([0])
    spec = np.empty((bands, len(starts)), dtype=np.float32); win = np.hanning(nfft)
    for j, s in enumerate(starts):
        frame = np.zeros(nfft); z=wav[s:s+nfft]; frame[:len(z)] = z
        p = np.log1p(np.abs(np.fft.rfft(frame*win)))
        spec[:,j] = np.asarray([p[a:b].mean() for a,b in zip(np.linspace(0,len(p),bands+1).astype(int)[:-1], np.linspace(0,len(p),bands+1).astype(int)[1:])])
    if spec.shape[1] > cols:
        e=np.linspace(0,spec.shape[1],cols+1).astype(int); spec=np.stack([spec[:,a:max(a+1,b)].mean(1) for a,b in zip(e[:-1],e[1:])],1)
    lo, hi = np.percentile(spec, [8, 99.5]); spec=np.clip((spec-lo)/max(1e-6,hi-lo),0,1)
    sy0, sh = 215, 270; cw=pw/spec.shape[1]; ch=sh/bands
    for j in range(spec.shape[1]):
        for k in range(bands):
            parts.append(f'<rect x="{x0+j*cw:.2f}" y="{sy0+(bands-1-k)*ch:.2f}" width="{cw+.15:.2f}" height="{ch+.15:.2f}" fill="{_heat(float(spec[k,j]))}"/>')
    parts += [f'<text x="8" y="{sy0+sh/2}" font-family="sans-serif" font-size="13">0–8 kHz</text>',
              f'<rect x="{x0}" y="{sy0}" width="{pw}" height="{sh}" fill="none" stroke="#777"/>']
    rt = np.asarray([float(e) for _, e in ref_tokens]); ht, hp = [], []
    for c in result.chunks:
        for piece in c.pieces:
            ht.append(min(dur, (c.k + 1) * .08)); hp.append(piece)
    yr, yh = 555, 620
    parts += [f'<text x="15" y="{yr+5}" fill="#238636" font-family="sans-serif">ref evidence</text>',
              f'<text x="15" y="{yh+5}" fill="#cf222e" font-family="sans-serif">hyp emission</text>']
    for sec in range(int(dur)+1):
        x=x0+pw*sec/max(.1,dur); parts += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="520" y2="680" stroke="#ddd"/>', f'<text x="{x:.1f}" y="705" text-anchor="middle" font-family="sans-serif" font-size="11">{sec}s</text>']
    for sec in rt:
        x=x0+pw*min(dur,sec)/max(.1,dur); parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{yr-14}" y2="{yr+14}" stroke="#238636" stroke-width="2"/>')
    for j,(sec,piece) in enumerate(zip(ht,hp)):
        x=x0+pw*sec/max(.1,dur); parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{yh-14}" y2="{yh+14}" stroke="#cf222e" stroke-width="2"/>')
        if j < 36:
            p=piece.replace("Ġ","_").replace("▁","_")[:12]
            parts.append(f'<text x="{x-2:.1f}" y="{yh+25}" transform="rotate(55 {x-2:.1f} {yh+25})" font-family="sans-serif" font-size="9">{_esc(p)}</text>')
    parts += [f'<line x1="{x0}" x2="{x1}" y1="{yr}" y2="{yr}" stroke="#238636" opacity=".35"/>',
              f'<line x1="{x0}" x2="{x1}" y1="{yh}" y2="{yh}" stroke="#cf222e" opacity=".35"/>', '</svg>']
    path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="checkpoint-34500 (checkpoint-33000 unavailable)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.output); (out / "audio").mkdir(parents=True, exist_ok=True); (out / "plots").mkdir(exist_ok=True)
    model, tok = load_model(a.checkpoint, device=a.device, dtype=torch.bfloat16)
    specs = [
        ("dev-clean", "librispeech-dev", "stream", ["dev-clean"], [22., 34.]),
        ("dev-other", "librispeech-dev", "stream", ["dev-other"], [22., 34.]),
        ("kspon-dev", "kspon-dev", "utt", ["dev"], [4., 8.]),
    ]
    rows = []
    for split, manifest, mode, subsets, targets in specs:
        ds = MonoStreamDataset([manifest], tok, mode=mode, subsets=subsets, delays=(2,), seed=0,
                               online=True, qc=True)
        for idx in choose(ds, targets):
            wav, ref_toks, _, _, item = ds.stream(idx, 2)
            result = transcribe(model, tok, wav, lang=item["lang"], delay=2)
            ref_raw = re.sub(r"\s+", " ", tok.decode([x for x, _ in ref_toks], skip_special_tokens=True)).strip()
            hyp_raw = result.text(tok, normalize=False)
            met = metric(ref_raw, hyp_raw, item["lang"])
            stem = f"{split}-{item['id']}".replace("/", "_")
            wav_name = f"audio/{stem}.wav"; png_name = f"plots/{stem}.svg"
            sf.write(out / wav_name, wav, 16000, subtype="PCM_16")
            render_plot(out / png_name, wav, ref_toks, result, tok,
                        f"{a.label} | {split} | {item['id']} | {met['name']} {met['value']:.1%}")
            rows.append({"split": split, "id": item["id"], "lang": item["lang"],
                         "duration_s": round(len(wav) / 16000, 3), "reference": ref_raw,
                         "hypothesis": hyp_raw, "metric": met, "stats": result.stats(),
                         "audio": wav_name, "plot": png_name,
                         "emissions": [{"chunk": c.k, "time_s": round(min(len(wav)/16000, (c.k + 1)*.08), 3),
                                        "pieces": c.pieces, "forced": c.forced, "tick_ms": round(c.tick_ms, 2)}
                                       for c in result.chunks if c.ids]})
            print(f"DONE {split} {item['id']} {met['name']}={met['value']:.4f}", flush=True)

    payload = {"checkpoint": a.checkpoint, "label": a.label, "selection": "nearest durations, IDs as tie-breaker",
               "delay_chunks": 2, "chunk_s": .08, "samples": rows}
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    vals = {}
    for r in rows: vals.setdefault(r["split"], []).append(r["metric"]["value"])
    md = [f"# {a.label} 정성 추론 샘플", "", f"- 실제 체크포인트: `{a.checkpoint}`",
          "- 선택: 세트별 고정 목표 길이에 가장 가까운 2개(ID로 동률 해소), 결과 기반 선별 없음",
          "- 스트리밍 설정: 80 ms chunk, delay=2 (설계 지연 160 ms), greedy, next_bias=0", "",
          "| split | id | duration | metric | reference | hypothesis |", "|---|---|---:|---:|---|---|"]
    for r in rows:
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
        md.append(f"| {r['split']} | `{r['id']}` | {r['duration_s']:.1f}s | {r['metric']['name']} {r['metric']['value']:.1%} | {esc(r['reference'])} | {esc(r['hypothesis'])} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cards = []
    for r in rows:
        m = r["metric"]; st = r["stats"]
        cards.append(f'''<article><h2>{html.escape(r["split"])} · {html.escape(r["id"])}</h2>
<p><b>{m["name"]} {m["value"]:.1%}</b> (S={m["S"]}, D={m["D"]}, I={m["I"]}, N={m["N"]}) · {r["duration_s"]:.1f}s · RTF {st["realtime_factor"]} · tick p50/p99 {st["tick_p50_ms"]:.1f}/{st["tick_p99_ms"]:.1f}ms</p>
<audio controls preload="metadata" src="{html.escape(r["audio"])}"></audio>
<dl><dt>정답</dt><dd>{html.escape(r["reference"])}</dd><dt>추론</dt><dd>{html.escape(r["hypothesis"])}</dd></dl>
<a href="{html.escape(r["plot"])}"><img src="{html.escape(r["plot"])}" loading="lazy"></a></article>''')
    page = f'''<!doctype html><meta charset="utf-8"><title>{html.escape(a.label)} samples</title>
<style>body{{font:15px system-ui;max-width:1450px;margin:2rem auto;padding:0 1rem;background:#f6f8fa;color:#1f2328}}article{{background:white;border:1px solid #d0d7de;border-radius:10px;padding:18px;margin:18px 0}}img{{width:100%;height:auto}}audio{{width:100%}}dt{{font-weight:700;margin-top:9px}}dd{{margin-left:0;padding:8px;background:#f6f8fa;border-radius:6px}}code{{background:#eee;padding:2px 4px}}</style>
<h1>{html.escape(a.label)} 정성 추론</h1><p>정확한 checkpoint-33000은 retention으로 삭제되어 가장 가까운 보존 스냅샷 checkpoint-34500을 사용했다. 각 세트의 샘플은 결과를 보기 전에 길이 기준으로 결정했다.</p>{''.join(cards)}'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print("OUTPUT", out, flush=True)


if __name__ == "__main__": main()
