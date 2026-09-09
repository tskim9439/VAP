#!/usr/bin/env python
"""StreamingEncoder(cache-aware, 80 ms 단위) 가 오프라인 전체 인코딩(NemotronOnline.forward) 과 프레임 단위로 같은지 + 청크당 지연. 이어서 LiveSession 전사가 transcribe() 와 같은지."""
import os, sys, time, glob, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.hf.infer import load_model, load_audio, transcribe
from vapasr.hf.live import StreamingEncoder, LiveSession, SR
model, tok = load_model("/soundai/Model/VAPASR/hf-C2/final", device="cuda")
wav = load_audio(sys.argv[1] if len(sys.argv) > 1 else "/soundai/DB/raw/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac")[: SR * 12]
lang = sys.argv[2] if len(sys.argv) > 2 else "English"
K = int(round(len(wav) / SR / 0.08)); w = torch.from_numpy(wav).cuda()
with torch.inference_mode(): off = model.encode(w[None], torch.tensor([len(wav)], device="cuda"), torch.tensor([K], device="cuda"))[0, 0]   # (K, D)
se = StreamingEncoder(model.encoder); frames = []; ticks = []
for i in range(0, len(wav), 1280):
    t = time.time(); fr = se.feed(wav[i: i + 1280]); torch.cuda.synchronize(); ticks.append((time.time() - t) * 1000); frames += fr
frames += se.feed(np.zeros(0, np.float32), final=True); st = torch.stack(frames)
n = min(len(st), off.shape[0]); d = (st[:n] - off[:n]).abs(); print(f"K={K} offline frames={off.shape[0]} streaming frames={len(st)} max|diff| first10={float(d[:10].max()):.2e} all={float(d.max()):.2e} mean={float(d.mean()):.2e} | per-80ms enc tick p50={np.percentile(ticks,50):.1f} ms p99={np.percentile(ticks,99):.1f} ms", flush=True)
print("frame-wise max diff (first 12):", [f"{float(x):.1e}" for x in d.max(1).values[:12]])
ref = transcribe(model, tok, wav, lang=lang, delay=4); s = LiveSession(model, tok, lang=lang, delay=4); ev = []
for i in range(0, len(wav), 1280): ev += s.feed(wav[i: i + 1280])
ev += s.finish(); tk = [e.tick_ms for e in ev]
print("transcribe :", ref.text(tok)); print("live       :", s.text()); print(f"same={ref.text(tok)==s.text()} live tick p50={np.percentile(tk,50):.0f} ms p99={np.percentile(tk,99):.0f} ms max={max(tk):.0f} ms", flush=True)
