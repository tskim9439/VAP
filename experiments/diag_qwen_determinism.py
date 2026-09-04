#!/usr/bin/env python
"""Qwen AuT(패치 마스크) fn 수준: (1) 같은 입력 2회 → 결정적인가, (2) prefix 불변성 fn(f[:, :, :L]) vs fn(f)[:, :n]."""
import sys, os, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.features import load_encoder
w = sorted(os.listdir("/data3/tskim/corpora/aihub/adult-vs02-wav"))[0]
x, sr = sf.read("/data3/tskim/corpora/aihub/adult-vs02-wav/" + w, dtype="float32", always_2d=True); x = torch.from_numpy(x.T).cuda()
for name in sys.argv[1:] or ["qwen-aut-causal"]:
    enc = load_encoder(name)
    with torch.inference_mode():
        f = enc.frontend(x); print(f"{name}: feats {tuple(f.shape)}", flush=True)
        a = enc._fn(f).float().cpu().numpy(); b = enc._fn(f).float().cpu().numpy(); dstd = a.reshape(-1, a.shape[-1]).std(0) + 1e-8
        r = (np.abs(a - b) / dstd).max(-1); print(f"  (1) 결정성: 동일 입력 2회 rel>1e-3 프레임 {int((r>1e-3).any(0).sum())}/{a.shape[1]} max {r.max():.2e}", flush=True)
        for L in (5000, 10000, 20000):     # 50 / 100 / 200 s (fbank 프레임)
            h = enc._fn(f[..., :L]).float().cpu().numpy(); n = h.shape[1] - 2; r = (np.abs(a[:, :n] - h[:, :n]) / dstd).max(-1)
            bad = np.flatnonzero((r > 1e-3).any(0)); print(f"  (2) prefix {L/100:.0f}s: rel>1e-3 {len(bad)}/{n} max {r.max():.2e} 첫 {round(bad[0]/12.5,1) if len(bad) else None}s", flush=True)
        # (3) 마스크 캐시 무효화 후 반복 — 캐시 키 문제인지
        h = enc._fn(f[..., :10000]).float().cpu().numpy(); h2 = enc._fn(f[..., :10000]).float().cpu().numpy(); r = (np.abs(h - h2) / dstd).max(-1)
        print(f"  (3) 100s 2회: rel>1e-3 {int((r>1e-3).any(0).sum())} max {r.max():.2e}", flush=True)
    del enc; torch.cuda.empty_cache()
