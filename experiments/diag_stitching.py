#!/usr/bin/env python
"""Encoder.encode 세그먼트 이어붙이기 정확성: 짧은 seg 로 나눈 결과 vs 한 번에(seg 를 파일보다 크게) 결과."""
import sys, os, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.features import load_encoder
w = sorted(os.listdir("/data3/tskim/corpora/aihub/adult-vs02-wav"))[0]
x, sr = sf.read("/data3/tskim/corpora/aihub/adult-vs02-wav/" + w, dtype="float32", always_2d=True); x = x.T
for name in sys.argv[1:]:
    enc = load_encoder(name); seg0, ctx0 = enc.seg_s, enc.ctx_s
    enc.seg_s = 1e6; full = enc.encode(x, dtype=np.float32)
    enc.seg_s = 100.0; enc.ctx_s = ctx0; segd = enc.encode(x, dtype=np.float32)          # 100 s 세그먼트 (경계 3개)
    n = min(full.shape[1], segd.shape[1]); dstd = full.reshape(-1, full.shape[-1]).std(0) + 1e-8      # 차원별 std 로 정규화
    r = (np.abs(full[:, :n] - segd[:, :n]) / dstd).max(-1)
    bad = np.flatnonzero((r > 1e-3).any(0))
    flips = (full[:, :n].astype(np.float16) != segd[:, :n].astype(np.float16)).mean()
    big = np.abs(full).max() / (np.abs(full).std() + 1e-8)
    print(f"{name} (ctx {ctx0}s): frames {full.shape[1]}/{segd.shape[1]} | fp32 rel(dim-std)>1e-3 {len(bad)} | max {r.max():.2e} | 위치 {[round(b/enc.frame_hz,1) for b in bad[:5]]} | fp16 flip 비율 {flips:.4f} | max|v|/std {big:.0f}", flush=True)
    del enc; torch.cuda.empty_cache()
