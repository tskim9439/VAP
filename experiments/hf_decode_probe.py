#!/usr/bin/env python
"""부분 학습 ckpt 의 free-running 디코드 폭주 확인: 청크당 상한 64(옛 기본) vs 8(새 기본) 로 dev-clean 스트림을 디코드해 토큰 수·시간을 비교.
  CUDA_VISIBLE_DEVICES=2 python experiments/hf_decode_probe.py --ckpt /soundai/Model/VAPASR/s1-C2/ckpt-last.pt --n 2"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=2); ap.add_argument("--caps", default="64,8"); a = ap.parse_args()
import torch
from vapasr.hf import VapAsrForStreamingASR
from vapasr.uslm.mono_data import MonoStreamDataset
QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"); t0 = time.time()
m, tok = VapAsrForStreamingASR.from_legacy(a.ckpt, QWEN); m = m.to("cuda").eval(); print(f"load {time.time()-t0:.0f}s", flush=True)
ds = MonoStreamDataset(["librispeech-dev"], tok, mode="stream", subsets=["dev-clean"], delays=(2,), seed=1, max_items=a.n, online=True)
for cap in [int(c) for c in a.caps.split(",")]:
    for i in range(len(ds)):
        f, ref, _, st, it = ds.stream(i, 2); w = torch.from_numpy(f).cuda(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): e, fc, tk, rd = m.stream_decode(w, ds.prefix(it["lang"], 2), 0, next_bias=0.0, runaway_cap=cap, max_total_per_chunk=(1e9 if cap >= 64 else 6.0))
        dt = time.time() - t; hyp = tok.decode([x for _, x in e])
        print(f"cap {cap} · 스트림 {i} K={it['K']} ref {len(ref)} tok → hyp {len(e)} tok · forced {fc} · {dt:.0f}s · tick p50 {sorted(tk)[len(tk)//2]:.0f}ms · '{hyp[:100]}'", flush=True)
