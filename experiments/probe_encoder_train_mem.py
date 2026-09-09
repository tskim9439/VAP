#!/usr/bin/env python
"""인코더 해동 학습(E1)의 GPU 메모리·스텝 시간 점검: C2 final + --train-encoder 로 KO/EN 배치 한 스텝 fwd/bwd(grad ckpt on, 인코더 fp32) → peak memory. 긴 항목으로 보수적으로 잰다."""
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--init", default="/soundai/Model/VAPASR/hf-C2/final"); ap.add_argument("--bs-ko", default="12,24,48,96"); ap.add_argument("--bs-en", default="6,12,24"); a = ap.parse_args()
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.uslm.mono_data import MonoStreamDataset, collate_streams
tok = load_tokenizer(a.init); model = VapAsrForStreamingASR.from_pretrained(a.init); model.encoder.set_trainable(True); model.config.encoder_trainable = True
model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None
model.cuda().train(); n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"trainable {n_tr/1e6:.0f}M", flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
def longest(name, n):
    ds = MonoStreamDataset([name], tok, mode="stream", delays=(2, 4), max_per_chunk=0, seed=0, max_items=4000, online=True)
    idx = sorted(range(len(ds)), key=lambda i: -ds.items[i]["K"])[:n] if hasattr(ds, "items") else list(range(n))
    return collate_streams([ds[i] for i in idx]), ds
for name, sizes, nw in (("kspon-full", a.bs_ko, 0.15), ("librispeech-960", a.bs_en, 0.3)):
    for bs in [int(x) for x in sizes.split(",")]:
        batch, ds = longest(name, bs); x = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items() if k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask")}
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16): out = model(**x, next_weight=nw)
            fwd = torch.cuda.max_memory_allocated() / 2**30; out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
            print(f"{name} bs={bs} Kmax={int(x['K'].max())} ({int(x['K'].max())*0.08:.0f}s) loss={float(out.loss):.3f} step {time.time()-t:.1f}s fwd-peak {fwd:.1f} GB peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"{name} bs={bs} Kmax={int(x['K'].max())}: OOM/{type(e).__name__} ({torch.cuda.max_memory_allocated()/2**30:.1f} GB 시점) {str(e)[:80]}", flush=True); opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); break
