#!/usr/bin/env python
"""HF 래퍼(vapasr.hf.VapAsrForStreamingASR) 와 기존 MonoInterleavedASR 의 수치 패리티 검사 (2026-09-08).
  python experiments/hf_parity.py --ckpt /soundai/Model/VAPASR/s1-C/ckpt-last.pt --out /soundai/users/tskim/VAPKT-data/ckpt/hf-parity
검사: 파라미터 동일 → 같은 배치 손실 동일 → save_pretrained/from_pretrained 왕복 후 손실 동일 → stream_decode 토큰열 동일."""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=3); ap.add_argument("--liger", action="store_true"); ap.add_argument("--bench", type=int, default=0, help="N>0: forward+backward N 회 처리량·메모리(Liger 전/후)")
a = ap.parse_args()
import torch, torch.nn as nn
dev = "cuda"; QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")
from vapasr.uslm.mono_data import MonoStreamDataset, collate_streams
from vapasr.uslm.mono_model import MonoInterleavedASR
from vapasr.uslm.model import Adapter
from vapasr.features.online import NemotronOnline
from vapasr.hf import VapAsrForStreamingASR

t0 = time.time()
# ── legacy
from qwen_asr import Qwen3ASRModel
qm = Qwen3ASRModel.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer; del root.thinker.audio_tower
ds = MonoStreamDataset(["librispeech-dev"], tok, mode="stream", subsets=["dev-clean"], delays=(2,), seed=1, max_items=a.n, online=True)
enc = NemotronOnline().set_trainable(False)
legacy = MonoInterleavedASR(thinker, tok, Adapter(), ds.sp_ids, lora_r=0, full_ft=True, encoder=enc).to(dev); legacy.adapter.float()
st = torch.load(a.ckpt, map_location="cpu"); st = st["model"] if "model" in st and "adapter" not in st else st; legacy.load_trainable_state(st); legacy.eval()
print(f"legacy 로드 {time.time()-t0:.0f}s", flush=True)
# ── HF (같은 ckpt 변환; 인코더 객체는 공유)
hf, tok2 = VapAsrForStreamingASR.from_legacy(a.ckpt, QWEN, load_encoder=False); hf = hf.to(dev); hf.encoder = legacy.encoder; hf.eval()
assert hf.config.sp_ids == ds.sp_ids, (hf.config.sp_ids, ds.sp_ids)
print(f"hf 변환 {time.time()-t0:.0f}s · thinker params {sum(p.numel() for p in hf.thinker.parameters())/1e6:.1f}M · adapter {sum(p.numel() for p in hf.adapter.parameters())/1e6:.2f}M", flush=True)
# 1) 파라미터 동일
ls, hs = legacy.thinker.state_dict(), hf.thinker.state_dict(); assert set(ls) == set(hs), (len(ls), len(hs), list(set(ls) ^ set(hs))[:5])
md = max((ls[k].float() - hs[k].float()).abs().max().item() for k in ls); ad = max((legacy.adapter.state_dict()[k].float() - hf.adapter.state_dict()[k].float()).abs().max().item() for k in legacy.adapter.state_dict())
print(f"1) 파라미터 max|Δ| thinker {md:.2e} adapter {ad:.2e}"); assert md == 0 and ad == 0
# 2) 같은 배치 손실
random.seed(0); torch.manual_seed(0); b = collate_streams([ds[i] for i in range(len(ds))]); x = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
def run_legacy():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return legacy(x["wav"], x["ids"], x["is_audio"], x["chunk_of"], x["labels"], x["mask"], next_weight=0.3, wav_len=x["wav_len"], K=x["K"])
def run_hf(m):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): return m(wav=x["wav"], wav_len=x["wav_len"], K=x["K"], ids=x["ids"], is_audio=x["is_audio"], chunk_of=x["chunk_of"], labels=x["labels"], mask=x["mask"], next_weight=0.3)
l0, p0 = run_legacy(); o = run_hf(hf)
print(f"2) loss legacy {l0.item():.6f} hf {o.loss.item():.6f} | next {p0['loss_next']:.4f}/{o.loss_next:.4f} text {p0['loss_text']:.4f}/{o.loss_text:.4f} top1 {p0['top1_text']:.4f}/{o.top1_text:.4f} (B={x['ids'].shape[0]}, L={x['ids'].shape[1]})")
assert abs(l0.item() - o.loss.item()) < 1e-3 * max(1, abs(l0.item())), "손실 불일치"
# 3) 저장/재로드 왕복
os.makedirs(a.out, exist_ok=True); t1 = time.time(); hf.save_pretrained(a.out, safe_serialization=True); tok2.save_pretrained(a.out)
sz = {f: os.path.getsize(os.path.join(a.out, f)) / 1e9 for f in os.listdir(a.out) if f.endswith(".safetensors")}
print(f"3) save_pretrained {time.time()-t1:.0f}s → {a.out} {json.dumps({k: round(v, 2) for k, v in sz.items()})} GB · files {sorted(os.listdir(a.out))}", flush=True)
t1 = time.time(); hf2 = VapAsrForStreamingASR.from_pretrained(a.out, load_encoder=False).to(dev); hf2.encoder = legacy.encoder; hf2.eval()
o2 = run_hf(hf2); print(f"   from_pretrained {time.time()-t1:.0f}s loss {o2.loss.item():.6f} (Δ {abs(o2.loss.item()-o.loss.item()):.2e}) · tied: {hf2.thinker.lm_head.weight.data_ptr() == hf2.get_input_embeddings().weight.data_ptr()}")
assert abs(o2.loss.item() - o.loss.item()) < 1e-5
from transformers import AutoTokenizer; tok3 = AutoTokenizer.from_pretrained(a.out); assert tok3.convert_tokens_to_ids("<NEXT_AUDIO>") == hf2.config.sp_ids["<NEXT_AUDIO>"], "tokenizer 특수 토큰 id 불일치"
# 4) stream_decode 동일
f, ref, _, stt, it = ds.stream(0, 2); w = torch.from_numpy(f).to(dev); pre = ds.prefix(it["lang"], 2)
with torch.autocast("cuda", dtype=torch.bfloat16): e1, f1, t1s, r1 = legacy.stream_decode(w, pre, 0, next_bias=0.0)
with torch.autocast("cuda", dtype=torch.bfloat16): e2, f2, t2s, r2 = hf2.stream_decode(w, pre, 0, next_bias=0.0)
same = e1 == e2; print(f"4) stream_decode 동일 {same} · tokens {len(e1)}/{len(e2)} forced {f1}/{f2} rounds {r1}/{r2} tick p50 {sorted(t1s)[len(t1s)//2]:.1f}/{sorted(t2s)[len(t2s)//2]:.1f} ms")
print("   hyp:", tok.decode([t for _, t in e2])[:120]); assert same
def bench(m, label, n):
    """학습 step 모사: autocast bf16 forward + backward (grad ckpt 켬). → step 시간·최대 메모리"""
    if n <= 0: return
    m.train(); m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for i in range(n + 2):
        if i == 2: torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): out = m(wav=x["wav"], wav_len=x["wav_len"], K=x["K"], ids=x["ids"], is_audio=x["is_audio"], chunk_of=x["chunk_of"], labels=x["labels"], mask=x["mask"], next_weight=0.3)
        out.loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = (time.time() - t) / n; print(f"   bench[{label}] {dt*1000:.0f} ms/step · peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB (B={x['ids'].shape[0]}, L={x['ids'].shape[1]})", flush=True); m.eval()
if a.liger:
    del legacy, hf; torch.cuda.empty_cache(); bench(hf2, "no-liger", a.bench)
    from vapasr.hf.liger import apply_liger_to_thinker
    n = apply_liger_to_thinker(hf2); o3 = run_hf(hf2); print(f"5) liger 적용 {n} · loss {o3.loss.item():.6f} (Δ {abs(o3.loss.item()-o.loss.item()):.2e})")
    with torch.autocast("cuda", dtype=torch.bfloat16): e3, *_ = hf2.stream_decode(w, pre, 0, next_bias=0.0)
    d = sum(1 for p_, q_ in zip(e2, e3) if p_ != q_) + abs(len(e2) - len(e3)); print(f"   liger stream_decode 동일 {e3 == e2} (다른 토큰 {d}/{len(e2)})")
    bench(hf2, "liger", a.bench)
print("PARITY OK")
