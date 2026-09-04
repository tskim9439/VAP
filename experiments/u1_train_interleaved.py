#!/usr/bin/env python
"""U1 — interleaved streaming ASR 학습 + 스트리밍 평가 (Nemotron [56,0] 캐시 특징 → adapter/merge → Qwen3-ASR thinker LoRA).
python experiments/u1_train_interleaved.py --init-ckpt /data4/tskim/VAPASR/experiments/uslm/u05-asr-distill-12k/ckpt.pt --steps 12000 --tag v0
평가: val 창(고정)을 chunk 단위 greedy 스트리밍 디코드 → 화자별 WER/CER, 토큰 지연(방출 시각 − 정렬 종료 시각) 분포, evidence 위반률, M 강제 비율.
"""
import os, sys, json, time, math, argparse, random, difflib
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, jiwer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.uslm.data import normalize_text, LANG
from vapasr.uslm.interleave_data import InterleavedWindowDataset, collate_windows, CHUNK_S
from vapasr.uslm.model import Adapter, InterleavedASR
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="otoSpeech,aihub-ts01-5"); ap.add_argument("--val", default="otoSpeech,aihub-ts01-5,aihub-vs02")
ap.add_argument("--init-ckpt", default=None, help="U0.5 ckpt.pt (adapter+lora) 또는 U1 ckpt"); ap.add_argument("--window-s", type=float, default=30.0)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--M", type=int, default=4); ap.add_argument("--eval-delay", type=int, default=2)
ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lr-adapter", type=float, default=1e-4)
ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--eval-windows", type=int, default=12)
ap.add_argument("--windows-per-conv", type=int, default=None); ap.add_argument("--log-every", type=int, default=50); ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-only", action="store_true")
a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); dev = "cuda"
out = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm", f"u1-interleaved{('-' + a.tag) if a.tag else ''}"); os.makedirs(out, exist_ok=True)
from qwen_asr import Qwen3ASRModel
import torch.nn as nn
qm = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer
del root.thinker.audio_tower
delays = tuple(int(x) for x in a.delays.split(","))
train_ds = InterleavedWindowDataset(a.train.split(","), tok, split="train", window_s=a.window_s, delays=delays, max_per_chunk=a.M, windows_per_conv=a.windows_per_conv, seed=a.seed)
val_ds = {m: InterleavedWindowDataset([m], tok, split="val", window_s=a.window_s, delays=(a.eval_delay,), max_per_chunk=a.M, windows_per_conv=4, max_windows=a.eval_windows, seed=1) for m in a.val.split(",")}
print(f"train windows {len(train_ds)} ({len(train_ds.convs)} convs) | val " + ", ".join(f"{k}:{len(v)}" for k, v in val_ds.items()), flush=True)
adapter = Adapter()
model = InterleavedASR(thinker, tok, adapter, train_ds.sp_ids, lora_r=a.lora_r).to(dev); model.adapter.float(); model.merge.float()
if a.init_ckpt:
    ck = torch.load(a.init_ckpt, map_location="cpu")
    if "special_rows" in ck or "merge" in ck: model.load_trainable_state(ck); print("U1 ckpt init ←", a.init_ckpt, flush=True)
    else: model.load_trainable_state(dict(adapter=ck["adapter"], lora=ck.get("lora"))); print("U0.5 init(adapter+lora) ←", a.init_ckpt, flush=True)
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) - model._embed().weight.numel() + len(model.special_rows) * model._embed().weight.shape[1]
print(f"trainable params {n_tr/1e6:.1f}M (+특수 토큰 행 {len(model.special_rows)})", flush=True)
dl = torch.utils.data.DataLoader(train_ds, a.bs, shuffle=True, num_workers=4, collate_fn=collate_windows, drop_last=True)
groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("adapter") or n.startswith("merge"))], "lr": a.lr_adapter},
          {"params": [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("adapter") or n.startswith("merge"))], "lr": a.lr}]
opt = torch.optim.AdamW(groups, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, a.steps)))))

def latency_stats(hyp, ref):
    """hyp [(k, id)], ref [(id, end_time)] → 일치 토큰의 지연(방출 (k+1)·80ms − 정렬 종료) 목록 (difflib 단조 정렬)."""
    h_ids = [t for _, t in hyp]; r_ids = [t for t, _ in ref]; lat = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h_ids, r_ids, autojunk=False).get_opcodes():
        if tag == "equal":
            for di in range(i2 - i1): lat.append((hyp[i1 + di][0] + 1) * CHUNK_S - ref[j1 + di][1])
    return lat

@torch.no_grad()
def evaluate():
    model.eval(); res = {}
    if a.lora_r > 0: model.thinker.merge_adapter()          # 스트리밍 디코드는 위치당 forward 1 회 → LoRA 분기 오버헤드 제거
    for name, ds in val_ds.items():
        lang = LANG[name]; R, H, lat, forced_tot, chunks_tot, n_tok = [], [], [], 0, 0, 0; t0 = time.time(); ex = None
        for (nm, cid, ws) in ds.items:
            f, streams, chunks, st = ds.window(nm, cid, ws, a.eval_delay)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emitted, forced = model.stream_decode(torch.from_numpy(f).to(dev), ds.prefix(lang, a.eval_delay), a.M)
            forced_tot += forced; chunks_tot += ds.K; n_tok += len(emitted)
            for s in (0, 1):
                hyp = [(k, t) for k, t, spk in emitted if spk == s]; ref = streams[s]
                if not ref and not hyp: continue
                R.append(normalize_text(tok.decode([t for t, _ in ref]), lang)); H.append(normalize_text(tok.decode([t for _, t in hyp]), lang)); lat += latency_stats(hyp, ref)
                if ex is None and ref: ex = (R[-1][:50], H[-1][:50])
        lat = np.array(lat) if lat else np.zeros(1); err = jiwer.cer(R, H) if lang == "Korean" else jiwer.wer(R, H)
        res[name] = {("cer" if lang == "Korean" else "wer"): err, "windows": len(ds.items), "segs": len(R), "matched": int(len(lat)), "lat_p50": float(np.median(lat)), "lat_p90": float(np.percentile(lat, 90)),
                     "lat_p99": float(np.percentile(lat, 99)), "viol": float((lat < 0).mean()), "viol_80ms": float((lat < -0.08).mean()), "forced_frac": forced_tot / max(1, chunks_tot), "tok_per_chunk": n_tok / max(1, chunks_tot), "sec": time.time() - t0, "example": ex}
        print(f"  [val] {name}: " + json.dumps(res[name], ensure_ascii=False), flush=True)
    if a.lora_r > 0: model.thinker.unmerge_adapter()
    model.train(); torch.cuda.empty_cache(); return res

hist = []
if a.eval_only:
    r = evaluate(); json.dump(dict(args=vars(a), eval=r), open(os.path.join(out, "eval.json"), "w"), indent=1, ensure_ascii=False); sys.exit(0)
step = 0; t0 = time.time(); model.train(); acc = {}
while step < a.steps:
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = model(b["feats"].to(dev), b["ids"].to(dev), b["is_audio"].to(dev), b["chunk_of"].to(dev), b["labels"].to(dev), b["mask"].to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); step += 1
        for k, v in parts.items(): acc[k] = acc.get(k, 0) + v
        if step % a.log_every == 0:
            print(f"  step {step}/{a.steps} loss {loss.item():.3f} next {acc['loss_next']/a.log_every:.3f} text {acc['loss_text']/a.log_every:.3f} L {b['ids'].shape[1]} {time.time()-t0:.0f}s", flush=True); acc = {}
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(); hist.append(dict(step=step, **r))
            torch.save(dict(**model.trainable_state(), step=step, args=vars(a)), os.path.join(out, "ckpt.pt"))
            json.dump(dict(args=vars(a), hist=hist, trainable_m=n_tr / 1e6, train_windows=len(train_ds)), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
        if step >= a.steps: break
print("saved", out)
