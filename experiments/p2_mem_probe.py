#!/usr/bin/env python
"""Phase 2 동적 배치 예산(--max-tokens) 실측 — 학습과 같은 모델 설정(E2 init, Phase 2 토큰, grad ckpt, Liger, 인코더 동결, AdamW 상태 포함)으로
가장 긴 창들을 예산별로 묶어 forward+backward+step 을 돌리고 GPU 최대 메모리·step 시간·토큰 처리량을 잰다. OOM 이 나면 거기서 멈춘다. GPU 1 장.
  python experiments/p2_mem_probe.py --init /soundai/Model/VAPASR/hf-E2/final --data <phase2 dir> --budgets 8000,12000,16000,20000,24000,28000 --gpu 1
결과: <out> 에 json (budget → peak_alloc_gb, peak_reserved_gb, sec/step, tokens/s, bs). 예산 선택 기준: 최대 메모리 143 GB 의 80 % ≈ 115 GB 아래에서 tokens/s 최대."""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--init", required=True); ap.add_argument("--data", required=True); ap.add_argument("--corpora", default="aihub134-1,ami,notsofar,icsi")
ap.add_argument("--budgets", default="8000,12000,16000,20000,24000,28000,32000"); ap.add_argument("--max-bs", type=int, default=48); ap.add_argument("--shape", default="long,short", help="long: 최장 창 소수 · short: 최단 창 다수(같은 예산)")
ap.add_argument("--mono-cache", default=None); ap.add_argument("--gpu", default=None); ap.add_argument("--out", default=None); ap.add_argument("--reps", type=int, default=2); ap.add_argument("--no-liger", action="store_true"); ap.add_argument("--train-encoder", action="store_true")
a = ap.parse_args()
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch, numpy as np
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.data.dialogue_dataset import DialogueWindowDataset, collate_dialogue
def log(*s): print(*s, flush=True)

t0 = time.time(); model = VapAsrForStreamingASR.from_pretrained(a.init); tok = load_tokenizer(a.init); cfg = model.config
if cfg.lanes == 0: model.add_phase2_tokens(tok, R=6)
else: model.add_phase2_tokens(tok, R=cfg.lanes)
model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
if not a.no_liger:
    try:
        from vapasr.hf.liger import apply_liger_to_thinker; log(f"liger: {apply_liger_to_thinker(model)}")
    except ImportError as e: log(f"liger 미적용({e})")
model.encoder.eval()
for p_ in model.encoder.parameters(): p_.requires_grad_(bool(a.train_encoder))
if a.train_encoder: model.encoder.train()
model.cuda(); params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=1e-6, weight_decay=0.01)
log(f"모델 준비 {time.time()-t0:.0f}s · 학습 파라미터 {sum(p.numel() for p in params)/1e6:.1f} M · GPU {torch.cuda.get_device_name()} {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB")

sets = {}
for c in a.corpora.split(","):
    p = os.path.join(a.data, f"{c}.refined.dialogues.jsonl")
    if os.path.exists(p): sets[c] = DialogueWindowDataset([p], tok, R=6, mono_cache_dir=a.mono_cache); log(f"  {c}: 창 {len(sets[c])}")
pool = [(int(e), c, i) for c, ds in sets.items() for i, e in enumerate(ds.est_lens())]; pool.sort()
log(f"창 {len(pool)} · 추정 길이 p50 {pool[len(pool)//2][0]} p90 {pool[int(len(pool)*0.9)][0]} max {pool[-1][0]}")

def batch_for(budget, shape):
    if shape == "long": cand = pool[-2000:]; est_max = cand[-1][0]; n = max(1, min(a.max_bs, budget // est_max)); pick = cand[-n:]
    else: cand = pool[:2000]; est_max = cand[-1][0]; n = max(1, min(a.max_bs, budget // est_max)); pick = cand[-n:]
    items = [sets[c][i] for _, c, i in pick]; return collate_dialogue(items), n, est_max

def step(x):
    x = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
    xin = {k: x[k] for k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "labels_alt", "activity", "activity_mask")}; xin["soft_w"] = x["soft_w_full"]
    with torch.autocast("cuda", dtype=torch.bfloat16): out = model(**xin, next_weight=0.3)
    out.loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); return float(out.loss)

res = {}; stop = False
for shape in a.shape.split(","):
    for b in [int(x) for x in a.budgets.split(",")]:
        if stop: break
        x, n, est_max = batch_for(b, shape); L = int(x["ids"].shape[1]); tokens = n * L
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        try:
            step(x); torch.cuda.synchronize(); t = time.time()
            for _ in range(a.reps): loss = step(x)
            torch.cuda.synchronize(); dt = (time.time() - t) / a.reps
            r = dict(shape=shape, budget=b, bs=n, L=L, est_max=est_max, tokens=tokens, sec=round(dt, 2), tokens_per_s=round(tokens / dt), peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 1), loss=round(loss, 3))
        except torch.cuda.OutOfMemoryError:
            r = dict(shape=shape, budget=b, bs=n, L=L, est_max=est_max, tokens=tokens, oom=True); stop = shape == "long"
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        res[f"{shape}:{b}"] = r; log(json.dumps(r))
    stop = False
if a.out: json.dump(res, open(a.out, "w"), indent=1); log("→", a.out)
