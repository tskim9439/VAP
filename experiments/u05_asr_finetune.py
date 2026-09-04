#!/usr/bin/env python
"""U0.5 — adapter(+증류 init) + thinker LoRA 오프라인 ASR 미세조정, val WER/CER (jiwer). 캐시 Nemotron 특징 사용(인코더 실행 없음).
python experiments/u05_asr_finetune.py [--init-adapter <adapter.pt>] [--steps 3000] [--bs 8] [--eval-utts 300]
"""
import os, sys, json, time, math, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # 단편화 완화 (공유 GPU)
import numpy as np, torch, jiwer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.uslm import UttFeatureDataset, collate_utts, normalize_text
from vapasr.uslm.model import Adapter, AdapterThinkerASR
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="otoSpeech,aihub-ts01-5"); ap.add_argument("--val", default="otoSpeech,aihub-ts01-5,aihub-vs02"); ap.add_argument("--init-adapter", default=None)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr-adapter", type=float, default=5e-4)
ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--eval-utts", type=int, default=300); ap.add_argument("--tag", default=""); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); dev = "cuda"
out = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm", f"u05-asr{('-' + a.tag) if a.tag else ''}"); os.makedirs(out, exist_ok=True)
from qwen_asr import Qwen3ASRModel
import torch.nn as nn
qm = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer
del root.thinker.audio_tower   # AuT 불필요 (adapter 가 대체)
adapter = Adapter()
if a.init_adapter:
    ck = torch.load(a.init_adapter, map_location="cpu"); adapter = Adapter(h=ck.get("hidden", 2048)); adapter.load_state_dict(ck["state"]); print("adapter init ←", a.init_adapter, flush=True)
model = AdapterThinkerASR(thinker, tok, adapter, lora_r=a.lora_r).to(dev); model.adapter.float()
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad); print(f"trainable params {n_tr/1e6:.1f}M", flush=True)
train_ds = UttFeatureDataset(a.train.split(","), split="train"); val_sets = {m: UttFeatureDataset([m], split="val", max_utts=a.eval_utts) for m in a.val.split(",")}
print(f"train utts {len(train_ds)} | val " + ", ".join(f"{k}:{len(v)}" for k, v in val_sets.items()), flush=True)
dl = torch.utils.data.DataLoader(train_ds, a.bs, shuffle=True, num_workers=4, collate_fn=collate_utts, drop_last=True)
params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("adapter")], "lr": a.lr_adapter}, {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("adapter")], "lr": a.lr}]
opt = torch.optim.AdamW(params, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 100) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

def evaluate(n_max=None):
    model.eval(); res = {}
    for name, ds in val_sets.items():
        vdl = torch.utils.data.DataLoader(ds, 8, collate_fn=collate_utts); refs, hyps = [], []; t0 = time.time()
        for b in vdl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.transcribe(b["feats"].to(dev), b["lens"], b["lang"])
            hyps += [x.split("<asr_text>")[-1] for x in h]; refs += b["text"]
            if n_max and len(refs) >= n_max: break
        lang = b["lang"][0]; R = [normalize_text(r, lang) for r in refs]; H = [normalize_text(x, lang) for x in hyps]
        res[name] = dict(**({"cer": jiwer.cer(R, H)} if lang == "Korean" else {"wer": jiwer.wer(R, H)}), n=len(R), sec=time.time() - t0, example=[refs[0][:40], hyps[0][:40]])
        print(f"  [val] {name}: {res[name]}", flush=True)
    model.train(); torch.cuda.empty_cache(); return res      # 평가 생성 후 캐시 반납 — 호스트 실측 25 GB 점유의 주범
step = 0; t0 = time.time(); hist = []; model.train()
while step < a.steps:
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(b["feats"].to(dev), b["lens"], b["lang"], b["text"])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); step += 1
        if step % 50 == 0: print(f"  step {step}/{a.steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(); hist.append(dict(step=step, **r))
            torch.save(dict(adapter=model.adapter.state_dict(), lora=model.thinker.state_dict() if a.lora_r == 0 else {k: v for k, v in model.thinker.state_dict().items() if "lora" in k}, step=step), os.path.join(out, "ckpt.pt"))
            json.dump(dict(args=vars(a), hist=hist, trainable_m=n_tr / 1e6, train_utts=len(train_ds)), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
        if step >= a.steps: break
print("saved", out)
