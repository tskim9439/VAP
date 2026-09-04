#!/usr/bin/env python
"""U1 진단 — 방출 결정 지점에서 모델이 무엇을 생각하는지 본다.
teacher forcing 으로 정답 시퀀스를 넣고, "텍스트를 내야 하는 위치"에서
  P(<NEXT_AUDIO>) vs P(정답 텍스트 토큰) 및 정답의 순위를 잰다.
이어서 <NEXT_AUDIO> 로짓에 페널티(RNN-T blank penalty 와 같은 역할)를 걸어 스트리밍 디코드를 sweep 한다.
python experiments/u1_diag_emission.py --ckpt .../u1-interleaved-v0/ckpt.pt --bias 0,1,2,3,4,6
"""
import os, sys, json, argparse, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, jiwer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.uslm.data import normalize_text, LANG
from vapasr.uslm.interleave_data import InterleavedWindowDataset, collate_windows, CHUNK_S
from vapasr.uslm.model import Adapter, InterleavedASR
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--manifests", default="otoSpeech,aihub-ts01-5")
ap.add_argument("--windows", type=int, default=4); ap.add_argument("--bias", default="0,1,2,3,4,6"); ap.add_argument("--delay", type=int, default=2); ap.add_argument("--M", type=int, default=4)
a = ap.parse_args(); dev = "cuda"
from qwen_asr import Qwen3ASRModel
import torch.nn as nn
qm = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer
del root.thinker.audio_tower
ds = {m: InterleavedWindowDataset([m], tok, split="val", delays=(a.delay,), max_per_chunk=a.M, windows_per_conv=4, max_windows=a.windows, seed=1) for m in a.manifests.split(",")}
any_ds = next(iter(ds.values()))
model = InterleavedASR(thinker, tok, Adapter(), any_ds.sp_ids, lora_r=16).to(dev); model.adapter.float(); model.merge.float()
ck = torch.load(a.ckpt, map_location="cpu"); model.load_trainable_state(ck); model.eval()
print(f"ckpt step {ck.get('step')} ← {a.ckpt}", flush=True)
NA = model.next_audio

@torch.no_grad()
def diag(name, d):
    """teacher forcing: 텍스트를 내야 하는 위치에서 P(NEXT_AUDIO) vs P(gold)."""
    b = collate_windows([d[i] for i in range(min(2, len(d)))])
    E = model.build(b["feats"].to(dev), b["ids"].to(dev), b["is_audio"].to(dev), b["chunk_of"].to(dev))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model.thinker(inputs_embeds=E, attention_mask=b["mask"].to(dev)).logits[:, :-1].float()
    tgt = b["labels"][:, 1:].to(dev); lp = logits.log_softmax(-1)
    is_txt = (tgt != -100) & (tgt != NA); is_na = tgt == NA
    def stat(m):
        if not m.any(): return None
        g = lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0][m]; na = lp[..., NA][m]
        rank = (lp[m] > g[:, None]).sum(-1)
        return dict(n=int(m.sum()), p_gold=float(g.exp().mean()), p_next=float(na.exp().mean()),
                    gold_top1=float((rank == 0).float().mean()), gold_top5=float((rank < 5).float().mean()), gold_rank_p50=float(rank.float().median()))
    print(f"  [{name}] 텍스트 위치: {stat(is_txt)}")
    print(f"  [{name}] NEXT 위치 : {stat(is_na)}")

@torch.no_grad()
def sweep(name, d, bias):
    lang = LANG[name]; R, H, lat, n_tok, forced = [], [], [], 0, 0
    for (nm, cid, ws) in d.items:
        f, streams, chunks, st = d.window(nm, cid, ws, a.delay)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            em, fo = model.stream_decode(torch.from_numpy(f).to(dev), d.prefix(lang, a.delay), a.M, next_bias=bias)
        n_tok += len(em); forced += fo
        for s in (0, 1):
            hyp = [t for k, t, sp in em if sp == s]; ref = [t for t, _ in streams[s]]
            if not ref and not hyp: continue
            R.append(normalize_text(tok.decode(ref), lang)); H.append(normalize_text(tok.decode(hyp), lang))
    err = jiwer.cer(R, H) if lang == "Korean" else jiwer.wer(R, H)
    return dict(bias=bias, err=round(err, 4), tok_per_chunk=round(n_tok / max(1, d.K * len(d.items)), 4), forced_frac=round(forced / max(1, d.K * len(d.items)), 4))

for name, d in ds.items():
    print(f"== {name} ({len(d.items)} 창)"); diag(name, d)
    model.thinker.merge_adapter()
    for bias in [float(x) for x in a.bias.split(",")]:
        print("   ", sweep(name, d, bias), flush=True)
    model.thinker.unmerge_adapter()
