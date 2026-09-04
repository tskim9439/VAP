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

def overlap_mask(d, b):
    """(B, L) — 각 위치가 속한 chunk 에서 **두 화자가 동시에 말하는가**. joint chunk token(평균) 손상 판정용."""
    import torch as _t
    B, L = b["ids"].shape; out = _t.zeros(B, L, dtype=_t.bool)
    for i, m in enumerate(b["meta"]):
        ac = d.convs[(m["manifest"], m["conv"])]; t0 = m["t0"]
        act = [[False] * d.K, [False] * d.K]
        for u in ac.utts:
            k0 = max(0, int((u["start"] - t0) / CHUNK_S)); k1 = min(d.K, int((u["end"] - t0) / CHUNK_S) + 1)
            for k in range(k0, k1): act[u["speaker"]][k] = True
        cur = -1
        for j in range(L):
            if b["is_audio"][i, j]: cur = int(b["chunk_of"][i, j])
            if 0 <= cur < d.K and act[0][cur] and act[1][cur]: out[i, j] = True
    return out

@torch.no_grad()
def diag(name, d):
    """teacher forcing: 텍스트를 내야 하는 위치에서 P(NEXT_AUDIO) vs P(gold). 겹침/비겹침으로 분리."""
    b = collate_windows([d[i] for i in range(min(2, len(d)))])
    E = model.build(b["feats"].to(dev), b["ids"].to(dev), b["is_audio"].to(dev), b["chunk_of"].to(dev), b["audio_spk"].to(dev))
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
    ov = overlap_mask(d, b)[:, 1:].to(dev)
    print(f"  [{name}] 텍스트 위치       : {stat(is_txt)}")
    print(f"  [{name}]   ├ 단독 발화 구간: {stat(is_txt & ~ov)}")
    print(f"  [{name}]   └ 겹침 구간    : {stat(is_txt & ov)}")
    print(f"  [{name}] NEXT 위치        : {stat(is_na)}")

@torch.no_grad()
def sweep(name, d, bias):
    """화자별 WER/CER 와 **화자 무시(pooled)** WER/CER 를 함께 낸다.
    둘의 격차 = 화자 배정 오류의 몫. joint chunk token(두 화자 평균)이 병목인지 판정한다."""
    lang = LANG[name]; R, H, Rp, Hp, n_tok, forced = [], [], [], [], 0, 0
    for (nm, cid, ws) in d.items:
        f, streams, chunks, st = d.window(nm, cid, ws, a.delay)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            em, fo = model.stream_decode(torch.from_numpy(f).to(dev), d.prefix(lang, a.delay), a.M, next_bias=bias)
        n_tok += len(em); forced += fo
        for s in (0, 1):
            hyp = [t for k, t, sp in em if sp == s]; ref = [t for t, _ in streams[s]]
            if not ref and not hyp: continue
            R.append(normalize_text(tok.decode(ref), lang)); H.append(normalize_text(tok.decode(hyp), lang))
        pooled_ref = [t for t, _ in sorted(streams[0] + streams[1], key=lambda x: x[1])]
        Rp.append(normalize_text(tok.decode(pooled_ref), lang)); Hp.append(normalize_text(tok.decode([t for _, t, _ in em]), lang))
    E = (lambda r, h: jiwer.cer(r, h) if lang == "Korean" else jiwer.wer(r, h))
    return dict(bias=bias, err=round(E(R, H), 4), err_pooled=round(E(Rp, Hp), 4),
                tok_per_chunk=round(n_tok / max(1, d.K * len(d.items)), 4), forced_frac=round(forced / max(1, d.K * len(d.items)), 4))

for name, d in ds.items():
    print(f"== {name} ({len(d.items)} 창)"); diag(name, d)
    model.thinker.merge_adapter()
    for bias in [float(x) for x in a.bias.split(",")]:
        print("   ", sweep(name, d, bias), flush=True)
    model.thinker.unmerge_adapter()
