#!/usr/bin/env python
"""U0.5 — 표현 증류로 adapter 초기화: Nemotron [56,0] 특징(12.5 Hz) → Qwen audio tower의 thinker 입력 임베딩(13 Hz) 회귀. 캐시 위에서 학습, 라벨 불필요.
python experiments/u05_distill_adapter.py --target qwen-aut-block8s [--manifests otoSpeech,aihub-ts01-5,aihub-vs02] [--epochs 3]
결과: $CKPT_EXP_DIR/uslm/adapter-distill-<target>/{adapter.pt, results.json}
"""
import os, sys, json, time, argparse, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.probe.data import FeatureIndex, resample_feats
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="nemotron-c0"); ap.add_argument("--target", default="qwen-aut-block8s")
ap.add_argument("--manifests", default="otoSpeech,aihub-ts01-5,aihub-vs02"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--window-s", type=float, default=20.0); ap.add_argument("--hidden", type=int, default=2048); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-frac", type=float, default=0.1)
a = ap.parse_args(); torch.manual_seed(0); random.seed(0)
FEAT = os.environ.get("DATA_FEATURE_CACHE_DIR", "/data3/tskim/features"); dev = "cuda"
pairs = []
for m in a.manifests.split(","):
    S, T = FeatureIndex(FEAT, a.source, m), FeatureIndex(FEAT, a.target, m)
    for cid in S.rows.keys() & T.rows.keys():
        pairs.append((S.rows[cid]["npy"], T.rows[cid]["npy"], S.rows[cid]["duration"]))
assert pairs, "공통 대화 없음 — 타깃 캐시 확인"
src_hz, tgt_hz = FeatureIndex(FEAT, a.source, a.manifests.split(",")[0]).frame_hz or 12.5, 13.0
random.shuffle(pairs); nval = max(1, int(len(pairs) * a.val_frac)); val, train = pairs[:nval], pairs[nval:]
print(f"pairs {len(pairs)} (train {len(train)}, val {len(val)}), {sum(p[2] for p in pairs)/3600:.1f} h | {a.source} {src_hz} Hz → {a.target} {tgt_hz} Hz", flush=True)

def windows(pairs, hop):
    out = []
    for s, t, dur in pairs:
        for w in range(int((dur - a.window_s) // hop) + 1): out.append((s, t, w * hop))
    return out
def load(s, t, s0):
    S = np.load(s, mmap_mode="r"); T = np.load(t, mmap_mode="r")
    x = np.asarray(S[:, int(round(s0 * src_hz)): int(round((s0 + a.window_s) * src_hz))]).astype(np.float32)
    y = np.asarray(T[:, int(round(s0 * tgt_hz)): int(round((s0 + a.window_s) * tgt_hz))]).astype(np.float32)
    x = resample_feats(x, src_hz, tgt_hz); n = min(x.shape[1], y.shape[1]); return torch.from_numpy(x[:, :n]), torch.from_numpy(y[:, :n])

class Adapter(nn.Module):
    def __init__(self, d_in=1024, d_out=1024, h=2048):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))
    def forward(self, x): return self.net(x)
ad = Adapter(h=a.hidden).to(dev); opt = torch.optim.AdamW(ad.parameters(), a.lr, weight_decay=0.01)
tr, va = windows(train, 10.0), windows(val, 20.0); random.shuffle(tr)
def batches(ws, bs):
    for i in range(0, len(ws), bs):
        xs, ys = zip(*[load(*w) for w in ws[i:i + bs]]); n = min(x.shape[1] for x in xs)
        yield torch.stack([x[:, :n] for x in xs]).to(dev), torch.stack([y[:, :n] for y in ys]).to(dev)
def evaluate():
    ad.eval(); cos = mse = base = n = 0.0
    with torch.no_grad():
        for x, y in batches(va, a.bs):
            p = ad(x); cos += F.cosine_similarity(p, y, dim=-1).mean().item(); mse += F.mse_loss(p, y).item()
            base += F.cosine_similarity(x, y, dim=-1).mean().item(); n += 1     # 무학습 기준(항등)
    ad.train(); return dict(cos=cos / n, mse=mse / n, cos_identity=base / n, tgt_var=float(y.var().item()))
t0 = time.time(); hist = []; step = 0
for ep in range(a.epochs):
    for x, y in batches(tr, a.bs):
        p = ad(x); loss = (1 - F.cosine_similarity(p, y, dim=-1)).mean() + 0.1 * F.mse_loss(p, y) / (y.var() + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 200 == 0: print(f"  ep {ep} step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    ev = evaluate(); hist.append(dict(epoch=ep, **ev)); print(f"[val] ep {ep} cos {ev['cos']:.4f} (identity {ev['cos_identity']:.4f}) mse/var {ev['mse']/ev['tgt_var']:.3f}", flush=True)
    _o = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm", f"adapter-distill-{a.target}"); os.makedirs(_o, exist_ok=True)
    torch.save(dict(state=ad.state_dict(), hidden=a.hidden, source=a.source, target=a.target, src_hz=src_hz, tgt_hz=tgt_hz, epoch=ep), os.path.join(_o, f"adapter-ep{ep}.pt"))
out = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm", f"adapter-distill-{a.target}"); os.makedirs(out, exist_ok=True)
torch.save(dict(state=ad.state_dict(), hidden=a.hidden, source=a.source, target=a.target, src_hz=src_hz, tgt_hz=tgt_hz), os.path.join(out, "adapter.pt"))
json.dump(dict(source=a.source, target=a.target, pairs=len(pairs), hours=sum(p[2] for p in pairs) / 3600, val=hist, sec=time.time() - t0), open(os.path.join(out, "results.json"), "w"), indent=1)
print("saved", out)
