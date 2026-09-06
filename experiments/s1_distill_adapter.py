#!/usr/bin/env python
"""Stage 1 — mono adapter 증류 초기화: Nemotron [56,0] 특징(12.5 Hz, (1,T,1024)) → Qwen audio tower 출력(qwen-aut-block8s, 13 Hz) 회귀. 라벨 불필요.
U0.5(`u05_distill_adapter.py`)와 같은 손실(cosine + 0.1·MSE/var)이지만 U0.5 파라미터를 읽지 않고 **LibriSpeech·KsponSpeech mono 스트림으로 새로** 만든다.
thinker 는 이 임베딩 공간의 벡터로 ASR 을 하도록 학습돼 있으므로, 출발점이 '무의미한 벡터' 에서 '거의 읽히는 벡터' 로 바뀐다.

사전: python experiments/s1_extract_features.py --encoder qwen-aut-block8s --manifest librispeech-100 --mode stream (kspon-100 도)
python experiments/s1_distill_adapter.py [--manifests librispeech-100,kspon-100] [--epochs 2] [--max-streams 20000]
결과: $MXC_CKPT_EXP_DIR/uslm/s1-adapter-distill/{adapter.pt, results.json}  → s1_train_mono.py --init-adapter …/adapter.pt
"""
import os, sys, json, time, argparse, random, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="nemotron-c0"); ap.add_argument("--target", default="qwen-aut-block8s")
ap.add_argument("--manifests", default="librispeech-100,kspon-100"); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max-streams", type=int, default=20000, help="언어별 상한(결정적 표본)"); ap.add_argument("--hidden", type=int, default=2048); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--val-frac", type=float, default=0.03); ap.add_argument("--gpu", default=None); a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        a.gpu = str(max([[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()], key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, torch.nn.functional as F
from vapasr.probe.data import FeatureIndex, resample_feats
from vapasr.uslm.model import Adapter
torch.manual_seed(0); random.seed(0); dev = "cuda"
FEAT = os.environ.get("MXC_DATA_FEATURE_CACHE_DIR", os.environ.get("DATA_FEATURE_CACHE_DIR", "/tmp"))
out = os.path.join(os.environ.get("MXC_CKPT_EXP_DIR", os.environ.get("CKPT_EXP_DIR", "/tmp")), "uslm", "s1-adapter-distill"); os.makedirs(out, exist_ok=True)
pairs = []
for m in a.manifests.split(","):
    S, T = FeatureIndex(FEAT, a.source, m), FeatureIndex(FEAT, a.target, m); ids = sorted(S.rows.keys() & T.rows.keys())
    ids = [i for i in ids if S.rows[i].get("mode", "stream") == "stream"]; random.Random(0).shuffle(ids); ids = ids[: a.max_streams]
    pairs += [(S.rows[i]["npy"], T.rows[i]["npy"], S.rows[i]["duration"], m) for i in ids]; print(f"  {m}: 공통 스트림 {len(ids)}", flush=True)
assert pairs, "공통 스트림 없음 — 타깃 캐시(qwen-aut-block8s) 확인"
src_hz, tgt_hz = 12.5, 13.0; random.shuffle(pairs); nval = max(1, int(len(pairs) * a.val_frac)); val, train = pairs[:nval], pairs[nval:]
print(f"pairs {len(pairs)} (train {len(train)}, val {len(val)}), {sum(p[2] for p in pairs)/3600:.1f} h | {a.source} {src_hz} Hz → {a.target} {tgt_hz} Hz", flush=True)
def load(s, t):
    x = np.asarray(np.load(s, mmap_mode="r")).astype(np.float32); y = np.asarray(np.load(t, mmap_mode="r")).astype(np.float32)   # (1,T,D)
    x = resample_feats(x, src_hz, tgt_hz); n = min(x.shape[1], y.shape[1]); return torch.from_numpy(x[0, :n]), torch.from_numpy(y[0, :n])
def batches(ps, bs):
    ps = list(ps); random.shuffle(ps)
    for i in range(0, len(ps), bs):
        xs, ys = zip(*[load(s, t) for s, t, _, _ in ps[i: i + bs]]); yield torch.cat(xs).to(dev), torch.cat(ys).to(dev)   # 프레임 단위 회귀라 이어 붙여도 됨
ad = Adapter(h=a.hidden).to(dev); opt = torch.optim.AdamW(ad.parameters(), a.lr, weight_decay=0.01)
@torch.no_grad()
def evaluate():
    ad.eval(); cos = mse = base = n = 0.0; tv = 0.0
    for x, y in batches(val, a.bs):
        p = ad(x); cos += F.cosine_similarity(p, y, dim=-1).mean().item(); mse += F.mse_loss(p, y).item(); base += F.cosine_similarity(x, y, dim=-1).mean().item(); tv += y.var().item(); n += 1
    ad.train(); return dict(cos=cos / n, mse=mse / n, cos_identity=base / n, tgt_var=tv / n)
hist = []; t0 = time.time(); step = 0
for ep in range(1, a.epochs + 1):
    for x, y in batches(train, a.bs):
        p = ad(x); loss = (1 - F.cosine_similarity(p, y, dim=-1)).mean() + 0.1 * F.mse_loss(p, y) / (y.var() + 1e-8)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
        if step % 200 == 0: print(f"  ep {ep} step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    ev = evaluate(); hist.append(dict(epoch=ep, step=step, **ev)); print(f"[val] ep {ep} cos {ev['cos']:.4f} (항등 {ev['cos_identity']:.4f}) mse/var {ev['mse']/max(ev['tgt_var'],1e-8):.4f}", flush=True)
torch.save(dict(adapter=ad.state_dict(), hidden=a.hidden, source=a.source, target=a.target, src_hz=src_hz, tgt_hz=tgt_hz, hist=hist), os.path.join(out, "adapter.pt"))
json.dump(dict(args=vars(a), pairs=len(pairs), hours=sum(p[2] for p in pairs) / 3600, val=hist, sec=time.time() - t0), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
print("→", os.path.join(out, "adapter.pt"))
