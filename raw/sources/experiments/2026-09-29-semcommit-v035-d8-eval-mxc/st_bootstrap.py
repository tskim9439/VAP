"""single-turn-v1: 새 모델 vs E2 짝 bootstrap(같은 발화, 주 지표 errors/n_ref 합산 비율). LibriSpeech 는 화자 군집, Kspon 은 발화 단위(화자 정보 없음).
Δ = 새 모델 − E2 (%p), 95 % 구간(2,000 회), 상대 변화."""
import json, glob, re, sys
import numpy as np
R = "/soundai/users/tskim/VAPKT-data/results"
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{R}/{d}/predictions-rank*.jsonl")):
        for l in open(f):
            r = json.loads(l); m = r["metrics"][r["primary_metric"]]
            out[(r["dataset"], r["delta"], r["id"])] = (m["errors"], m["n_ref"])
    return out
a = load("single-turn-E2-final-v1"); b = load("single-turn-semcommit-v035-d8-v1")
rng = np.random.default_rng(7); B = 2000
cells = sorted({(k[0], k[1]) for k in a} & {(k[0], k[1]) for k in b})
print(f"{'set/δ':28s} {'n':>5s} {'E2':>7s} {'new':>7s} {'Δ%p':>7s} {'95% CI':>17s} {'rel':>7s}  cluster")
for ds, dl in cells:
    ids = sorted(k[2] for k in a if k[0] == ds and k[1] == dl and k in b)
    ea = np.array([a[(ds, dl, i)][0] for i in ids], float); eb = np.array([b[(ds, dl, i)][0] for i in ids], float)
    n = np.array([a[(ds, dl, i)][1] for i in ids], float); assert all(a[(ds, dl, i)][1] == b[(ds, dl, i)][1] for i in ids)
    spk = [re.match(r"(\d+)-\d+-\d+$", i.rsplit("/", 1)[-1]) for i in ids]
    if all(spk):
        keys = sorted({m.group(1) for m in spk}); ix = {k: j for j, k in enumerate(keys)}; g = np.array([ix[m.group(1)] for m in spk]); cl = f"speaker({len(keys)})"
    else:
        g = np.arange(len(ids)); cl = "utt"
    G = g.max() + 1; EA = np.bincount(g, ea, G); EB = np.bincount(g, eb, G); N = np.bincount(g, n, G)
    base = (eb.sum() - ea.sum()) / n.sum() * 100
    s = rng.integers(0, G, (B, G)); cnt = np.stack([np.bincount(r, minlength=G) for r in s])
    d = (cnt @ EB - cnt @ EA) / (cnt @ N) * 100; lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{ds + '/δ' + str(dl):28s} {len(ids):5d} {ea.sum() / n.sum() * 100:7.2f} {eb.sum() / n.sum() * 100:7.2f} {base:+7.2f} [{lo:+6.2f}, {hi:+6.2f}] {base / (ea.sum() / n.sum() * 100) * 100:+6.1f}%  {cl}")
