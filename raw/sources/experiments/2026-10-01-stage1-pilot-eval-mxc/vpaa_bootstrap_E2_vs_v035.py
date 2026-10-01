import json, glob, numpy as np
R = "/soundai/users/tskim/VAPKT-data/results"
man = {json.loads(l)["id"]: json.loads(l)["session"] for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-vpaa-v1/manifest.jsonl")}
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{R}/{d}/predictions-rank*.jsonl")):
        for l in open(f):
            r = json.loads(l); m = r["metrics"][r["primary_metric"]]; out[(r["delta"], r["id"])] = (m["errors"], m["n_ref"])
    return out
a = load("single-turn-vpaa-E2-final"); b = load("single-turn-vpaa-semcommit-v035-d8")
rng = np.random.default_rng(7); B = 2000
for d in (2, 4, 6):
    ids = sorted(i for (dd, i) in a if dd == d and (d, i) in b)
    sess = sorted({man[i] for i in ids}); si = {s: k for k, s in enumerate(sess)}
    E = np.zeros((len(sess), 3))
    for i in ids:
        k = si[man[i]]; E[k, 0] += a[(d, i)][0]; E[k, 1] += b[(d, i)][0]; E[k, 2] += a[(d, i)][1]
    pa, pb = E[:, 0].sum() / E[:, 2].sum(), E[:, 1].sum() / E[:, 2].sum()
    idx = rng.integers(0, len(sess), (B, len(sess))); S = E[idx].sum(1); diff = (S[:, 1] - S[:, 0]) / S[:, 2] * 100
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"delta{d}: E2 {pa*100:.2f}  v035-d8 {pb*100:.2f}  diff {100*(pb-pa):+.2f}%p [{lo:+.2f}, {hi:+.2f}]  rel {100*(pb/pa-1):+.1f}%  sessions={len(sess)} utt={len(ids)}  {'SIG' if lo > 0 or hi < 0 else 'n.s.'}")
