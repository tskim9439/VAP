"""q06-s2-ast vs Stage 1 (q06-s1m3-d10), same items, paired bootstrap 2,000 (LibriSpeech: speaker clusters, others: utterances).
Δ = s2-ast − Stage 1 (%p of summed errors / summed reference units)."""
import glob, json, re, collections
import numpy as np
R = "/soundai/users/tskim/VAPKT-data/results"
def load(d):
    out = {}
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l); m = r["metrics"][r["primary_metric"]]; out[(r["dataset"], str(r["delta"]), r["id"])] = (m["errors"], m["n_ref"])
    return out
rng = np.random.default_rng(7); B = 2000
order = {"4": 0, "6": 1, "8": 2, "10": 3, "final": 4}
print(f"{'set':26s} {'δ':6s} {'n':>5s} {'Stage1':>7s} {'s2-ast':>7s} {'Δ%p':>7s} {'95% CI':>17s}")
for old_d, new_d in (("single-turn-vpaa-kspon-q06-s1m3-d10", "single-turn-vpaa-kspon-q06-s2-ast"), ("single-turn-yodas-de-q06-s1m3-d10", "single-turn-yodas-de-q06-s2-ast"),
                     ("single-turn-librispeech-clean-q06-s1m3-d10", "single-turn-librispeech-clean-q06-s2-ast"), ("single-turn-librispeech-other-q06-s1m3-d10", "single-turn-librispeech-other-q06-s2-ast")):
    a, b = load(old_d), load(new_d)
    for ds, dl in sorted({k[:2] for k in b if k in a}, key=lambda x: (x[0], order[x[1]])):
        ids = sorted(k[2] for k in b if k[:2] == (ds, dl) and k in a)
        ea = np.array([a[(ds, dl, i)][0] for i in ids], float); eb = np.array([b[(ds, dl, i)][0] for i in ids], float); n = np.array([b[(ds, dl, i)][1] for i in ids], float)
        if ds.startswith("librispeech"):
            spk = [re.match(r"(\d+)-\d+-\d+$", i.rsplit("/", 1)[-1]).group(1) for i in ids]; keys = sorted(set(spk)); ix = {k: j for j, k in enumerate(keys)}; g = np.array([ix[s] for s in spk])
        else:
            g = np.arange(len(ids))
        G = g.max() + 1; EA, EB, N = np.bincount(g, ea, G), np.bincount(g, eb, G), np.bincount(g, n, G)
        s = rng.integers(0, G, (B, G)); cnt = np.stack([np.bincount(r, minlength=G) for r in s])
        d = (cnt @ EB - cnt @ EA) / (cnt @ N) * 100; lo, hi = np.percentile(d, [2.5, 97.5]); base = (eb.sum() - ea.sum()) / n.sum() * 100
        print(f"{ds:26s} δ{dl:5s} {len(ids):5d} {ea.sum() / n.sum() * 100:7.2f} {eb.sum() / n.sum() * 100:7.2f} {base:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]{' *' if lo > 0 or hi < 0 else ''}")
