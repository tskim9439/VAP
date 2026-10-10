"""LibriSpeech test-clean/other: q06-s1m3-d10 vs earlier 0.6B streaming models, paired speaker-cluster bootstrap (2,000).
Δ = new − old (%p, summed errors / summed reference words), 95 % interval, relative change."""
import json, glob, re
import numpy as np
R = "/soundai/users/tskim/VAPKT-data/results"
def load(dirs):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{R}/{d}/predictions*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if not r["dataset"].startswith("librispeech"): continue
                m = r["metrics"][r.get("primary_metric", "wer")]
                out[(r["dataset"], str(r["delta"]), r["id"])] = (m["errors"], m["n_ref"])
    return out
new = load(["single-turn-librispeech-clean-q06-s1m3-d10", "single-turn-librispeech-other-q06-s1m3-d10"])
olds = {"35000": load(["single-turn-35000-d2-d4-v1", "single-turn-35000-d6-v1"]), "E2": load(["single-turn-E2-final-v1"]),
        "v035-d8": load(["single-turn-semcommit-v035-d8-v1"]), "qwen3-asr-0.6b-offline": load(["single-turn-qwen3-asr-0.6b-offline"]),
        "qwen3-asr-1.7b-offline": load(["single-turn-qwen3-asr-1.7b-offline"])}
rng = np.random.default_rng(7); B = 2000
print("== new model WER (%)")
for ds, dl in sorted({(k[0], k[1]) for k in new}):
    v = [x for k, x in new.items() if k[0] == ds and k[1] == dl]
    print(f"{ds:24s} δ{dl:6s} n={len(v):5d} WER={sum(e for e, _ in v) / sum(n for _, n in v) * 100:6.2f}")
print("\n== paired bootstrap, Δ = q06-s1m3-d10 − old")
print(f"{'old':24s} {'set':24s} {'δnew/δold':10s} {'n':>5s} {'old':>7s} {'new':>7s} {'Δ%p':>7s} {'95% CI':>17s} {'rel':>7s}")
for name, a in olds.items():
    for ds in ("librispeech-test-clean", "librispeech-test-other"):
        pairs = [("final" if d == "offline" else d, d) for d in sorted({k[1] for k in a if k[0] == ds})]
        pairs += [("4", d) for d in ("6", "8") if name in ("35000", "v035-d8") and any(k[1] == d for k in a)]   # lookahead-matched: new δ4+rc3 = 560 ms
        for dl_new, dl_old in pairs:
            ids = sorted(k[2] for k in a if k[0] == ds and k[1] == dl_old and (ds, dl_new, k[2]) in new)
            if not ids: continue
            ea = np.array([a[(ds, dl_old, i)][0] for i in ids], float); eb = np.array([new[(ds, dl_new, i)][0] for i in ids], float)
            n = np.array([new[(ds, dl_new, i)][1] for i in ids], float)
            assert all(a[(ds, dl_old, i)][1] == new[(ds, dl_new, i)][1] for i in ids), "reference length mismatch"
            spk = [re.match(r"(\d+)-\d+-\d+$", i.rsplit("/", 1)[-1]).group(1) for i in ids]
            keys = sorted(set(spk)); ix = {k: j for j, k in enumerate(keys)}; g = np.array([ix[s] for s in spk]); G = len(keys)
            EA, EB, N = np.bincount(g, ea, G), np.bincount(g, eb, G), np.bincount(g, n, G)
            base = (eb.sum() - ea.sum()) / n.sum() * 100; old = ea.sum() / n.sum() * 100
            s = rng.integers(0, G, (B, G)); cnt = np.stack([np.bincount(r, minlength=G) for r in s])
            d = (cnt @ EB - cnt @ EA) / (cnt @ N) * 100; lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"{name:24s} {ds:24s} {dl_new + '/' + dl_old:10s} {len(ids):5d} {old:7.2f} {eb.sum() / n.sum() * 100:7.2f} {base:+7.2f} [{lo:+6.2f}, {hi:+6.2f}] {base / old * 100:+6.1f}%{' *' if lo > 0 or hi < 0 else ''}")
