"""q17-s1(multi-lookahead 학습) 평가 context 비교: [56,0] δ2/3/4/6 + final vs [56,3] δ2/3/4/6 + final, 같은 총지연 짝 bootstrap."""
import json, glob, numpy as np, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.aa_wer import aa_summary
R = "/soundai/users/tskim/VAPKT-data/results"
MAN = {json.loads(l)["id"]: json.loads(l) for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-vpaa-kspon-v1/manifest.jsonl")}
def load(*dirs):
    out = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{R}/{d}/predictions*.jsonl")):
            for l in open(f):
                r = json.loads(l); m = r["metrics"][r["primary_metric"]]; out[(r["dataset"], str(r["delta"]), r["id"])] = (m["errors"], m["n_ref"], r)
    return out
A = load("single-turn-vpaa-kspon-q17-s1", "single-turn-vpaa-kspon-q17-s1-d3"); B = load("single-turn-vpaa-kspon-q17-s1-rc3")
DS = ["voxpopuli-aa-test", "kspon-eval_clean", "kspon-eval_other"]
def rate(m, ds, dl):
    v = [x for (d, de, i), x in m.items() if d == ds and de == dl]
    if not v: return "–"
    s = f"{100*sum(x[0] for x in v)/sum(x[1] for x in v):.2f}"
    return s + (f"/{100*aa_summary([x[2] for x in v])['wer_duration_weighted']:.2f}" if ds.startswith("vox") else "")
print("총지연(평균): [56,0]+δd = 80d ms, [56,3]+δd = 80d+120 ms\n")
for ds in DS:
    print(ds); print("  [56,0] " + " | ".join(f"{dl}: {rate(A, ds, dl)}" for dl in ("2", "3", "4", "6", "final")))
    print("  [56,3] " + " | ".join(f"{dl}: {rate(B, ds, dl)}" for dl in ("2", "3", "4", "6", "final")))
rng = np.random.default_rng(7); Bn = 2000
print("\n짝 bootstrap Δ = [56,3] − [56,0] (%p, 95 % CI)")
for ds in DS:
    for db, da in (("2", "3"), ("2", "4"), ("3", "4"), ("3", "6"), ("4", "6"), ("6", "6"), ("final", "final")):
        ids = sorted(i for (d, de, i) in B if d == ds and de == db and (d, da, i) in A)
        if not ids: continue
        key = (lambda i: MAN[i]["session"]) if ds.startswith("vox") else (lambda i: i)
        gs = sorted({key(i) for i in ids}); gi = {g: k for k, g in enumerate(gs)}; E = np.zeros((len(gs), 3))
        for i in ids:
            k = gi[key(i)]; E[k, 0] += A[(ds, da, i)][0]; E[k, 1] += B[(ds, db, i)][0]; E[k, 2] += A[(ds, da, i)][1]
        pa, pb = E[:, 0].sum() / E[:, 2].sum(), E[:, 1].sum() / E[:, 2].sum()
        idx = rng.integers(0, len(gs), (Bn, len(gs))); S = E[idx].sum(1); diff = (S[:, 1] - S[:, 0]) / S[:, 2] * 100
        lo, hi = np.percentile(diff, [2.5, 97.5])
        lat = lambda r, d: "end" if d == "final" else f"{80*int(d)+(120 if r else 0)}"
        print(f"{ds:18s} [56,3]δ{db}({lat(1,db)}ms) vs [56,0]δ{da}({lat(0,da)}ms): {100*pb:.2f} vs {100*pa:.2f} Δ {100*(pb-pa):+.2f} [{lo:+.2f},{hi:+.2f}] {'유의' if lo>0 or hi<0 else 'n.s.'}")
