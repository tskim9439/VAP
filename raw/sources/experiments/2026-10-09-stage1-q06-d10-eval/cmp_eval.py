import glob, json, random, collections
R = "/soundai/users/tskim/VAPKT-data/results"
MODELS = {"new": "single-turn-vpaa-kspon-q06-s1m3-d10", "q06-s1": "single-turn-vpaa-kspon-q06-s1",
          "q17-s1m-r3": "single-turn-vpaa-kspon-q17-s1m-r3", "new-de": "single-turn-yodas-de-q06-s1m3-d10"}
def load(d):
    out = collections.defaultdict(dict)                      # (dataset, delta) -> id -> (errors, n_ref)
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f, encoding="utf-8"):
            p = json.loads(l); m = p["metrics"][p["primary_metric"]]
            out[(p["dataset"], str(p["delta"]))][p["id"]] = (m["errors"], m["n_ref"])
    return out
P = {k: load(v) for k, v in MODELS.items()}
def rate(x): return 100 * sum(e for e, _ in x) / max(1, sum(n for _, n in x))
rng = random.Random(0)
def boot(a, b, B=1000):
    ids = sorted(set(a) & set(b)); diffs = []
    for _ in range(B):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(rate([a[i] for i in s]) - rate([b[i] for i in s]))
    diffs.sort(); return diffs[int(0.025 * B)], diffs[int(0.975 * B)]
keys = sorted({k for m in P.values() for k in m}, key=lambda k: (k[0], k[1] != "final", int(k[1]) if k[1].isdigit() else 99))
for ds, dl in keys:
    row = [f"{ds:22s} δ{dl:5s}"]
    new = P["new"].get((ds, dl)) or P["new-de"].get((ds, dl))
    row.append(f"new {rate(new.values()):6.2f} (n={len(new)})" if new else "new     -")
    for base in ("q06-s1", "q17-s1m-r3"):
        b = P[base].get((ds, dl))
        if not b:
            row.append(f"{base} -"); continue
        s = f"{base} {rate(b.values()):6.2f}"
        if new:
            lo, hi = boot(new, b); s += f" Δ[{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}"
        row.append(s)
    print(" | ".join(row))
