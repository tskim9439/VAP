"""r1/r2 (v0.2 labels) vs r3 (v0.3) vs r4 (v0.3.2) against gold v1, δ4: P/R/F1/latency per config + ASR error rate (base = E2 + tokens, no training)."""
import json, os, sys
GS = "/data4/tskim/semcommit/gold/v1/scores"; E = "/data4/tskim/semcommit/eval"
parts = ("ls-test", "gs-test", "ks-eval", "ks-long"); cfgs = sys.argv[1:] or ["bias=0", "bias=1", "bias=2", "theta=0.2"]
def gold(part):
    out = {}
    for f in (f"models-{part}.json", f"models-r3-{part}.json", f"models-r4-{part}.json", f"models-r5-{part}.json"):
        if os.path.exists(f"{GS}/{f}"): out.update(json.load(open(f"{GS}/{f}")))
    return out
def asr(run, part):
    for d in ("v0.2-r1", "v0.2-r2", "v0.2-gold", "v0.3-r3", "v0.3.2-r4", "v0.3.3-r5"):
        p = f"{E}/{d}/{run}-d4-{part}.json"
        if os.path.exists(p):
            c = json.load(open(p))["configs"]; x = (c.get("bias=0") or next(iter(c.values())))["overall"]["asr"]
            k = "cer_nospace" if "cer_nospace" in x else "wer"; return f"{'CER' if k != 'wer' else 'WER'} {100 * x[k]['rate']:.2f}"
    return "-"
res = {}
for part in parts:
    g = gold(part); print(f"== {part}  (gold COMMIT {next(iter(next(iter(g.values())).values()))['gold_commit'] if g else '?'})")
    for run in ("base", "r1", "r2", "r3", "r5"):
        k = f"{run}-d4-{part}"
        if k not in g and run != "base": print(f"  {run}: (none)"); continue
        row = [f"{run:4s} {asr(run, part):>10s}"]
        for c in cfgs:
            v = (g.get(k) or {}).get(c)
            if v: row.append(f"{c}: {v['P'] if v['P'] is not None else float('nan'):.2f}/{v['R']:.2f}/{(v['F1'] or 0):.2f} n{v['n_hyp']} lat{(v['latency_s'] or {}).get('p50') or float('nan'):.2f}")
            res.setdefault(part, {}).setdefault(run, {})[c] = v
        print("  " + " | ".join(row))
json.dump(res, open("/data4/tskim/semcommit/tools/compare_models.out.json", "w"), indent=1)
