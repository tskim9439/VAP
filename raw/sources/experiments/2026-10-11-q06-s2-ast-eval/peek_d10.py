import glob, json, collections
R = "/soundai/users/tskim/VAPKT-data/results"
def load(d):
    out = {}
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l); out[(r["dataset"], str(r["delta"]), r["id"])] = r
    return out
a, b = load("single-turn-vpaa-kspon-q06-s1m3-d10"), load("single-turn-vpaa-kspon-q06-s2-ast")
for dl in ("8", "10"):
    sdi = collections.Counter(); sdi0 = collections.Counter()
    for k, r in b.items():
        if k[0] == "kspon-eval_clean" and k[1] == dl and k in a:
            for x in ("substitutions", "deletions", "insertions"):
                sdi[x] += r["metrics"]["cer_nospace"][x]; sdi0[x] += a[k]["metrics"]["cer_nospace"][x]
    print(f"kspon clean δ{dl}: Stage1 {dict(sdi0)}  s2 {dict(sdi)}")
diff = sorted(((r["metrics"]["cer_nospace"]["errors"] - a[k]["metrics"]["cer_nospace"]["errors"], k) for k, r in b.items() if k[0] == "kspon-eval_clean" and k[1] == "10" and k in a), reverse=True)
print("top error increases δ10:", [d for d, _ in diff[:15]], "sum top20", sum(d for d, _ in diff[:20]), "total", sum(d for d, _ in diff))
for d, k in diff[:4]:
    r = b[k]; print(f"--- +{d} {k[2]} audio {r['audio_s']:.1f}s forced {r['forced']} flush {r['flush_rounds']}\n REF: {r['reference'][:150]}\n S1 : {a[k]['hypothesis'][:150]}\n S2 : {r['hypothesis'][:300]}")
