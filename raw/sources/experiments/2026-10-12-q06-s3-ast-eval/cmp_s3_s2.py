import json, glob, collections
R = "/soundai/users/tskim/VAPKT-data/results"
def tr(d):
    return json.load(open(f"{R}/{d}/summary.json")).get("translation", {})
def silent(d):
    c = collections.Counter(); t = collections.Counter()
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("translation_reference") is None: continue
            k = (r["dataset"], str(r["delta"])); t[k] += 1
            if not r["token_ids"]: c[(k, "silent")] += 1
            elif not r["translation_hypothesis"]: c[(k, "tonly")] += 1
    return c, t
for s2d, s3d in (("ast-fleurs-q06-s2-ast", "ast-fleurs-q06-s3-ast"), ("ast-fleurs-en-peak05-q06-s2-ast", "ast-fleurs-en-peak05-q06-s3-ast")):
    a, b = tr(s2d), tr(s3d); cb, tb = silent(s3d); ca, ta = silent(s2d)
    for k in sorted(b):
        if not any(k.endswith(x) for x in ("delta-4", "delta-10", "delta-final")): continue
        x, y = a.get(k, {}), b[k]; ds, dl = k.split("/"); dl = dl.replace("delta-", "")
        lat = lambda v: f"{v['laal_s']:.2f}" if "laal_s" in v else "  - "
        print(f"{k:34s} BLEU {x.get('bleu', 0):5.2f} → {y['bleu']:5.2f} ({y['bleu'] - x.get('bleu', 0):+5.2f}) chrF {x.get('chrf', 0):5.2f} → {y['chrf']:5.2f}  LAAL {lat(x)} → {lat(y)}  "
              f"empty {x.get('empty_translations')} → {y['empty_translations']} (silent {ca[((ds, dl), 'silent')]}→{cb[((ds, dl), 'silent')]}, transcript-only {ca[((ds, dl), 'tonly')]}→{cb[((ds, dl), 'tonly')]})")
