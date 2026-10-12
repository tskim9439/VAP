import json, glob, collections, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.ast_eval import translation_report
R = "/soundai/users/tskim/VAPKT-data/results"
keep = {json.loads(l)["id"] for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-covost2-sub7.jsonl")}
def load(d):
    return [json.loads(l) for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl") for l in open(f) if json.loads(l)["id"] in keep]
s2, s3 = load("ast-covost2-q06-s2-ast"), load("ast-covost2-sub7-q06-s3-ast")
a, b = translation_report(s2), translation_report(s3)
def cnt(rows):
    c = collections.Counter()
    for r in rows:
        if r.get("translation_reference") is None: continue
        k = f"{r['dataset']}/delta-{r['delta']}"
        if not r["token_ids"]: c[(k, "s")] += 1
        elif not r["translation_hypothesis"]: c[(k, "t")] += 1
    return c
ca, cb = cnt(s2), cnt(s3)
for k in sorted(b):
    x, y = a[k], b[k]; lat = lambda v: f"{v['laal_s']:.2f}" if "laal_s" in v else "  - "
    print(f"{k:28s} n={y['n']} BLEU {x['bleu']:5.2f} → {y['bleu']:5.2f} ({y['bleu'] - x['bleu']:+5.2f}) chrF {x['chrf']:5.2f} → {y['chrf']:5.2f} LAAL {lat(x)} → {lat(y)} "
          f"empty {x['empty_translations']} → {y['empty_translations']} (silent {ca[(k,'s')]}→{cb[(k,'s')]}, transcript-only {ca[(k,'t')]}→{cb[(k,'t')]})")
