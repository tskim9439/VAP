import collections, glob, json
from sacrebleu.metrics import BLEU
O = "/soundai/users/tskim/VAPKT-data/results/ast-fleurs-teacher-ceiling-20261011"
g = collections.defaultdict(list)
for l in open(f"{O}/translations.jsonl"):
    r = json.loads(l); g[(r["dataset"], r["cond"])].append((r["hypothesis"], r["translation_reference"], r["raw_output"]))
R = "/soundai/users/tskim/VAPKT-data/results"
for d in ("ast-fleurs-q06-s2-ast", "ast-fleurs-en-peak05-q06-s2-ast"):
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["dataset"] in ("fleurs-de-en", "fleurs-ko-en", "fleurs-en-de-peak05", "fleurs-en-ko-peak05"):
                g[(r["dataset"].replace("-peak05", ""), f"model-δ{r['delta']}")].append((r["translation_hypothesis"], r["translation_reference"], ""))
for (ds, c), rs in sorted(g.items()):
    if ds.endswith("-ko"): continue
    b = BLEU().corpus_score([h for h, _, _ in rs], [[r for _, r, _ in rs]])
    ml = sum(1 for *_, o in rs if o and len([x for x in o.splitlines() if x.strip()]) > 1)
    print(f"{ds:14s} {c:14s} BLEU {b.score:5.2f} BP {b.bp:.3f} len_ratio {b.sys_len / b.ref_len:.3f} prec1/4 {b.precisions[0]:.1f}/{b.precisions[3]:.1f}" + (f" multiline_raw={ml}" if ml else ""))
