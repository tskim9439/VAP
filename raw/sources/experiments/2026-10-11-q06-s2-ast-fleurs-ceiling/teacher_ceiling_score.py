"""Score teacher_ceiling_gen.py output next to the model's own FLEURS results (same rows). usage: <conda python> ... <out_dir>"""
import collections, glob, json, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.ast_eval import corpus_scores
out = sys.argv[1]
g = collections.defaultdict(list)
for l in open(f"{out}/translations.jsonl"):
    r = json.loads(l); g[(r["dataset"], r["cond"])].append(r)
R = "/soundai/users/tskim/VAPKT-data/results"
for d in ("ast-fleurs-q06-s2-ast", "ast-fleurs-en-peak05-q06-s2-ast"):
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if str(r["delta"]) in ("10", "final"):
                ds = r["dataset"].replace("-peak05", ""); tag = ("model-peak05-" if "peak05" in r["dataset"] else "model-") + ("d10" if str(r["delta"]) == "10" else "final")
                g[(ds, tag)].append(dict(hypothesis=r["translation_hypothesis"], translation_reference=r["translation_reference"], tgt_lang=r["tgt_lang"], id=r["id"]))
rep = {}
for (ds, c), rs in sorted(g.items()):
    s = corpus_scores([r["hypothesis"] for r in rs], [r["translation_reference"] for r in rs], rs[0]["tgt_lang"])
    if rs[0]["tgt_lang"] == "Korean":
        from sacrebleu.metrics import BLEU
        s["bleu_13a"] = round(BLEU(tokenize="13a").corpus_score([r["hypothesis"] for r in rs], [[r["translation_reference"] for r in rs]]).score, 2)
    s["empty"] = sum(not r["hypothesis"] for r in rs); rep[f"{ds}/{c}"] = s
    print(f"{ds:14s} {c:22s} n={s['n']:4d} BLEU {s['bleu']:6.2f} chrF {s['chrf']:6.2f}" + (f" BLEU13a {s['bleu_13a']:5.2f}" if "bleu_13a" in s else "") + f" empty={s['empty']}")
json.dump(rep, open(f"{out}/scores.json", "w"), indent=1, ensure_ascii=False)
