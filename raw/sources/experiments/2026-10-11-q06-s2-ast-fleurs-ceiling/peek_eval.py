"""Read-only peek at (partial) eval predictions: ASR error rate per dataset/δ and translation BLEU/chrF/LAAL. Writes nothing."""
import glob, json, sys, collections
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.ast_eval import translation_report
R = "/soundai/users/tskim/VAPKT-data/results"
for d in sys.argv[1:]:
    rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/{d}/predictions-rank*.jsonl")) for l in open(f)]
    print(f"== {d}: {len(rows)} predictions")
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        m = r["metrics"][r["primary_metric"]]; a = acc[(r["dataset"], str(r["delta"]), r["primary_metric"])]
        a[0] += m["errors"]; a[1] += m["n_ref"]; a[2] += 1; a[3] += r.get("forced", 0)
    order = {"4": 0, "6": 1, "8": 2, "10": 3, "final": 4}
    for (ds, dl, pm), (e, n, c, fz) in sorted(acc.items(), key=lambda x: (x[0][0], order.get(x[0][1], 9))):
        print(f"  ASR {ds:26s} δ{dl:6s} {pm:12s} {e / n * 100:6.2f}  n={c}  forced={fz}")
    if any(r.get("translation_reference") is not None for r in rows):
        rep = translation_report(rows)
        for k in sorted(rep, key=lambda k: (k.split("/")[0], order.get(k.split("delta-")[1], 9))):
            v = rep[k]
            print(f"  ST  {k:32s} BLEU {v['bleu']:6.2f} chrF {v['chrf']:6.2f} n={v['n']:5d} empty={v['empty_translations']:3d}"
                  + (f" LAAL {v['laal_s']:.2f}/{v['laal_median_s']:.2f}s AL {v['al_s']:.2f}/{v['al_median_s']:.2f}s pre_end {v['pre_end']:.2f}" if 'laal_s' in v else ""))
