"""single-turn-v1 summary.json 비교: 세트 × δ 별 주 지표(EN WER, KO CER nospace)와 방출 지연 p50(예측 jsonl 에서)."""
import json, sys, os, statistics
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("E2", "single-turn-E2-final-v1"), ("v035-d8", "single-turn-semcommit-v035-d8-v1"), ("ckpt35000", "single-turn-35000-d2-d4-v1"), ("ckpt35000", "single-turn-35000-d6-v1")]
rows = {}
for name, d in runs:
    p = os.path.join(R, d, "summary.json")
    if not os.path.exists(p): continue
    s = json.load(open(p))
    for g, v in s["groups"].items():
        ds, dl = g.split("/delta-"); m = v["metrics"]
        prim = "wer" if ds.startswith("librispeech") else "cer_nospace"
        rows.setdefault((ds, int(dl)), {})[name] = (m[prim]["rate"], v.get("forced", 0), s["complete"])
sets = ["librispeech-test-clean", "librispeech-test-other", "kspon-eval_clean", "kspon-eval_other"]
names = ["E2", "v035-d8", "ckpt35000"]
print("set/δ".ljust(30) + "".join(n.rjust(12) for n in names) + "   Δ(v035−E2) rel")
for ds in sets:
    for dl in (2, 4, 6, 8):
        r = rows.get((ds, dl));
        if not r: continue
        cells = [f"{r[n][0]*100:.2f}" if n in r else "-" for n in names]
        rel = f"{(r['v035-d8'][0] / r['E2'][0] - 1) * 100:+.1f}%" if "E2" in r and "v035-d8" in r else ""
        print(f"{ds}/δ{dl}".ljust(30) + "".join(c.rjust(12) for c in cells) + "   " + rel)
