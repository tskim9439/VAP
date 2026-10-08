"""Apply the selection rules to the YODAS-Granary German sample (Granary G, Qwen Q, Whisper W)."""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/Users/taesookim/Desktop/VAPKT/experiments")
import yodas_de_qwen_check as y  # noqa: E402

D = sys.argv[1]
rows = {r["utt_id"]: r for r in map(json.loads, open(f"{D}/results.jsonl", encoding="utf-8"))}
for v in ("v2", "v3"):
    for r in map(json.loads, open(f"{D}/whisper-large-{v}.jsonl", encoding="utf-8")):
        rows[r["utt_id"]][f"w{v}"] = r["hyp"] or ""


def dist(a, b, metric):
    na, nb = y.norm_de(a), y.norm_de(b)
    if metric == "wer":
        return y.edit_ops(na, nb) / max(1, len(na))
    ca, cb = list("".join(na)), list("".join(nb))
    return y.edit_ops(ca, cb) / max(1, len(ca))


R = list(rows.values())
tot_h = {t: sum(r["duration"] for r in R if t in ("all", r["task"])) / 3600 for t in ("all", "asr_only", "ast")}
cache = {}
for r in R:
    for m in ("wer", "cer"):
        for v in ("v2", "v3"):
            w = r[f"w{v}"]
            cache[(r["utt_id"], m, "GQ")] = dist(r["ref"], r["hyp"], m)
            cache[(r["utt_id"], m, f"GW{v}")] = dist(r["ref"], w, m)
            cache[(r["utt_id"], m, f"QW{v}")] = dist(r["hyp"], w, m)

print("agreement medians (CER, space-insensitive):")
for k in ("GQ", "GWv2", "GWv3", "QWv2", "QWv3"):
    xs = sorted(cache[(r["utt_id"], "cer", k)] for r in R)
    print(f"  {k:5s} median {xs[len(xs)//2]:.3f}  <=2% {sum(x <= .02 for x in xs)/len(xs):.2f}  <=5% {sum(x <= .05 for x in xs)/len(xs):.2f}")

print("\nrule: A = G within t of Q and W -> Granary text; B = else Q within t of W -> Qwen text; else drop")
print("metric thr  teacher | kept% (utts)  kept% (hours: all / asr_only / ast)  A% B%")
for m in ("wer", "cer"):
    for t in (0.03, 0.05):
        for v in ("v2", "v3"):
            stat = defaultdict(float)
            for r in R:
                a = cache[(r["utt_id"], m, "GQ")] <= t and cache[(r["utt_id"], m, f"GW{v}")] <= t
                b = not a and cache[(r["utt_id"], m, f"QW{v}")] <= t
                h = r["duration"] / 3600
                for tk in ("all", r["task"]):
                    if a or b:
                        stat[("kept_h", tk)] += h
                stat["A"] += a
                stat["B"] += b
            n = len(R)
            print(f"{m} {t:.2f} {v}     | {(stat['A']+stat['B'])/n:5.1%}         "
                  f"{stat[('kept_h','all')]/tot_h['all']:5.1%} / {stat[('kept_h','asr_only')]/tot_h['asr_only']:5.1%} / "
                  f"{stat[('kept_h','ast')]/tot_h['ast']:5.1%}        {stat['A']/n:5.1%} {stat['B']/n:5.1%}")
json.dump(R, open(f"{D}/merged.json", "w"), ensure_ascii=False)
