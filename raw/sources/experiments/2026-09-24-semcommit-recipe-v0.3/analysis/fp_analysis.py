"""v0.3 A labels at gold NO: by branch (punct/other), with P(SAFE) mean / P(REVISION); what a punct filter would cost/gain."""
import json, collections
GD = "/data4/tskim/semcommit/gold/v1"; L3 = "/data4/tskim/semcommit/labels/v0.3"
for part in ("ls-test", "gs-test", "ks-eval", "ks-long"):
    W = {r["id"]: r for r in map(json.loads, open(f"{GD}/words-{part}.jsonl"))}; G = {r["id"]: r for r in map(json.loads, open(f"{GD}/gold-{part}.jsonl"))}
    rows = []
    for r in map(json.loads, open(f"{L3}/labels-{part}.jsonl")):
        g = G[r["id"]]
        for c in r["candidates"]:
            if c["grade"] != "A": continue
            lab = "C" if c["after_word"] in g["commit"] else "A" if c["after_word"] in g["ambig"] else "N"
            rows.append((c["punct"], c["p_safe_mean"], c["p_rev"], lab, r["id"], c["after_word"]))
    for br in (True, False):
        R = [x for x in rows if x[0] == br]
        if not R: continue
        cnt = collections.Counter(x[3] for x in R)
        print(f"{part:8s} {'punct' if br else 'other'}: A {len(R)}  gold C {cnt['C']} AMBIG {cnt['A']} NO {cnt['N']}", end="")
        if br:
            for t in (0.02, 0.05, 0.1, 0.2):
                keep = [x for x in R if x[1] >= t]; k = collections.Counter(x[3] for x in keep)
                print(f" | p_safe>={t}: lose C {cnt['C'] - k['C']} drop NO {cnt['N'] - k['N']}", end="")
            for t in (0.9, 0.7, 0.5):
                keep = [x for x in R if x[2] <= t]; k = collections.Counter(x[3] for x in keep)
                print(f" | p_rev<={t}: lose C {cnt['C'] - k['C']} drop NO {cnt['N'] - k['N']}", end="")
        print()
    ex = [x for x in rows if x[3] == "N"][:4]
    for pb, ps, pr, lab, sid, i in ex:
        ws = W[sid]["words"]; print(f"      FP {'P' if pb else 'O'} ps={ps:.2f} pr={pr:.2f}: …{' '.join(w['text'] for w in ws[max(0, i - 6):i + 1])} ‖ {' '.join(w['text'] for w in ws[i + 1:i + 6])}")
