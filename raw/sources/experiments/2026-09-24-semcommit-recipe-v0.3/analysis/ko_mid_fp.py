import json
GD = "/data4/tskim/semcommit/gold/v1"; L = "/data4/tskim/semcommit/labels/v0.3.1"
for part in ("ks-eval", "ks-long"):
    W = {r["id"]: r for r in map(json.loads, open(f"{GD}/words-{part}.jsonl"))}; G = {r["id"]: r for r in map(json.loads, open(f"{GD}/gold-{part}.jsonl"))}
    for r in map(json.loads, open(f"{L}/labels-{part}.jsonl")):
        g = G[r["id"]]; ws = W[r["id"]]["words"]; n = len(ws)
        for c in r["candidates"]:
            i = c["after_word"]
            if c["grade"] != "A" or i == n - 1: continue
            lab = "C" if i in g["commit"] else "A" if i in g["ambig"] else "N"
            if lab == "C": continue
            nxt = [w.get("tags") or [] for w in ws[i + 1:i + 3]]
            print(f"{part} {lab} ps={c['p_safe_mean']:.2f} pr={c['p_rev']:.2f} src={','.join(c['sources'])} | {' '.join(w['text'] for w in ws[max(0, i - 5):i + 1])} ‖ {' '.join(w['text'] for w in ws[i + 1:i + 6])} | next tags {nxt}")
