import json, sys, collections
sys.path.insert(0, "/home/tskim/VAP")
from experiments.semcommit_gold import split_half
GD = "/data4/tskim/semcommit/gold/v1"; L = "/data4/tskim/semcommit/labels/v0.3.2"
for part in ("ls-test", "gs-test", "ks-eval", "ks-long", "ls-train", "ks-train"):
    G = {r["id"]: r for r in map(json.loads, open(f"{GD}/gold-{part}.jsonl"))} if "train" not in part else {}
    c = collections.Counter(); words = collections.Counter()
    for r in map(json.loads, open(f"{L}/labels-{part}.jsonl")):
        g = G.get(r["id"]); h = split_half(r["id"])
        for x in r["candidates"]:
            if not x.get("resp_head"): continue
            lab = "-" if g is None else "C" if x["after_word"] in g["commit"] else "A" if x["after_word"] in g["ambig"] else "N"
            c[(h if g else "all", lab, x["grade"])] += 1
    print(part, dict(sorted(c.items())))
print(json.dumps(json.load(open(f"{L}/thresholds.report.json"))["languages"]["Korean"]["branches"], indent=None)[:600])
