"""Blind spot-check sample from graded semcommit labels: per language, stratified by grade (A / N / B), shuffled, grade hidden in items.jsonl, key in key.jsonl."""
import json, random, sys
D, L, out = "/data4/tskim/semcommit/data/v0", "/data4/tskim/semcommit/labels/v0.2", sys.argv[1]
plan = {("ls-test", "A"): 30, ("ls-test", "N"): 10, ("ls-test", "B"): 10, ("ks-eval", "A"): 30, ("ks-eval", "N"): 10, ("ks-eval", "B"): 10}
rng = random.Random(7); items = []; keys = []
for s in ("ls-test", "ks-eval"):
    W = {r["id"]: r for r in map(json.loads, open(f"{D}/words-{s}.jsonl"))}
    pool = {"A": [], "N": [], "B": []}
    for r in map(json.loads, open(f"{L}/labels-{s}.jsonl")):
        for c in r["candidates"]: pool[c["grade"]].append((r["id"], c))
    for g in ("A", "N", "B"):
        for sid, c in rng.sample(pool[g], min(plan[(s, g)], len(pool[g]))):
            ws = [w["text"] for w in W[sid]["words"]]; k = c["after_word"]
            items.append(dict(stream=sid, lang=W[sid]["lang"], after_word=k, prefix=" ".join(ws[:k + 1]), future=" ".join(ws[k + 1:k + 13]), stream_end=k == len(ws) - 1))
            keys.append(dict(stream=sid, after_word=k, set=s, grade=g, why=c["why"], B=c["B"], C=c["C"], future_unobserved=c["future_unobserved"]))
order = list(range(len(items))); rng.shuffle(order)
with open(f"{out}/items.jsonl", "w") as f, open(f"{out}/key.jsonl", "w") as g:
    for n, i in enumerate(order): f.write(json.dumps(dict(n=n, **items[i]), ensure_ascii=False) + "\n"); g.write(json.dumps(dict(n=n, **keys[i]), ensure_ascii=False) + "\n")
print(len(items), "items")
