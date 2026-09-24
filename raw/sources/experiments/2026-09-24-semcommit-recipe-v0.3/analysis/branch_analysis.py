"""v0.3: per-branch marginal precision on dev/test halves; other-branch sweep; KO punct FP by stream-final vs mid."""
import json, collections, sys
sys.path.insert(0, "/home/tskim/VAP")
from experiments.semcommit_gold import split_half
GD = "/data4/tskim/semcommit/gold/v1"; L3 = "/data4/tskim/semcommit/labels/v0.3"
def lab(g, i): return "C" if i in g["commit"] else "A" if i in g["ambig"] else "N"
allc = collections.defaultdict(list)
for part in ("ls-test", "gs-test", "ks-eval", "ks-long"):
    G = {r["id"]: {"commit": set(r["commit"]), "ambig": set(r["ambig"]), "n": r["n_words"]} for r in map(json.loads, open(f"{GD}/gold-{part}.jsonl"))}
    for r in map(json.loads, open(f"{L3}/labels-{part}.jsonl")):
        g = G[r["id"]]; h = split_half(r["id"])
        for c in r["candidates"]:
            allc[part].append(dict(h=h, br="punct" if c["punct"] else "other", grade=c["grade"], ps=c.get("p_safe_mean"), pr=c.get("p_rev"),
                                   gl=lab(g, c["after_word"]), final=c["after_word"] == g["n"] - 1, src=c.get("sources")))
print("keys sample:", sorted(r["candidates"][0].keys()))
print("\n== A labels by branch (C/AMBIG/NO, marginal P = C/(C+NO)) ==")
for part, C in allc.items():
    for h in ("dev", "test"):
        s = []
        for br in ("punct", "other"):
            k = collections.Counter(x["gl"] for x in C if x["h"] == h and x["br"] == br and x["grade"] == "A")
            p = k["C"] / (k["C"] + k["N"]) if k["C"] + k["N"] else float("nan")
            s.append(f"{br} {k['C']}/{k['A']}/{k['N']} P {p:.3f}")
        print(f"{part:8s} {h:4s}  " + " | ".join(s))
print("\n== other-branch sweep (all other candidates, not only A): C/NO kept at p_safe>=t, p_rev<=r ==")
for lang, parts in (("EN", ("ls-test", "gs-test")), ("KO", ("ks-eval", "ks-long"))):
    for h in ("dev", "test"):
        X = [x for p in parts for x in allc[p] if x["h"] == h and x["br"] == "other" and x["ps"] is not None and not any(str(t).startswith("tag") for t in (x["src"] or []))]
        tot = collections.Counter(x["gl"] for x in X)
        row = []
        for t in (0.35, 0.5, 0.7, 0.9, 0.95, 0.99):
            for r in (0.02, 0.2):
                k = collections.Counter(x["gl"] for x in X if x["ps"] >= t and (x["pr"] or 0) <= r)
                row.append(f"{t}/{r}:{k['C']}-{k['N']}")
        print(f"{lang} {h:4s} other cands C {tot['C']} AMB {tot['A']} NO {tot['N']} | " + " ".join(row))
print("\n== punct-branch A: stream-final vs mid (C/AMBIG/NO) and p_safe<0.02 subset ==")
for part, C in allc.items():
    for fin in (True, False):
        X = [x for x in C if x["br"] == "punct" and x["grade"] == "A" and x["final"] == fin]
        k = collections.Counter(x["gl"] for x in X); lo = collections.Counter(x["gl"] for x in X if (x["ps"] or 0) < 0.02)
        print(f"{part:8s} {'final' if fin else 'mid  '}: {k['C']}/{k['A']}/{k['N']}  (p_safe<0.02: {lo['C']}/{lo['A']}/{lo['N']})")
