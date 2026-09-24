"""No-punctuation re-annotation vs gold (which was annotated with the punctuated/original transcript visible)."""
import json, sys, collections
G = sys.argv[1]; name = sys.argv[2]
ann = json.load(open(f"{G}/nopunct/{name}.json")); ids = json.load(open(f"{G}/nopunct/{name}.ids.json"))
W, GD = {}, {}
for part in ("ls-test", "gs-test", "ks-eval", "ks-long"):
    for l in open(f"{G}/words-{part}.jsonl"): r = json.loads(l); W[r["id"]] = r
for f in ("gold-ls-test", "gold-gs", "gold-ks-short", "gold-ks-long"):
    for l in open(f"{G}/out/{f}.jsonl"): r = json.loads(l); GD[r["id"]] = r
c = collections.Counter()
for sid in ids:
    g = GD[sid]; a = ann[sid]; gc, ga = set(g["commit"]), set(g["ambig"]); nc, na = set(a["commit"]), set(a["ambig"])
    for w in W[sid]["words"]:
        i = w["i"]; kind = "punct" if "punct_final" in (w.get("tags") or []) else "other"
        gl = "C" if i in gc else "A" if i in ga else "N"; nl = "C" if i in nc else "A" if i in na else "N"
        c[(kind, gl, nl)] += 1
for kind in ("punct", "other"):
    tot = {k: sum(v for (kk, gl, nl), v in c.items() if kk == kind and gl == k) for k in "CAN"}
    both = c[(kind, "C", "C")]; gC = tot["C"]; nC = sum(v for (kk, gl, nl), v in c.items() if kk == kind and nl == "C")
    f1 = 2 * both / (gC + nC) if gC + nC else float("nan")
    print(f"{name} {kind:5s}: gold C {gC} A {tot['A']} N {tot['N']} | no-punct C {nC} | both C {both} | COMMIT F1 {f1:.3f} | "
          f"gold C → no-punct C/A/N {c[(kind,'C','C')]}/{c[(kind,'C','A')]}/{c[(kind,'C','N')]} | gold N → no-punct C {c[(kind,'N','C')]}")
allC = sum(v for (k, gl, nl), v in c.items() if gl == "C"); allN = sum(v for (k, gl, nl), v in c.items() if nl == "C"); bothC = sum(v for (k, gl, nl), v in c.items() if gl == "C" and nl == "C")
print(f"{name} all: COMMIT F1 {2 * bothC / (allC + allN):.3f} (gold {allC}, no-punct {allN}, both {bothC})")
