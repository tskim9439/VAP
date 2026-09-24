"""Model <SEM_END> at gold positions, split by what the v0.3 labels had there: A(punct) / A(other) / B / N / non-candidate. Usage: attrib.py <streams.jsonl> <part> [config]"""
import json, sys, collections
sys.path.insert(0, "/home/tskim/VAP")
from vapasr.hf.commit_metrics import SEM_END_ID, map_events, ref_chunk
GD = "/data4/tskim/semcommit/gold/v1"; L3 = "/data4/tskim/semcommit/labels/v0.3"
sp, part = sys.argv[1], sys.argv[2]; cfg = sys.argv[3] if len(sys.argv) > 3 else "bias=0"
W = {r["id"]: r for r in map(json.loads, open(f"{GD}/words-{part}.jsonl"))}; G = {r["id"]: r for r in map(json.loads, open(f"{GD}/gold-{part}.jsonl"))}
LB = {r["id"]: {c["after_word"]: ("A_" + ("punct" if c["punct"] else "other") if c["grade"] == "A" else c["grade"]) for c in r["candidates"]} for r in map(json.loads, open(f"{L3}/labels-{part}.jsonl"))}
cnt = collections.Counter(); ex = []
for r in map(json.loads, open(sp)):
    name = r["config"]["name"] if isinstance(r["config"], dict) else str(r["config"])
    if name != cfg or r["id"] not in G: continue
    ws = sorted(W[r["id"]]["words"], key=lambda w: int(w["i"])); K = int(r["K"]); d = int(r["delta"]); sem = (r.get("event_ids") or [SEM_END_ID])[0]
    ev = [e for e in r["hyp"]["events"] if int(e["id"]) == int(sem)]
    pos = map_events([w["text"] for w in ws], r["hyp"]["words"], ev, [ref_chunk(float(w["end_time"]), d, K) for w in ws], K)
    g = G[r["id"]]; C, Am = set(g["commit"]), set(g["ambig"]); seen = set()
    for e, p in zip(ev, pos):
        if e.get("mid_word") or p < 0 or p in seen: cnt[("err", "mid/no_word/dup")] += 1; continue
        seen.add(p); gl = "hit" if p in C else "ambig" if p in Am else "NO"
        src = LB.get(r["id"], {}).get(p, "noncand"); cnt[(gl, src)] += 1
        if gl == "NO" and len(ex) < int(__import__("os").environ.get("EX", "0")):
            ex.append(f"   {src:8s} p={p}/{len(ws) - 1}: {' '.join(w['text'] for w in ws[max(0, p - 5):p + 1])} ‖ {' '.join(w['text'] for w in ws[p + 1:p + 5])}")
tot = collections.Counter()
for (gl, src), n in cnt.items(): tot[gl] += n
print(f"{sp.split('/')[-1]} {cfg}: " + " | ".join(f"{gl} {tot[gl]}: " + ", ".join(f"{src} {n}" for (g2, src), n in sorted(cnt.items()) if g2 == gl) for gl in ("hit", "NO", "ambig", "err")))
print("\n".join(ex))
