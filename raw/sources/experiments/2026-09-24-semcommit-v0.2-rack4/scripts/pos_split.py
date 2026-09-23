"""SEM hits split by reference position: A candidates at the stream's last word vs mid-stream (text-position mapping, as commit_metrics). Also TURN early-fire examples."""
import json, sys, collections
sys.path.insert(0, "/home/tskim/VAP")
from vapasr.hf.commit_metrics import map_events, ref_chunk, SEM_END_ID, TURN_END_ID, emit_time
words_f, labels_f, streams_f = sys.argv[1:4]; cfgs = sys.argv[4].split(",")
W = {r["id"]: r for r in map(json.loads, open(words_f))}; L = {r["id"]: r for r in map(json.loads, open(labels_f))}
res = {c: collections.Counter() for c in cfgs}; ex = []
for l in open(streams_f):
    r = json.loads(l); c = r["config"]["name"]
    if c not in res: continue
    ws = sorted(W[r["id"]]["words"], key=lambda x: int(x["i"])); last = len(ws) - 1; K = int(r["K"])
    A = {int(x["after_word"]) for x in L.get(r["id"], {}).get("candidates", []) if x["grade"] == "A"}
    ev = [e for e in r["hyp"]["events"] if int(e["id"]) == SEM_END_ID]
    pos = set(map_events([x["text"] for x in ws], r["hyp"]["words"], ev, [ref_chunk(float(x["end_time"]), int(r["delta"]), K) for x in ws], K))
    for a in A:
        k = "end" if a == last else "mid"; res[c]["A_" + k] += 1; res[c]["hit_" + k] += a in pos
    for p in pos: res[c]["commit_" + ("end" if p == last else "mid")] += 1
    if c == cfgs[0]:
        tk = [int(k) for k, t in r["emitted"] if int(t) == TURN_END_ID]
        t_last = float(ws[-1]["end_time"]) if ws else 0
        for k in tk:
            if emit_time(k, K) < t_last - 1e-6 and len(ex) < 8: ex.append((r["id"], round(emit_time(k, K), 2), round(t_last, 2), " ".join(x["text"] for x in ws)[:80], r["hyp"]["text"][:80]))
for c, v in res.items():
    print(c, dict(v), f"R_end {v['hit_end']/max(1,v['A_end']):.2f} R_mid {v['hit_mid']/max(1,v['A_mid']):.2f}")
print("TURN early examples (id, turn_time, last_word_end, ref, hyp):"); [print("  ", e) for e in ex]
