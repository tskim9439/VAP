"""Candidate rule-based negatives vs gold: (1) positions inside a stream-initial reply-word prefix (not the stream's last word),
(2) Korean stream-final word with a connective ending and no punct_final. Prints gold C/AMBIG/NO per rule and half."""
import json, sys, collections, re
sys.path.insert(0, "/home/tskim/VAP")
from experiments.semcommit_gold import split_half
from vapasr.data.semcommit_llm import RESPONSE_TOKENS
GD = "/data4/tskim/semcommit/gold/v1"; V0 = "/data4/tskim/semcommit/data/v0"
norm = lambda t: re.sub(r"[^\w'-]", "", str(t).lower())
CONN = ("니까", "는데", "은데", "던데", "는지", "고", "서", "면", "지만", "려고", "으려고", "가지고", "가지구", "다가", "거나", "든지", "면서", "도록", "게", "며", "니", "러")
def prefix_positions(ws, lang):
    toks = RESPONSE_TOKENS.get(lang, ()); out = []
    for p, w in enumerate(ws):
        if norm(w["text"]) not in toks: break
        if p < len(ws) - 1: out.append(p)
    return out
def conn_final(ws, lang):
    if lang != "Korean" or not ws: return None
    w = ws[-1]; t = norm(w["text"])
    if "punct_final" in (w.get("tags") or []): return None
    for e in sorted(CONN, key=len, reverse=True):
        if t.endswith(e) and len(t) > len(e): return e
    return None
for part in ("ks-eval", "ks-long", "gs-test", "ls-test"):
    W = {r["id"]: r for r in map(json.loads, open(f"{GD}/words-{part}.jsonl"))}; G = {r["id"]: r for r in map(json.loads, open(f"{GD}/gold-{part}.jsonl"))}
    c1, c2, ends = collections.Counter(), collections.Counter(), collections.Counter()
    for sid, g in G.items():
        ws = sorted(W[sid]["words"], key=lambda w: int(w["i"])); h = split_half(sid); lab = lambda i: "C" if i in g["commit"] else "A" if i in g["ambig"] else "N"
        for p in prefix_positions(ws, g["lang"]): c1[(h, lab(p))] += 1
        e = conn_final(ws, g["lang"])
        if e: c2[(h, lab(len(ws) - 1))] += 1; ends[(e, lab(len(ws) - 1))] += 1
    f = lambda c: {h: f"C {c[(h, 'C')]} AMB {c[(h, 'A')]} NO {c[(h, 'N')]}" for h in ("dev", "test")}
    print(f"{part:8s} reply-prefix {f(c1)} | conn-final {f(c2)}")
    if ends: print("          endings:", dict(sorted(ends.items())))
for s in ("ks-train", "ls-train"):
    W = [json.loads(l) for l in open(f"{V0}/words-{s}.jsonl")]; n1 = n2 = 0
    for r in W:
        ws = sorted(r["words"], key=lambda w: int(w["i"])); n1 += len(prefix_positions(ws, r["lang"])); n2 += conn_final(ws, r["lang"]) is not None
    print(s, "reply-prefix positions", n1, "conn-final streams", n2, "of", len(W))
