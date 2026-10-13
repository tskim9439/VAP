import json, sys, collections
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.textnorm_de import score_de
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
def align(r, h):
    n, m = len(r), len(h); D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1): D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(r[i-1] != h[j-1]))
    ops = []; i, j = n, m
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1] + (r[i-1] != h[j-1]):
            if r[i-1] != h[j-1]: ops.append(("S", r[i-1], h[j-1]))
            i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j] + 1: ops.append(("D", r[i-1], "")); i -= 1
        else: ops.append(("I", "", h[j-1])); j -= 1
    return ops
FILL = {"äh", "ähm", "öh", "öhm", "hm", "hmm", "mhm", "eh", "ehm", "em", "äm", "ah", "oh", "aha", "hä", "mh", "hehe", "haha", "hahaha", "hehehe"}
sub = collections.Counter(); dele = collections.Counter(); ins = collections.Counter(); N = 0; E = 0
errs_nf = []; refw = 0
for r in rows:
    R, H = score_de(r["ref"]).split(), score_de(r["hyp"]).split(); refw += len(R)
    ops = align(R, H); E += len(ops)
    for o, a, b in ops:
        if o == "S": sub[(a, b)] += 1
        elif o == "D": dele[a] += 1
        else: ins[b] += 1
    R2, H2 = [w for w in R if w not in FILL], [w for w in H if w not in FILL]
    errs_nf.append(len(align(R2, H2)) / max(1, len(R2)) if R2 else (0.0 if not H2 else 1.0))
print(f"utts {len(rows)} refwords {refw} pooled WER {E/refw:.3f}")
print("errors: S", sum(sub.values()), "D", sum(dele.values()), "I", sum(ins.values()))
print("filler deletions", sum(v for k, v in dele.items() if k in FILL), "filler insertions", sum(v for k, v in ins.items() if k in FILL))
print("top subs", [(a+">"+b, c) for (a, b), c in sub.most_common(160)])
print("top dels", dele.most_common(25))
print("top ins", ins.most_common(15))
import numpy as np
e = np.array(errs_nf); print("no-filler per-utt: median", np.median(e), "share>=0.2", (e >= 0.2).mean(), ">=0.3", (e >= 0.3).mean(), ">=0.5", (e >= 0.5).mean())
