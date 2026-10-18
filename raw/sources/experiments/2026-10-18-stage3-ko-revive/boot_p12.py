# p12 vs p11 — 대화(conv_id) 단위 짝 bootstrap(코퍼스별 독일어 포함): 같은 held-out 창에서 두 run 의 cp·agn·cp−agn·이어짐 차이(pt)의 95 % 구간. 읽기만 한다.
import json, os, random, collections
R = "/soundai/users/tskim/VAPKT-data/results"
def load(d):
    out = {}
    for l in open(f"{R}/stage3-attr-errors-{d}/windows.jsonl"):
        w = json.loads(l); out[(w["id"], w["delta"])] = w
    return out
A = {n: load(d) for n, d in (("p11@2000", "st3-p11-revive-ck2000"), ("p11@3000", "st3-p11-revive-ck3000"), ("p11 final", "st3-p11-revive-final"),
                               ("p12@2000", "st3-p12-revive2-ck2000"), ("p12@3000", "st3-p12-revive2-ck3000"), ("p12 final", "st3-p12-revive2-final"))
     if os.path.exists(f"{R}/stage3-attr-errors-{d}/windows.jsonl")}
def boot(a, b, delta, lang=None, B=2000, corpus=None):
    keys = [k for k in A[a] if k[1] == delta and k in A[b] and (lang is None or A[a][k]["lang"] == lang) and (corpus is None or k[0].startswith(corpus + ":"))]
    by = collections.defaultdict(list)
    for k in keys: by[k[0].rsplit("@", 1)[0]].append(k)
    convs = list(by); rng = random.Random(0)
    def stat(ks):
        n = sum(A[a][k]["n_ref"] for k in ks) or 1
        f = lambda r, m: sum(m(A[r][k]) for k in ks) * 100 / n
        ms = {"cp": lambda w: w["cp"], "agn": lambda w: w["agn"], "cp−agn": lambda w: w["cp"] - w["agn"], "이어짐": lambda w: w["whole_streak_units"], "통째": lambda w: sum(v for k, v in w["units"].items() if k.startswith("whole"))}
        return {m: f(b, g) - f(a, g) for m, g in ms.items()}
    base = stat(keys); draws = collections.defaultdict(list)
    for _ in range(B):
        ks = [k for c in (rng.choice(convs) for _ in convs) for k in by[c]]
        for m, v in stat(ks).items(): draws[m].append(v)
    return {m: (base[m], sorted(v)[int(.025 * B)], sorted(v)[int(.975 * B)]) for m, v in draws.items()}, len(convs)
for a, b in [(a, b) for a, b in (("p11@2000", "p12@2000"), ("p11@3000", "p12@3000"), ("p11 final", "p12 final")) if a in A and b in A]:
    for d in (10, 4):
        for lang, corpus in ((None, None), ("KO", None), ("EN", None), ("DE", None), ("DE", "nsc"), ("DE", "kec"), ("DE", "ztc")):
            r, nc = boot(a, b, d, lang, corpus=corpus)
            if not nc: continue
            print(f"{b} − {a} δ{d} {corpus or lang or '전체'} (대화 {nc}): " + " · ".join(f"{m} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, (v, lo, hi) in r.items()))
