# 대화(conv_id) 단위 짝 bootstrap: 같은 held-out 창에서 두 run 의 cp·agn·cp−agn·이어짐 차이(pt)의 95 % 구간. 읽기만 한다.
import json, os, random, collections
R = "/soundai/users/tskim/VAPKT-data/results"
def load(d):
    out = {}
    for l in open(f"{R}/stage3-attr-errors-{d}/windows.jsonl"):
        w = json.loads(l); out[(w["id"], w["delta"])] = w
    return out
A = {n: load(d) for n, d in (("r1@2000", "q06-st3-r1-ck2000-v3"), ("p10@2000", "st3-p10-sim-ck2000"), ("p10@3000", "st3-p10-sim-ck3000"), ("p10 final", "st3-p10-sim-final"),
                               ("p11@2000", "st3-p11-revive-ck2000"), ("p11@3000", "st3-p11-revive-ck3000"), ("p11 final", "st3-p11-revive-final"))
     if os.path.exists(f"{R}/stage3-attr-errors-{d}/windows.jsonl")}
def boot(a, b, delta, lang=None, B=2000):
    keys = [k for k in A[a] if k[1] == delta and k in A[b] and (lang is None or A[a][k]["lang"] == lang)]
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
for a, b in [(a, b) for a, b in (("p10@2000", "p11@2000"), ("p10@3000", "p11@3000"), ("p10 final", "p11 final"), ("r1@2000", "p11@3000"), ("p10@3000", "p11 final")) if a in A and b in A]:
    for d in (10, 4):
        for lang in (None, "KO", "EN", "DE"):
            r, nc = boot(a, b, d, lang)
            print(f"{b} − {a} δ{d} {lang or '전체'} (대화 {nc}): " + " · ".join(f"{m} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, (v, lo, hi) in r.items()))
