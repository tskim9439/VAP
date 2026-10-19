# st4-p1 vs p11 ck3000 — Stage 3 회귀(같은 held-out 창 200·받아쓰기 prefix): 대화 평가 요약 + 대화(conv_id) 단위 짝 bootstrap 95 % 구간. 읽기만 한다.
import json, os, random, collections
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("p11@3000", "st3-p11-revive-ck3000"), ("st4@2000", "st4-p1-ck2000"), ("st4@3000", "st4-p1-ck3000"), ("st4 final", "st4-p1-final")]
S = {n: json.load(open(f"{R}/stage3-eval-{d}/summary.json")) for n, d in runs if os.path.exists(f"{R}/stage3-eval-{d}/summary.json")}
for d in (10, 4):
    print(f"[δ{d} 코퍼스 평균]".ljust(14), " ".join(f"{n:>9s}" for n in S))
    for m, f in (("err_slot", "{:9.2f}"), ("err_cp", "{:9.2f}"), ("err_agn", "{:9.2f}"), ("swap_share", "{:9.3f}"), ("eot_recall", "{:9.3f}"), ("eot_precision", "{:9.3f}"), ("eot_hold_fire", "{:9.3f}")):
        print(m.ljust(14), " ".join(f.format(sum(s[k][m] for k in s if k.endswith(f"/d{d}")) / len([k for k in s if k.endswith(f"/d{d}")])) for s in S.values()))
for d in (10, 4):
    print(f"[δ{d} 코퍼스별 slot · cp · agn]")
    for c in ("aihub134-1", "aihub134-2", "aihub71631", "otoSpeech", "kec", "nsc", "ztc"):
        print(c.ljust(14), " ".join(f"{s[f'{c}/d{d}']['err_slot']:5.1f}·{s[f'{c}/d{d}']['err_cp']:5.1f}·{s[f'{c}/d{d}']['err_agn']:4.1f}" for s in S.values()))

def load(d):
    p = f"{R}/stage3-attr-errors-{d}/windows.jsonl"
    return {(w["id"], w["delta"]): w for w in map(json.loads, open(p))} if os.path.exists(p) else None
A = {n: w for n, d in runs if (w := load(d))}
def boot(a, b, delta, lang=None, corpus=None, B=2000):
    keys = [k for k in A[a] if k[1] == delta and k in A[b] and (lang is None or A[a][k]["lang"] == lang) and (corpus is None or k[0].startswith(corpus + ":"))]
    by = collections.defaultdict(list)
    for k in keys: by[k[0].rsplit("@", 1)[0]].append(k)
    convs = list(by); rng = random.Random(0)
    ms = {"cp": lambda w: w["cp"], "agn": lambda w: w["agn"], "cp−agn": lambda w: w["cp"] - w["agn"]}
    def stat(ks):
        n = sum(A[a][k]["n_ref"] for k in ks) or 1
        return {m: (sum(g(A[b][k]) for k in ks) - sum(g(A[a][k]) for k in ks)) * 100 / n for m, g in ms.items()}
    base = stat(keys); draws = collections.defaultdict(list)
    for _ in range(B):
        ks = [k for c in (rng.choice(convs) for _ in convs) for k in by[c]]
        for m, v in stat(ks).items(): draws[m].append(v)
    return {m: (base[m], sorted(v)[int(.025 * B)], sorted(v)[int(.975 * B)]) for m, v in draws.items()}, len(convs)
print("[짝 bootstrap: st4 − p11@3000, pt, 95 % 구간]")
for b in ("st4@2000", "st4@3000", "st4 final"):
    if b not in A or "p11@3000" not in A: continue
    for d in (10, 4):
        for lang, corpus in ((None, None), ("KO", None), ("EN", None), ("DE", None), (None, "aihub134-1"), (None, "aihub134-2"), (None, "aihub71631")):
            r, nc = boot("p11@3000", b, d, lang, corpus)
            if nc: print(f"{b} δ{d} {corpus or lang or '전체'} (대화 {nc}): " + " · ".join(f"{m} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, (v, lo, hi) in r.items()))
