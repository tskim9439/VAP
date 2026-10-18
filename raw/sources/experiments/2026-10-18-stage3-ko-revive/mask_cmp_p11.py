# 마스크 구간별 귀속 — 같은 2 화자 창(δ10)에서 p10 대비 p11(격리 발화 되살리기)의 귀속 몫과 참조 교대당 오배정, 차이의 대화 단위 bootstrap 95 % 구간.
# 창 마스크 비율은 평가 원출력의 masked_chunks/K(평가는 두 run 이 같다). 읽기만 한다.  python3 mask_cmp_p11.py [ck2000|ck3000|final]
import collections, json, random, sys
R = "/soundai/users/tskim/VAPKT-data/results"; tag = sys.argv[1] if len(sys.argv) > 1 else "ck3000"
M = {(r["id"], r["delta"]): r["masked_chunks"] / max(1, r["K"]) for r in map(json.loads, open(f"{R}/stage3-eval-st3-p10-sim-{tag}/predictions-rank0.jsonl"))}
W = lambda d: {(w["id"], w["delta"]): w for w in map(json.loads, open(f"{R}/stage3-attr-errors-{d}/windows.jsonl"))}
A, B = W(f"st3-p10-sim-{tag}"), W(f"st3-p11-revive-{tag}")
mis = lambda w: w["cp"] - w["agn"]


def boot(ks, n_boot=2000):
    by = collections.defaultdict(list)
    for k in ks: by[k[0].rsplit("@", 1)[0]].append(k)
    convs = list(by); rng = random.Random(0)
    d = lambda kk: 100 * (sum(mis(B[k]) for k in kk) - sum(mis(A[k]) for k in kk)) / max(1, sum(A[k]["n_ref"] for k in kk))
    v = sorted(d([k for c in (rng.choice(convs) for _ in convs) for k in by[c]]) for _ in range(n_boot))
    return d(ks), v[int(.025 * n_boot)], v[int(.975 * n_boot)], len(convs)


print(f"[{tag} δ10, 2 화자 창] 마스크 구간 · 창 · 귀속 몫 p10 → p11 (차이 [95 %]) · 교대당 오배정 p10 → p11")
for lang in ("KO", "DE", "EN"):
    keys = [k for k in A if k[1] == 10 and k in B and k in M and A[k]["lang"] == lang and A[k].get("n_spk") == 2]
    for name, lo, hi in (("< 5 %", 0, .05), ("≥ 5 %", .05, 1.01), ("5–20 %", .05, .2), ("20–40 %", .2, .4), ("≥ 40 %", .4, 1.01)):
        ks = [k for k in keys if lo <= M[k] < hi]
        if len(ks) < 5: continue
        n = sum(A[k]["n_ref"] for k in ks); ch = max(1, sum(A[k]["ref_changes"] for k in ks)); dv, lo95, hi95, nc = boot(ks)
        print(f"{lang} {name:>8} 창 {len(ks):4d}(대화 {nc:3d}) · 귀속 몫 {100 * sum(mis(A[k]) for k in ks) / n:5.2f} → {100 * sum(mis(B[k]) for k in ks) / n:5.2f}"
              f" ({dv:+.2f} [{lo95:+.2f}, {hi95:+.2f}]) · 교대당 {sum(mis(A[k]) for k in ks) / ch:.2f} → {sum(mis(B[k]) for k in ks) / ch:.2f}")
