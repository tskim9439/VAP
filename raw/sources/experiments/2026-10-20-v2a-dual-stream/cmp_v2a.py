# V2-a 파일럿 대조 — 같은 held-out 창(Stage 3 회귀 평가 → cst_attr_errors)에서 대조군 대 V2-a 의 화자 귀속을 대화(conv_id) 단위 짝 bootstrap 으로 비교한다. 읽기만 한다.
#   python cmp_v2a.py <대조 태그> <V2-a 태그> [<대조 태그> <V2-a 태그> …]   (태그 = results/stage3-attr-errors-<태그>, 예 st4-m2a-ck1500 st4-v2a-ck1500)
# 지표(참조 단위 대비 pt; 대조값과 차이 = V2-a − 대조, 95 % 구간): cp · agn · 귀속 몫(cp − agn) · 턴 통째 오배정(whole_short + whole_long) · 연속 뒤바뀜(whole_streak).
# 묶음: 전체 · 한·영·독 · 겹침 창(overlap ≥ 5 %).
import collections, json, random, sys
R = "/soundai/users/tskim/VAPKT-data/results"
M = {"cp": lambda w: w["cp"], "agn": lambda w: w["agn"], "attr": lambda w: w["cp"] - w["agn"],
     "whole": lambda w: sum(v for k, v in w["units"].items() if k.startswith("whole_")), "streak": lambda w: w["whole_streak_units"]}


def load(tag):
    return {(w["id"], w["delta"]): w for w in map(json.loads, open(f"{R}/stage3-attr-errors-{tag}/windows.jsonl"))}


def boot(A, B, keys, n_boot=2000):
    by = collections.defaultdict(list)
    for k in keys: by[k[0].rsplit("@", 1)[0]].append(k)
    convs = list(by); rng = random.Random(0)
    def stat(ks):
        n = sum(A[k]["n_ref"] for k in ks) or 1
        return {m: 100 * (sum(g(B[k]) for k in ks) - sum(g(A[k]) for k in ks)) / n for m, g in M.items()}
    base = {m: 100 * sum(g(A[k]) for k in keys) / (sum(A[k]["n_ref"] for k in keys) or 1) for m, g in M.items()}
    diff = stat(keys); draws = collections.defaultdict(list)
    for _ in range(n_boot):
        for m, v in stat([k for c in (rng.choice(convs) for _ in convs) for k in by[c]]).items(): draws[m].append(v)
    return base, {m: (diff[m], sorted(v)[int(.025 * n_boot)], sorted(v)[int(.975 * n_boot)]) for m, v in draws.items()}, len(convs)


def main(tags):
    assert tags and len(tags) % 2 == 0, "대조·V2-a 태그를 짝으로 준다"
    for a, b in zip(tags[::2], tags[1::2]):
        A, B = load(a), load(b); common = sorted(set(A) & set(B))
        assert all(A[k]["n_ref"] == B[k]["n_ref"] for k in common), "참조 단위 수가 다르다 — 같은 평가 창이 아니다"
        print(f"== {b} − {a}: 공통 창·δ {len(common)}(대조 {len(A)} · V2-a {len(B)})")
        for d in (10, 4):
            for name, f in (("전체", lambda w: True), ("KO", lambda w: w["lang"] == "KO"), ("EN", lambda w: w["lang"] == "EN"), ("DE", lambda w: w["lang"] == "DE"),
                            ("겹침≥5%", lambda w: w["overlap"] >= 0.05)):
                keys = [k for k in common if k[1] == d and f(A[k])]
                if not keys: continue
                base, r, nc = boot(A, B, keys)
                print(f"  δ{d} {name:6s}(대화 {nc:3d}): " + " · ".join(f"{m} {base[m]:5.2f} {v:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, (v, lo, hi) in r.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
