# 1.7B Stage 1(q17-s1m3-d10-s0 — 0.6B q06-s1m3-d10 와 같은 레시피) 단일 발화 ASR 비교(2026-10-24, 평가 job 94840): 2026-10-09 cmp_eval.py 와 같은 계산
# (발화 재표본 bootstrap 1,000 회, Δ = 1.7B − 기준, * = 95 % 구간이 0 을 넘지 않음). 기준: q06 = 같은 레시피 0.6B, q17-r3 = 옛 1.7B Stage 1(q17-s1m-r3, 한·영만).
#   PYTHONPATH 불필요 — python3 cmp_s1.py
import collections, glob, json, random
R = "/soundai/users/tskim/VAPKT-data/results"
NEW = {"ko-en": "single-turn-vpaa-kspon-q17-s1m3-d10-s0", "de": "single-turn-yodas-de-q17-s1m3-d10-s0"}
BASE = {"q06": {"ko-en": "single-turn-vpaa-kspon-q06-s1m3-d10", "de": "single-turn-yodas-de-q06-s1m3-d10"}, "q17-r3": {"ko-en": "single-turn-vpaa-kspon-q17-s1m-r3"}}


def load(d):
    out = collections.defaultdict(dict)                      # (dataset, delta) -> id -> (errors, n_ref)
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f, encoding="utf-8"):
            p = json.loads(l); m = p["metrics"][p["primary_metric"]]
            out[(p["dataset"], str(p["delta"]))][p["id"]] = (m["errors"], m["n_ref"])
    return out


def rate(x): return 100 * sum(e for e, _ in x) / max(1, sum(n for _, n in x))
rng = random.Random(0)
def boot(a, b, B=1000):
    ids = sorted(set(a) & set(b)); diffs = []
    for _ in range(B):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(rate([a[i] for i in s]) - rate([b[i] for i in s]))
    diffs.sort(); return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


for part in ("ko-en", "de"):
    new = load(NEW[part]); bases = {k: load(v[part]) for k, v in BASE.items() if part in v}
    for ds, dl in sorted(new, key=lambda k: (k[0], k[1] != "final", int(k[1]) if k[1].isdigit() else 99)):
        a = new[(ds, dl)]; row = [f"{ds:24s} δ{dl:5s} 1.7B {rate(a.values()):6.2f} (n={len(a)})"]
        for bk, bv in bases.items():
            b = bv.get((ds, dl))
            if not b: row.append(f"{bk} -"); continue
            lo, hi = boot(a, b); row.append(f"{bk} {rate(b.values()):6.2f} Δ[{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}")
        print(" | ".join(row), flush=True)
