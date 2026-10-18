# 창 마스크 비율 구간별 귀속 몫 — stage3 평가 원출력(masked_chunks·K)과 attr_errors windows.jsonl 을 (id, δ)로 붙인다.
import json, sys
from collections import defaultdict
ev, at = sys.argv[1], sys.argv[2]
mk = {}
for l in open(ev):
    r = json.loads(l); mk[(r["id"], r["delta"])] = (r["masked_chunks"] / max(1, r["K"]), r.get("n_spk"))
bins = [(0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, 1.01)]
agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # n_win, n_ref, cp, agn, streak_units, ref_changes
miss = 0
for l in open(at):
    r = json.loads(l)
    if r["delta"] != 10 or r.get("n_spk") != 2: continue
    m = mk.get((r["id"], r["delta"]))
    if m is None: miss += 1; continue
    b = next(i for i, (lo, hi) in enumerate(bins) if lo <= m[0] < hi)
    for key in (r["lang"], r["corpus"]):
        a = agg[(key, b)]; a[0] += 1; a[1] += r["n_ref"]; a[2] += r["cp"]; a[3] += r["agn"]; a[4] += r.get("whole_streak_units", 0); a[5] += r.get("ref_changes", 0)
print(f"join 실패 {miss}")
print(f"{'키':<12}{'마스크 구간':<12}{'창':>5}{'n_ref':>8}{'cp%':>7}{'agn%':>7}{'귀속몫':>7}{'이어짐%':>8}{'교대/창':>8}{'오배정/교대':>10}")
for key in sorted({k for k, _ in agg}):
    for b, (lo, hi) in enumerate(bins):
        a = agg.get((key, b))
        if not a or a[0] < 5: continue
        n = a[1]; print(f"{key:<12}{f'{lo:.0%}–{min(hi,1):.0%}':<12}{a[0]:>5}{n:>8}{100*a[2]/n:>7.1f}{100*a[3]/n:>7.1f}{100*(a[2]-a[3])/n:>7.1f}{100*a[4]/n:>8.1f}{a[5]/a[0]:>8.1f}{(a[2]-a[3])/max(1,a[5]):>10.2f}")
