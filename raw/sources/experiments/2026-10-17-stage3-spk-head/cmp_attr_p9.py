import json
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("r1@2000", "q06-st3-r1-ck2000-v3"), ("p9@2000", "st3-p9-spkhead-ck2000"), ("p9@3000", "st3-p9-spkhead-ck3000"), ("r1 final", "q06-st3-r1-final-v3"), ("p9 final", "st3-p9-spkhead-final")]
S = {n: json.load(open(f"{R}/stage3-attr-errors-{d}/summary.json"))["results"] for n, d in runs}
for d in ("d10", "d4"):
    print(f"[오류 유형 {d}]".ljust(18), " ".join(f"{n:>9s}" for n in S))
    rows = [("cp", lambda x: x["cp"]), ("agn", lambda x: x["agn"]), ("cp−agn", lambda x: x["cp"] - x["agn"]), ("오배정", lambda x: x["mislabel"])]
    rows += [(t, (lambda t: lambda x: x["types"][t]["pt"])(t)) for t in ("whole_long", "whole_short", "late", "early", "inner")]
    rows += [("통째 홀로", lambda x: x["whole_iso_pt"]), ("통째 이어짐", lambda x: x["whole_streak_pt"]), ("이어짐 끝 닿음", lambda x: 100 * x["whole_streak_edge"]),
             ("가설/참조 교대", lambda x: x["change_ratio"]), ("짧은 run 통째%", lambda x: 100 * x["whole_short_rate"]), ("긴 run 통째%", lambda x: 100 * x["whole_long_rate"])]
    for name, f in rows: print(name.ljust(18), " ".join(f"{f(s[d]):9.2f}" for s in S.values()))
for k in ("d10/KO", "d10/EN", "d10/DE", "d10/교대/분 ≥20", "d10/교대/분 10–20", "d10/겹침 0 %", "d10/겹침 ≥5 %"):
    print(k.ljust(18), " ".join(f"{s[k]['cp']:5.1f}/{s[k]['mislabel']:4.1f}/{s[k]['whole_streak_pt']:4.1f}" for s in S.values()), " (cp/오배정/이어짐)")
