# p12 vs p11 같은 step: 화자 귀속 오류 유형(results/stage3-attr-errors-*/summary.json). 읽기만 한다.
import json, os
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("p11@2000", "st3-p11-revive-ck2000"), ("p12@2000", "st3-p12-revive2-ck2000"), ("p11@3000", "st3-p11-revive-ck3000"), ("p12@3000", "st3-p12-revive2-ck3000"),
        ("p11 final", "st3-p11-revive-final"), ("p12 final", "st3-p12-revive2-final")]
S = {n: json.load(open(f"{R}/stage3-attr-errors-{d}/summary.json"))["results"] for n, d in runs if os.path.exists(f"{R}/stage3-attr-errors-{d}/summary.json")}
for d in ("d10", "d4"):
    print(f"[오류 유형 {d}, 단위 가중]".ljust(20), " ".join(f"{n:>9s}" for n in S))
    rows = [("cp", lambda x: x["cp"]), ("agn", lambda x: x["agn"]), ("cp−agn", lambda x: x["cp"] - x["agn"])]
    rows += [(t, (lambda t: lambda x: x["types"][t]["pt"])(t)) for t in ("whole_long", "whole_short", "late", "early", "inner")]
    rows += [("통째 홀로", lambda x: x["whole_iso_pt"]), ("통째 이어짐", lambda x: x["whole_streak_pt"]), ("가설/참조 교대", lambda x: x["change_ratio"])]
    for name, f in rows: print(name.ljust(20), " ".join(f"{f(s[d]):9.2f}" for s in S.values()))
for k in ("d10/KO", "d10/EN", "d10/DE", "d10/교대/분 ≥20", "d10/겹침 ≥5 %"):
    print(k.ljust(20), " ".join(f"{s[k]['cp']:4.1f}/{s[k]['cp'] - s[k]['agn']:4.1f}" for s in S.values()), " (cp/귀속 몫)")
