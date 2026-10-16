# p6·p7 비교: 대화 평가(맞장구 참조, δ10·δ4 평균과 코퍼스별 cp) + Stage 2 축약 평가 요약.
import json, os
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("p1b", "stage3-eval-st3-p1b-base-20261016-bcref"), ("p3", "stage3-eval-st3-p3-bc-20261016-bcref"), ("p4", "stage3-eval-st3-p4-lang-20261016-bcref"),
        ("p6", "stage3-eval-st3-p6-combo-20261016"), ("p7", "stage3-eval-st3-p7-rep75-20261016")]
S = {n: json.load(open(f"{R}/{d}/summary.json")) for n, d in runs if os.path.exists(f"{R}/{d}/summary.json")}
for d in (10, 4):
    print(f"[δ{d} 평균]".ljust(10), " ".join(f"{n:>8s}" for n in S))
    for m, f in (("err_cp", "{:8.1f}"), ("err_agn", "{:8.1f}"), ("eot_recall", "{:8.3f}"), ("eot_precision", "{:8.3f}"), ("eot_hold_fire", "{:8.3f}")):
        vals = []
        for n, s in S.items():
            ks = [k for k in s if k.endswith(f"/d{d}")]; vals.append(f.format(sum(s[k][m] for k in ks) / len(ks)))
        print(m.ljust(10), " ".join(vals))
print("[δ10 cp 코퍼스별]")
for c in ("aihub134-2", "aihub134-1", "aihub71631", "otoSpeech", "kec", "ztc", "nsc"):
    print(c.ljust(12), " ".join(f"{S[n].get(f'{c}/d10', {}).get('err_cp', float('nan')):8.1f}" for n in S))
