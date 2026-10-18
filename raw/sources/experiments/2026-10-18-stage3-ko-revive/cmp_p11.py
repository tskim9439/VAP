import json, os
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("r1@2000", "q06-st3-r1-ck2000"), ("p10@2000", "st3-p10-sim-ck2000"), ("p11@2000", "st3-p11-revive-ck2000"), ("p10@3000", "st3-p10-sim-ck3000"),
        ("p11@3000", "st3-p11-revive-ck3000"), ("p10 final", "st3-p10-sim-final"), ("p11 final", "st3-p11-revive-final")]
S = {n: json.load(open(f"{R}/stage3-eval-{d}/summary.json")) for n, d in runs if os.path.exists(f"{R}/stage3-eval-{d}/summary.json")}      # 끝난 것만
for d in (10, 4):
    print(f"[δ{d} 코퍼스 평균]".ljust(14), " ".join(f"{n:>9s}" for n in S))
    for m, f in (("err_cp", "{:9.2f}"), ("err_agn", "{:9.2f}"), ("eot_recall", "{:9.3f}"), ("eot_precision", "{:9.3f}"), ("eot_hold_fire", "{:9.3f}")):
        print(m.ljust(14), " ".join(f.format(sum(s[k][m] for k in s if k.endswith(f"/d{d}")) / len([k for k in s if k.endswith(f"/d{d}")])) for s in S.values()))
print("[δ10 코퍼스별 cp · agn]")
for c in ("aihub134-2", "aihub134-1", "aihub71631", "otoSpeech", "kec", "ztc", "nsc"):
    print(c.ljust(14), " ".join(f"{s[f'{c}/d10']['err_cp']:5.1f}·{s[f'{c}/d10']['err_agn']:4.1f}" for s in S.values()))
