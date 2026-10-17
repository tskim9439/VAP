import json, os
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("r1@2000", "stage3-eval-q06-st3-r1-ck2000"), ("p8@2000", "stage3-eval-st3-p8-recover-ck2000"), ("r1@3000", "stage3-eval-q06-st3-r1-ck3000"), ("p8@3000", "stage3-eval-st3-p8-recover-ck3000"), ("r1 final", "stage3-eval-q06-st3-r1-final"), ("p8 final", "stage3-eval-st3-p8-recover-final")]
S = {n: json.load(open(f"{R}/{d}/summary.json")) for n, d in runs if os.path.exists(f"{R}/{d}/summary.json")}
for d in (10, 4):
    print(f"[δ{d} 평균]".ljust(12), " ".join(f"{n:>9s}" for n in S))
    for m, f in (("err_cp", "{:9.1f}"), ("err_agn", "{:9.1f}"), ("eot_recall", "{:9.3f}"), ("eot_precision", "{:9.3f}"), ("eot_hold_fire", "{:9.3f}")):
        print(m.ljust(12), " ".join(f.format(sum(s[k][m] for k in s if k.endswith(f"/d{d}")) / len([k for k in s if k.endswith(f"/d{d}")])) for s in S.values()))
print("[δ10 코퍼스별 cp · agn]")
for c in ("aihub134-2", "aihub134-1", "aihub71631", "otoSpeech", "kec", "ztc", "nsc"):
    print(c.ljust(12), " ".join(f"{s[f'{c}/d10']['err_cp']:5.1f}·{s[f'{c}/d10']['err_agn']:4.1f}" for s in S.values()))
