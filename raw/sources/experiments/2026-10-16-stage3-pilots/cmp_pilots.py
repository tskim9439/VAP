# 파일럿 비교(δ 별): 코퍼스마다 cp·agn·EOT 재현율·정밀도·보류 오방출. s4 는 KO·DE = 재채점본, otoSpeech = 고친 참조(oto3).
import json, sys
R = "/soundai/users/tskim/VAPKT-data/results"; d = sys.argv[1]
def L(p): return json.load(open(f"{R}/{p}/summary.json"))
s4 = L("stage3-eval-s4-20261015-rescored"); s4.update(L("stage3-eval-s4-oto3-20261015"))
runs = [("s4", s4)] + [(n, L(f"stage3-eval-st3-{n}-20261016")) for n in ("p1b-base", "p2-hard", "p3-bc", "p4-lang", "p5-rep")]
order = ["aihub134-2", "aihub134-1", "aihub71631", "otoSpeech", "kec", "ztc", "nsc"]
for metric, fmt in (("err_cp", "{:6.1f}"), ("err_agn", "{:6.1f}"), ("err_slot", "{:6.1f}"), ("eot_recall", "{:6.3f}"), ("eot_precision", "{:6.3f}"), ("eot_hold_fire", "{:6.3f}")):
    print(f"\n[{metric} δ{d}]".ljust(18), " ".join(f"{n:>9s}" for n, _ in runs))
    tot = {n: [] for n, _ in runs}
    for c in order:
        k = f"{c}/d{d}"; row = []
        for n, s in runs:
            v = s.get(k, {}).get(metric); row.append(f"{fmt.format(v):>9s}" if v is not None else f"{'-':>9s}")
            if v is not None: tot[n].append(v)
        print(c.ljust(17), " ".join(row))
    print("평균".ljust(17), " ".join(f"{fmt.format(sum(v) / len(v)):>9s}" if v else f"{'-':>9s}" for v in tot.values()))
