import json, sys
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [(n, json.load(open(f"{R}/{d}/summary.json"))) for n, d in [x.split("=") for x in sys.argv[1:]]]
keys = sorted(runs[0][1])
print("코퍼스/δ".ljust(20), " | ".join(f"{n:^38s}" for n, _ in runs))
print("".ljust(20), " | ".join(f"{'slot':>6s} {'cp':>6s} {'agn':>6s} {'EOT R':>6s} {'P':>5s} {'hold':>5s}" for _ in runs))
for k in keys:
    row = []
    for n, s in runs:
        v = s.get(k)
        row.append(f"{v['err_slot']:6.2f} {v['err_cp']:6.2f} {v['err_agn']:6.2f} {v['eot_recall']:6.3f} {v['eot_precision']:5.3f} {v['eot_hold_fire']:5.3f}" if v else " " * 38)
    print(k.ljust(20), " | ".join(row), f"({runs[0][1][k]['unit']})")
