import json
R = "/soundai/users/tskim/VAPKT-data/results"
for run in ("p1b-base", "p2-hard", "p3-bc", "p4-lang", "p5-rep"):
    s = json.load(open(f"{R}/stage3-eval-st3-{run}-20261016-bcref/summary.json")); ks = [k for k in s if k.endswith("/d10")]
    a = lambda m: sum(s[k][m] for k in ks) / len(ks)
    print(f"{run:9s} δ10 맞장구 참조: cp {a('err_cp'):.1f} · EOT R {a('eot_recall'):.3f} P {a('eot_precision'):.3f} hold {a('eot_hold_fire'):.3f}")
