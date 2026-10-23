import json, glob, collections
R = "/soundai/users/tskim/VAPKT-data/results"
for n in ("st3-p11-revive-ck3000", "st3-p11-revive-noeot-ck3000", "v3-p1-tel-xlw-final", "v3-p1-tel-xlw-noeot-final"):
    rows = [json.loads(l) for f in glob.glob(f"{R}/stage3-eval-{n}/predictions-rank*.jsonl") for l in open(f)]
    rows = [r for r in rows if int(r["delta"]) == 10]; by = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for r in rows:
        x = by[r["corpus"]]; x[0] += r["n_ref"]; x[1] += r["n_hyp"]; x[2] += r["agn"]; x[3] += r["cp"]; x[4] += int(r.get("forced", 0) > 0); x[5] += 1
    print(f"== {n}")
    for c, x in sorted(by.items()):
        print(f"  {c:11s} agn {100 * x[2] / x[0]:5.1f} · cp {100 * x[3] / x[0]:5.1f} · 가설/참조 {x[1] / x[0]:.2f} · 강제 진행 창 {x[4]}/{x[5]}")
