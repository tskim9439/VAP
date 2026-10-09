import gzip, json, glob, collections
R = "/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/fix-val-gen-v3-bf16"
for f in sorted(glob.glob(R + "/*/u-*.jsonl.gz")):
    c = collections.Counter(); n = 0
    for l in gzip.open(f, "rt", encoding="utf-8"):
        x = json.loads(l); n += 1
        if x["pass_v2"]: continue
        c["v2fail"] += 1
        j = x["judge_v3"] or {}
        if x["pass_v3"]: c[f"flip_{j.get('content')}"] += 1
        if x["final"] == "repair": c["repair_ok"] += 1
        if "repair_translation" in x: c["repaired"] += 1
    rep_rate = c["repair_ok"] / max(1, c["repaired"])
    cur = n - c["v2fail"] + c["flip_same"] + c["flip_minor"] + c["repair_ok"]
    strict_est = n - c["v2fail"] + c["flip_same"] + c["repair_ok"] + rep_rate * c["flip_minor"]
    print(f.split("fix-val-gen-v3-bf16/")[1], dict(c), f"units {n} · v2 {1 - c['v2fail']/n:.3f} · current {cur/n:.3f} · strict(est) {strict_est/n:.3f} · repair success {rep_rate:.2f}")
