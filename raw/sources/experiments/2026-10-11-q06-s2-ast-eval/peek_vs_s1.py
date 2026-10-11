"""Partial paired comparison: q06-s2-ast vs Stage 1 (q06-s1m3-d10) on the same (dataset, δ, id) available so far. Prints only."""
import glob, json, sys, collections
R = "/soundai/users/tskim/VAPKT-data/results"
def load(d):
    out = {}
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l); m = r["metrics"][r["primary_metric"]]
            out[(r["dataset"], str(r["delta"]), r["id"])] = (m["errors"], m["n_ref"], bool(r["token_ids"]))
    return out
for new_d, old_d in (("single-turn-vpaa-kspon-q06-s2-ast", "single-turn-vpaa-kspon-q06-s1m3-d10"), ("single-turn-yodas-de-q06-s2-ast", "single-turn-yodas-de-q06-s1m3-d10"),
                     ("single-turn-librispeech-clean-q06-s2-ast", "single-turn-librispeech-clean-q06-s1m3-d10"), ("single-turn-librispeech-other-q06-s2-ast", "single-turn-librispeech-other-q06-s1m3-d10")):
    a, b = load(old_d), load(new_d)
    cells = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for k, (e, n, nz) in b.items():
        if k in a:
            c = cells[k[:2]]; c[0] += a[k][0]; c[1] += e; c[2] += n; c[3] += 1; c[4] += int(not nz)
    order = {"4": 0, "6": 1, "8": 2, "10": 3, "final": 4}
    for (ds, dl), (ea, eb, n, c, sil) in sorted(cells.items(), key=lambda x: (x[0][0], order[x[0][1]])):
        print(f"{ds:24s} δ{dl:6s} n={c:5d}  Stage1 {ea / n * 100:6.2f}  s2-ast {eb / n * 100:6.2f}  Δ {(eb - ea) / n * 100:+5.2f}  silent(s2)={sil}")
