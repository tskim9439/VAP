import glob, json, numpy as np
R = "/soundai/users/tskim/VAPKT-data/results"
def rows(d):
    out = {}
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["dataset"] == "fleurs-ko-en": out[(str(r["delta"]), r["id"])] = r
    return out
a, b = rows("ast-fleurs-q06-s3-ast"), rows("ast-fleurs-q06-s4-ast")
m = lambda r: r["metrics"][r["primary_metric"]]
for dl in ("4", "10"):
    ids = sorted(k[1] for k in b if k[0] == dl and k in a)
    for nm, rr in (("s3", a), ("s4", b)):
        e = np.array([m(rr[(dl, i)])["errors"] for i in ids]); n = np.array([m(rr[(dl, i)])["n_ref"] for i in ids])
        c = e / np.maximum(n, 1); hl = [len(rr[(dl, i)]["hypothesis"]) / max(1, len(rr[(dl, i)]["reference"])) for i in ids]
        tk = np.mean([len(rr[(dl, i)]["token_ids"]) for i in ids])
        print(dl, nm, "pooled CER %.2f · CER>50%% %d · hyp/ref len>1.5 %d · tokens/utt %.1f" % (100 * e.sum() / n.sum(), int((c > 0.5).sum()), sum(x > 1.5 for x in hl), tk))
    worst = sorted(ids, key=lambda i: m(b[(dl, i)])["errors"] - m(a[(dl, i)])["errors"])[-2:]
    for i in worst:
        print("  ", i, "| s3:", a[(dl, i)]["hypothesis"][:80], "| s4:", b[(dl, i)]["hypothesis"][:80])
