import json, os, glob
R = "/soundai/users/tskim/VAPKT-data/runs/semcommit/eval/v035-snap0929-d8"
def g(d, *ks):
    for k in ks:
        if d is None: return None
        d = d.get(k) if isinstance(d, dict) else None
    return d
def f(x, n=3): return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
rows = []
for s in ("ks-eval", "ks-long", "ls-test", "gs-test"):
    for d in (2, 3, 4, 6, 8):
        p = f"{R}/{s}-d{d}.json"
        if not os.path.exists(p): print(s, d, "missing"); continue
        r = json.load(open(p))
        best = None
        for name, c in r["configs"].items():
            o = c["overall"]; t2 = g(o, "timing", "late2"); tx = o["text"]
            asr = o.get("asr") or {}; ak = next(iter(asr)) if asr else None
            rec = dict(set=s, d=d, cfg=name, f1=t2["f1"], P=t2["precision"], R=t2["recall"], lat50=g(t2, "latency_s", "p50"), lat90=g(t2, "latency_s", "p90"),
                       tP=tx["precision"], tR=tx["recall"], tF=tx["f1"], pcr=tx["pcr"], bcr=tx.get("B_commit_rate"), hn=tx.get("hardneg_commit_rate"),
                       semmin=g(o, "events", "sem_per_min"), asr=(ak, g(asr, ak, "rate")) if ak else None)
            rows.append(rec)
            if name == "bias=0": base = rec
            if best is None or (rec["tF"] or 0) > (best["tF"] or 0): best = rec
        for tag, x in (("bias0", base), ("best", best)):
            print(f"{s:8s} δ{d} {tag:5s} {x['cfg']:8s} | win2 P {f(x['P'])} R {f(x['R'])} F1 {f(x['f1'])} | text P {f(x['tP'])} R {f(x['tR'])} F1 {f(x['tF'])} PCR {f(x['pcr'])} hardneg {f(x['hn'])} | lat p50 {f(x['lat50'],2)} p90 {f(x['lat90'],2)} s | SEM/min {f(x['semmin'],1)} | {x['asr'][0] if x['asr'] else ''} {f(x['asr'][1]) if x['asr'] else ''}")
json.dump(rows, open(f"{R}/summary-rows.json", "w"), indent=1)
