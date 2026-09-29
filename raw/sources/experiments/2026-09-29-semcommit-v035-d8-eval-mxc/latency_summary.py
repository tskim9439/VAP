"""골드 재채점(latency/*.json) 요약: 세트 × δ, bias 0 과 세트·δ 별 text F1 최대 bias — 창 없는 commit 지연·표시 지연·after_text."""
import json, os, sys
R = "/soundai/users/tskim/VAPKT-data/runs/semcommit/eval/v035-snap0929-d8/latency"
f = lambda x, n=3: "  -  " if x is None else f"{x:.{n}f}"
rows = []
print(f"{'set':8s} {'δ':>2s} {'cfg':>8s} | {'txtF1':>5s} {'PCR':>5s} | {'nA':>4s} {'cov':>5s} {'exact':>5s} {'c50':>5s} {'c90':>5s} {'c95':>5s} {'r@.32':>5s} {'r@.64':>5s} {'r@1':>5s} {'exc90':>5s} | "
      f"{'d50':>5s} {'d90':>5s} {'ftl50':>5s} | {'at.n':>5s} {'lag0':>5s} {'lag≥2':>5s} {'flush':>5s} | wk")
for s in ("ks-eval", "ks-long", "ls-test", "gs-test"):
    for d in (2, 3, 4, 6, 8):
        p = f"{R}/{s}-d{d}.json"
        if not os.path.exists(p): print(s, d, "missing"); continue
        r = json.load(open(p)); wk = r["scoring"].get("word_k")
        best = max(r["configs"].items(), key=lambda kv: kv[1]["overall"]["text"]["f1"] or 0)[0]
        for cfg in dict.fromkeys(("bias=0", best)):
            o = r["configs"][cfg]["overall"]; t = o["text"]; c = o.get("commit") or {}; cl = c.get("latency_s") or {}; ra = c.get("recall_at") or {}
            dp = o.get("display") or {}; dl = dp.get("latency_s") or {}; at = o.get("after_text") or {}
            rec = dict(set=s, d=d, cfg=cfg, best=cfg == best, tF=t["f1"], pcr=t["pcr"], nA=c.get("n_A"), cov=c.get("coverage"), exact=c.get("exact_rate"),
                       c50=cl.get("p50"), c90=cl.get("p90"), c95=cl.get("p95"), c99=cl.get("p99"), r32=ra.get("0.32"), r64=ra.get("0.64"), r1=ra.get("1"),
                       exc90=(c.get("excess_chunks") or {}).get("p90"), d50=dl.get("p50"), d90=dl.get("p90"), ftl50=(dp.get("first_token_s") or {}).get("p50"),
                       at_n=at.get("n"), lag0=at.get("zero_rate"), lag2=at.get("ge2_rate"), at_flush=at.get("in_flush"), disp_missing=dp.get("missing"))
            rows.append(rec)
            print(f"{s:8s} {d:2d} {cfg:>8s} | {f(rec['tF'])} {f(rec['pcr'])} | {rec['nA'] or 0:4d} {f(rec['cov'])} {f(rec['exact'])} {f(rec['c50'],2)}  {f(rec['c90'],2)}  {f(rec['c95'],2)}  "
                  f"{f(rec['r32'])} {f(rec['r64'])} {f(rec['r1'])} {f(rec['exc90'],1)}   | {f(rec['d50'],2)}  {f(rec['d90'],2)}  {f(rec['ftl50'],2)}  | {rec['at_n'] or 0:5d} {f(rec['lag0'])} {f(rec['lag2'])} "
                  f"{rec['at_flush'] or 0:5d} | {json.dumps(wk, ensure_ascii=False)[:60] if cfg == 'bias=0' else ''}")
json.dump(rows, open(f"{R}/latency-summary-rows.json", "w"), indent=1)
