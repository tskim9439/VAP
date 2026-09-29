"""골드 v1(COMMIT/AMBIG/NO, Claude 이중 주석) 기준 채점 — semcommit_gold.py score-eval 과 같은 함수(commit_metrics.gold_commit_counts·finalize_gold_commits),
+ sha1 dev/test 절반(semcommit_gold.split_half: 교사 임계값 튜닝이 dev 절반을 썼다). bias 는 dev 절반 F1 로 고르고 test 절반에서 보고한다."""
import json, sys, importlib.util, os
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
spec = importlib.util.spec_from_file_location("sg", "/soundai/users/tskim/VAPKT/experiments/semcommit_gold.py"); sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)
from vapasr.hf.commit_metrics import gold_commit_counts, finalize_gold_commits, merge, ref_chunk, SEM_END_ID
R = "/soundai/users/tskim/VAPKT-data/runs/semcommit/eval/v035-snap0929-d8"; G1 = "/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1"
res = {}
for s in ("ks-eval", "ks-long", "ls-test", "gs-test"):
    W = sg.words_of([f"{G1}/words-{s}.jsonl"]); G = sg._load_gold(f"{G1}/gold-{s}.jsonl")
    for d in (2, 3, 4, 6, 8):
        agg = {}
        for r in sg.read_jsonl(f"{R}/{s}-d{d}.streams.jsonl"):
            sid = r["id"]
            if sid not in G or sid not in W: continue
            ws = sorted(W[sid]["words"], key=lambda w: int(w["i"])); K = int(r["K"]); dd = int(r["delta"])
            sem = (r.get("event_ids") or [SEM_END_ID])[0]; ev = [e for e in r["hyp"]["events"] if int(e["id"]) == int(sem)]
            c = gold_commit_counts(ws, G[sid], r["hyp"]["words"], ev, [ref_chunk(float(w["end_time"]), dd, K) for w in ws], K)
            name = r["config"]["name"]
            for h in ("all", sg.split_half(sid)): agg[(name, h)] = merge(agg.get((name, h)), c)
        res[f"{s}-d{d}"] = {f"{n}|{h}": finalize_gold_commits(v) for (n, h), v in sorted(agg.items())}
json.dump(res, open(f"{R}/gold-score.json", "w"), indent=1, ensure_ascii=False)
fm = lambda x, n=3: "  -  " if x is None else f"{x:.{n}f}"
print(f"{'set':8s} {'δ':>2s} | {'bias0 all: P':>12s} {'R':>5s} {'F1':>5s} {'PCRg':>5s} {'lat50':>5s} | {'bias0 test: P':>13s} {'R':>5s} {'F1':>5s} | {'dev→bias':>8s} {'test P':>6s} {'R':>5s} {'F1':>5s} {'PCRg':>5s} | gold C")
for key, cf in res.items():
    s, d = key.rsplit("-d", 1)
    a0 = cf["bias=0|all"]; t0 = cf["bias=0|test"]
    devs = {k.split("|")[0]: v for k, v in cf.items() if k.endswith("|dev")}
    b = max(devs, key=lambda n: devs[n]["F1"] or 0); tb = cf[f"{b}|test"]
    print(f"{s:8s} {d:>2s} | {fm(a0['P']):>12s} {fm(a0['R'])} {fm(a0['F1'])} {fm(a0['PCR_gold'])} {fm((a0['latency_s'] or {}).get('p50'),2):>5s} | {fm(t0['P']):>13s} {fm(t0['R'])} {fm(t0['F1'])} | "
          f"{b.replace('bias=',''):>8s} {fm(tb['P']):>6s} {fm(tb['R'])} {fm(tb['F1'])} {fm(tb['PCR_gold'])} | {a0['gold_commit']} (test {t0['gold_commit']})")
