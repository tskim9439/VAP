# E3 대 E2 손실 나누기(mxc 호스트, 2026-10-26): AMI δ10 창마다 E3(예측 활동) cp − E2(정답 활동) cp 를, E1 이 창 화자 수를 적게·맞게·많게 셌는지로 나눠
#   참조 단어 대비 pt 로 낸다. ssh mxc python3 - < e3_vs_e2.py(json 만 — 시스템 python 으로 돈다).
import collections, glob, json
R = "/soundai/users/tskim/VAPKT-data/results"
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json")); SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
rows = lambda d: {(r["id"], r["delta"]): r for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl") for r in map(json.loads, open(f, encoding="utf-8"))}
e2, e3 = rows("stage3-eval-amitgt-q17-e2-tgt-final"), rows("stage3-eval-amie3-t0.45-q17-e2-tgt-final")
for corpus in ("ami-sdm", "ami-ihmmix"):
    for sub in ("eval", "dev"):
        ks = [k for k in e3 if k[1] == 10 and k[0].startswith(corpus + ":") and SUB[k[0].split(":", 1)[1].split("@")[0]] == sub]
        nref = sum(e2[k]["n_ref"] for k in ks); g = collections.defaultdict(lambda: [0, 0.0, 0])
        for k in ks:
            nh, ns = len(e3[k].get("hyp_slots") or []), e3[k]["n_spk"]; cls = "적게" if nh < ns else "맞음" if nh == ns else "많게"
            g[cls][0] += 1; g[cls][1] += e3[k]["cp"] - e2[k]["cp"]; g[cls][2] += e2[k]["n_ref"]
        tot = sum(v[1] for v in g.values())
        print(f"{corpus} {sub} δ10: E3−E2 = {100 * tot / nref:+.2f} pt · " + " · ".join(f"화자 수 {c} 창 {v[0]}: {100 * v[1] / nref:+.2f} pt({100 * v[1] / tot:.0f} %)" for c, v in sorted(g.items())))
