import json, os, glob, random, collections
R = "/soundai/users/tskim/VAPKT-data/results"
runs = [("p9ck3000", "stage3-eval-st3-p9-spkhead-ck3000", "stage3-attr-errors-st3-p9-spkhead-ck3000"), ("guide-ins", "stage3-eval-p9ck3000-guide-ins", "stage3-attr-errors-p9ck3000-guide-ins"),
        ("guide-hold", "stage3-eval-p9ck3000-guide-hold", "stage3-attr-errors-p9ck3000-guide-hold"),
        ("hold-only", "stage3-eval-p9ck3000-guide-holdonly", "stage3-attr-errors-p9ck3000-guide-holdonly")]
runs = [r for r in runs if os.path.exists(f"{R}/{r[1]}/summary.json") and os.path.exists(f"{R}/{r[2]}/summary.json")]
S = {n: json.load(open(f"{R}/{e}/summary.json")) for n, e, _ in runs}
for d in (10, 4):
    print(f"[δ{d} 코퍼스 평균]".ljust(14), " ".join(f"{n:>11s}" for n in S))
    for m, f in (("err_cp", "{:11.2f}"), ("err_agn", "{:11.2f}"), ("eot_recall", "{:11.3f}"), ("eot_precision", "{:11.3f}"), ("eot_hold_fire", "{:11.3f}")):
        print(m.ljust(14), " ".join(f.format(sum(s[k][m] for k in s if k.endswith(f"/d{d}")) / len([k for k in s if k.endswith(f"/d{d}")])) for s in S.values()))
print("[δ10 코퍼스별 cp · agn]")
for c in ("aihub134-2", "aihub134-1", "aihub71631", "otoSpeech", "kec", "ztc", "nsc"):
    print(c.ljust(14), " ".join(f"{s[f'{c}/d10']['err_cp']:6.1f}·{s[f'{c}/d10']['err_agn']:4.1f}" for s in S.values()))
A = {n: json.load(open(f"{R}/{a}/summary.json"))["results"] for n, _, a in runs}
for d in ("d10", "d4"):
    print(f"[오류 유형 {d}, 단위 가중]".ljust(20), " ".join(f"{n:>11s}" for n in A))
    for name, f in (("cp", lambda x: x["cp"]), ("agn", lambda x: x["agn"]), ("cp−agn", lambda x: x["cp"] - x["agn"]), ("긴 턴 통째", lambda x: x["types"]["whole_long"]["pt"]),
                    ("짧은 턴 통째", lambda x: x["types"]["whole_short"]["pt"]), ("늦은 교대", lambda x: x["types"]["late"]["pt"]), ("이른 교대", lambda x: x["types"]["early"]["pt"]),
                    ("헛교대", lambda x: x["types"]["inner"]["pt"]), ("이어짐", lambda x: x["whole_streak_pt"]), ("가설/참조 교대", lambda x: x["change_ratio"])):
        print(name.ljust(20), " ".join(f"{f(s[d]):11.2f}" for s in A.values()))
# 안내 발동 수
for n, e, _ in runs[1:]:
    rows = [json.loads(l) for f in glob.glob(f"{R}/{e}/predictions-rank*.jsonl") for l in open(f)]
    ins = sum(r.get("guide_insert", 0) for r in rows); hold = sum(r.get("guide_hold", 0) for r in rows)
    print(f"{n}: 창·δ {len(rows)} · 삽입 {ins} (창당 {ins / len(rows):.2f}) · 막음 {hold} · 발동 창 비율 {sum((r.get('guide_insert', 0) + r.get('guide_hold', 0)) > 0 for r in rows) / len(rows):.2f}")
# 짝 bootstrap(대화 단위) — 안내 − 안내 없음
def load(a):
    return {(w["id"], w["delta"]): w for w in map(json.loads, open(f"{R}/{a}/windows.jsonl"))}
W = {n: load(a) for n, _, a in runs}
for n in list(W)[1:]:
    for dl in (10, 4):
        keys = [k for k in W["p9ck3000"] if k[1] == dl and k in W[n]]; by = collections.defaultdict(list)
        for k in keys: by[k[0].rsplit("@", 1)[0]].append(k)
        cv = list(by); rng = random.Random(0)
        def st(ks):
            t = sum(W["p9ck3000"][k]["n_ref"] for k in ks) or 1
            g = lambda r, m: sum(m(W[r][k]) for k in ks) * 100 / t
            return {m: g(n, f) - g("p9ck3000", f) for m, f in (("cp", lambda w: w["cp"]), ("agn", lambda w: w["agn"]), ("cp−agn", lambda w: w["cp"] - w["agn"]), ("이어짐", lambda w: w["whole_streak_units"]))}
        base = st(keys); dr = collections.defaultdict(list)
        for _ in range(2000):
            ks = [k for c in (rng.choice(cv) for _ in cv) for k in by[c]]
            for m, v in st(ks).items(): dr[m].append(v)
        print(f"{n} − p9ck3000 δ{dl}: " + " · ".join(f"{m} {base[m]:+.2f} [{sorted(v)[50]:+.2f}, {sorted(v)[1949]:+.2f}]" for m, v in dr.items()))
