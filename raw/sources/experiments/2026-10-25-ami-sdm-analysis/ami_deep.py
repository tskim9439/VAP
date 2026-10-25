# AMI 오류 세분(2026-10-25, 사용자 "ami sdm 의 수치를 20 미만으로 내리고 싶은데 평가 결과를 세밀하게 분석해보고 알려줘"): 창 평가 원출력
# (cst_stage3_eval — ami-bench-v1 창 20–90 s, 4 슬롯 선언)을 평가와 같은 규칙(window_ref·parse_emitted·score_window)으로 다시 풀어 단어 단위로 오류를 나눈다.
# 행마다 n_ref·cp·agn 이 원출력과 같은지, 정렬 오류 수가 agn 과 같은지 대조한다. 읽기만 한다(산출 json 은 --out).
#   PYTHONPATH=/soundai/users/tskim/VAPKT python ami_deep.py <결과 폴더> [--deltas 10] [--workers 16] [--out <json>]
# 나누는 축(모두 공식 dev·eval × 조건 × δ 마다 참조 단위 합으로):
#   S·D·I — 화자 무시 정렬(cst_eval.align, agn 과 같은 비용). 가설/참조 단위 비.
#   군말 — FILL 단위의 참조 비율·그 단위의 S+D, 군말을 양쪽에서 뺀 agn·cp.
#   겹침 — 참조 단위 시각에 다른 참조 화자도 말하는가(busy_of 의 발화 구간). 겹침/홀로 단위의 S+D 율과 오류 몫.
#   창 가장자리 — 창 처음·끝 EDGE s 안 참조 단위의 S+D 율(가운데와 비교).
#   창 화자 수 1–4 — cp·agn·귀속과 단어 몫.
#   슬롯 쓰임 — 창의 참조 화자 수 × 가설이 쓴 슬롯 수. 짝지은 단위(일치·대치)의 가설 슬롯 × 참조 화자 표로 순도(가설 슬롯마다 주 화자 몫,
#     낮으면 여러 화자를 한 슬롯에 합침)·포괄(참조 화자마다 주 슬롯 몫, 낮으면 한 화자를 여러 슬롯에 나눔).
#   화자 말 몫 — 창 안 단어 몫(주 ≥ 50 %, 중 20–50 %, 소 5–20 %, 극소 < 5 %)별 cp 짝 오류율(score_window 의 순열로 짝지은 비용).
#   귀속 오류 유형 — cst_eval.attr_errors(cst_attr_errors 와 같은 호출, 짧은 run ≤ 3 단어) 단위 수, 겹침/홀로.
#   회의 — 회의별 cp·agn·귀속.
import argparse, collections, glob, json, os, sys
from itertools import permutations

sys.path.insert(0, "/soundai/users/tskim/VAPKT")
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
FILL = {"um", "uh", "hmm", "mm", "mmm", "mhm", "hm", "er", "erm", "ah", "eh", "huh", "uhm", "mm-hmm", "uh-huh", "uh-oh"}   # ami_breakdown.py 와 같다
EDGE = 3.0
G = {}


def share_bucket(x):
    return "주(≥50%)" if x >= .5 else "중(20–50%)" if x >= .2 else "소(5–20%)" if x >= .05 else "극소(<5%)"


def one(r):
    from experiments.cst_stage3_eval import window_ref, words_of
    from experiments.cst_attr_errors import busy_of
    from vapasr.data.cst_eval import align, attr_errors, ed, parse_emitted, score_window, units
    c, i = G["index"][r["id"]]; ds = G["sets"][c]; tok = G["tok"]; cid, t0, L = ds.items[i]; d = ds.dlgs[cid]; lang = d.lang
    ref_w, _, mask, sp = window_ref(ds, i, r["delta"], G["eot"], tok)
    hs, _ = parse_emitted([tuple(e) for e in r["emitted"]], G["spk"], G["eot"], drop_chunks=mask); hyp_w = [words_of(tok, h) for h in hs]
    sc = score_window(ref_w, hyp_w, lang); same = all(sc[k] == r[k] for k in ("n_ref", "cp", "agn"))
    busy, _, _ = busy_of(ds.eps[cid], sp, t0, L)
    RU = [(u, t, s) for w, t, s in sorted([(w, t, s) for s, ws in enumerate(ref_w) for w, t in ws], key=lambda x: x[1]) for u in units(w, lang)]
    HU = [(u, t, s) for w, t, s in sorted([(w, t, s) for s, ws in enumerate(hyp_w) for w, t in ws], key=lambda x: x[1]) for u in units(w, lang)]
    al = align([x[0] for x in RU], [x[0] for x in HU]); st = [None] * len(RU); S = D = I = 0; M = collections.Counter()
    for a, b in al:
        if a is None: I += 1
        elif b is None: D += 1; st[a] = "D"
        else:
            st[a] = "C" if RU[a][0] == HU[b][0] else "S"; S += st[a] == "S"; M[(HU[b][2], RU[a][2])] += 1
    same = same and S + D + I == sc["agn"]
    other = lambda t, s: any(a <= t <= b for k, iv in enumerate(busy) if k != s for a, b in iv)
    out = collections.Counter(n_ref=sc["n_ref"], cp=sc["cp"], agn=sc["agn"], slot=sc["slot"], S=S, D=D, I=I, n_hyp=len(HU))
    for (u, t, s), x in zip(RU, st):
        bad = x in ("S", "D"); ov = other(t, s); edge = t < EDGE or t > L - EDGE
        out["ov_units" if ov else "solo_units"] += 1; out["ov_err" if ov else "solo_err"] += bad
        out["edge_units" if edge else "mid_units"] += 1; out["edge_err" if edge else "mid_err"] += bad
        if u in FILL: out["fill_ref"] += 1; out["fill_err"] += bad
    out["fill_hyp"] = sum(u in FILL for u, _, _ in HU)
    out["agn_nf"] = ed([u for u, _, _ in RU if u not in FILL], [u for u, _, _ in HU if u not in FILL])
    n = max(2, len(ref_w), len(hyp_w)); U = lambda ws: [u for w, _ in ws for u in units(w, lang)]
    Rs = [U(ref_w[k]) if k < len(ref_w) else [] for k in range(n)]; Hs = [U(hyp_w[k]) if k < len(hyp_w) else [] for k in range(n)]
    nf = lambda xs: [x for x in xs if x not in FILL]
    cnf = [[ed(nf(Rs[a]), nf(Hs[b])) for b in range(n)] for a in range(n)]
    out["cp_nf"] = min(sum(cnf[p[h]][h] for h in range(n)) for p in permutations(range(n))); out["n_ref_nf"] = sum(len(nf(x)) for x in Rs)
    cost = [[ed(Rs[a], Hs[b]) for b in range(n)] for a in range(n)]; perm = sc["perm"]
    spk = []
    for h in range(n):
        a = perm[h]
        if a < len(ref_w) and Rs[a]: spk.append((share_bucket(len(Rs[a]) / sc["n_ref"]), len(Rs[a]), cost[a][h]))
    n_ref_spk = sum(1 for x in Rs if x); n_hyp_sl = sum(1 for x in Hs if x)
    ae = attr_errors(ref_w, hyp_w, lang, perm, busy, 3); same = same and ae.get("agn") == sc["agn"]
    return dict(key=(r["corpus"], SUB[cid.split(":", 1)[1]], r["delta"]), meeting=cid.split(":", 1)[1], same=same, cnt=dict(out), spk=spk,
                cells=(n_ref_spk, n_hyp_sl), M=[[h, a, v] for (h, a), v in M.items()], ae_units=dict(ae["units"]), ae_mis=ae["n_mis"], ae_pair=ae["n_pair"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10])
    ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1"); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import multiprocessing as mp
    from transformers import AutoTokenizer
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import SLOTS
    conf = json.load(open(os.path.join(a.pred, "config-rank0.json")))
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.pred, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["delta"] in a.deltas]
    tok = AutoTokenizer.from_pretrained(conf["checkpoint"]); ck = json.load(open(os.path.join(conf["checkpoint"], "config.json"))); sp = ck["sp_ids"]
    w = conf.get("window", (20.0, 40.0))
    ns = argparse.Namespace(corpora=conf["corpora"], data=a.data, split=None, deltas=a.deltas, limit_windows=conf["limit_windows"],
                            backchannel=tuple(conf["backchannel"]) if conf.get("backchannel") else None, max_speakers=conf.get("max_speakers", 2),
                            window=w if w == "full" else tuple(w), hop=conf.get("hop", 10.0), window_draws=conf.get("window_draws", 0), declare=conf.get("declare"))
    sets, items = windows_of(ns, tok)
    G.update(sets=sets, tok=tok, eot=sp["<EOT>"], spk=tuple(sp[t] for t in SLOTS[: len(ck.get("slot_tokens") or SLOTS[:2])]),
             index={f"{sets[c].items[i][0]}@{sets[c].items[i][1]}": (c, i) for c, i in items})
    print(f"원출력 {len(rows)} 행(δ {a.deltas}) · 모델 슬롯 {len(G['spk'])} · 워커 {a.workers}", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool: res = list(pool.imap_unordered(one, rows, chunksize=4))
    bad = [x for x in res if not x["same"]]; assert not bad, f"원출력과 점수가 다른 창 {len(bad)} 개"
    agg = collections.defaultdict(lambda: dict(cnt=collections.Counter(), spk=collections.defaultdict(lambda: [0, 0]), cells=collections.defaultdict(lambda: [0, 0, 0, 0]),
                                                M=collections.Counter(), ae=collections.Counter(), mis=0, pair=0, meet=collections.defaultdict(collections.Counter),
                                                nspk=collections.defaultdict(collections.Counter)))
    for x in res:
        g = agg[x["key"]]; c = x["cnt"]; g["cnt"].update(c); g["mis"] += x["ae_mis"]; g["pair"] += x["ae_pair"]; g["ae"].update(x["ae_units"])
        for b, n_, e_ in x["spk"]: g["spk"][b][0] += n_; g["spk"][b][1] += e_
        cell = g["cells"][tuple(x["cells"])]; cell[0] += 1; cell[1] += c["n_ref"]; cell[2] += c["cp"]; cell[3] += c["agn"]
        for h, r_, v in x["M"]: g["M"][(h, r_)] += v                                                # 창마다 슬롯 번호가 따로라 표는 창 안에서만 뜻이 있다 — 순도·포괄은 창마다 셈
        g["meet"][x["meeting"]].update(n_ref=c["n_ref"], cp=c["cp"], agn=c["agn"])
        g["nspk"][x["cells"][0]].update(n_ref=c["n_ref"], cp=c["cp"], agn=c["agn"], win=1)
    # 순도·포괄은 창 단위로(같은 슬롯 번호라도 창이 다르면 다른 화자일 수 있다)
    pur = collections.defaultdict(lambda: [0, 0, 0])
    for x in res:
        Mx = collections.Counter({(h, r_): v for h, r_, v in x["M"]}); tot = sum(Mx.values())
        if not tot: continue
        hs_, rs_ = {h for h, _ in Mx}, {r_ for _, r_ in Mx}
        p = pur[x["key"]]; p[0] += sum(max(Mx[(h, r_)] for r_ in rs_) for h in hs_); p[1] += sum(max(Mx[(h, r_)] for h in hs_) for r_ in rs_); p[2] += tot
    out = {}
    for k in sorted(agg):
        g = agg[k]; c = g["cnt"]; n = c["n_ref"]; pc = lambda v, m=n: round(100 * v / max(1, m), 2)
        o = dict(windows=sum(v[0] for v in g["cells"].values()), n_ref=n, cp=pc(c["cp"]), agn=pc(c["agn"]), attr=pc(c["cp"] - c["agn"]), slot=pc(c["slot"]),
                 S=pc(c["S"]), D=pc(c["D"]), I=pc(c["I"]), hyp_ref_ratio=round(c["n_hyp"] / max(1, n), 3),
                 fill_ref_share=pc(c["fill_ref"]), fill_hyp_per_ref=pc(c["fill_hyp"]), fill_err_rate=pc(c["fill_err"], c["fill_ref"]),
                 agn_nofill=pc(c["agn_nf"], c["n_ref_nf"]), cp_nofill=pc(c["cp_nf"], c["n_ref_nf"]),
                 overlap_share=pc(c["ov_units"]), err_rate_overlap=pc(c["ov_err"], c["ov_units"]), err_rate_solo=pc(c["solo_err"], c["solo_units"]),
                 err_share_overlap=pc(c["ov_err"], c["ov_err"] + c["solo_err"]),
                 edge_share=pc(c["edge_units"]), err_rate_edge=pc(c["edge_err"], c["edge_units"]), err_rate_mid=pc(c["mid_err"], c["mid_units"]),
                 purity=round(pur[k][0] / max(1, pur[k][2]), 3), coverage=round(pur[k][1] / max(1, pur[k][2]), 3),
                 mis_share_of_pairs=round(g["mis"] / max(1, g["pair"]), 4),
                 by_nspk={s: dict(word_share=pc(v["n_ref"]), cp=pc(v["cp"], v["n_ref"]), agn=pc(v["agn"], v["n_ref"]), attr=pc(v["cp"] - v["agn"], v["n_ref"]), win=v["win"])
                          for s, v in sorted(g["nspk"].items())},
                 slot_cells={f"참조{a}·가설{b}": dict(win=v[0], word_share=pc(v[1]), cp=pc(v[2], v[1]), attr=pc(v[2] - v[3], v[1])) for (a, b), v in sorted(g["cells"].items())},
                 by_speaker_share={b: dict(word_share=pc(v[0]), cp_err_rate=pc(v[1], v[0]), err_share=pc(v[1], c["cp"])) for b, v in sorted(g["spk"].items())},
                 attr_types={t: pc(v) for t, v in sorted(g["ae"].items(), key=lambda kv: -kv[1])},
                 meetings={m: dict(cp=pc(v["cp"], v["n_ref"]), agn=pc(v["agn"], v["n_ref"]), n_ref=v["n_ref"]) for m, v in sorted(g["meet"].items(), key=lambda kv: -kv[1]["cp"] / max(1, kv[1]["n_ref"]))})
        out["/".join(map(str, k))] = o
        print(f"\n== {k[0]} {k[1]} δ{k[2]} · 창 {o['windows']} · 참조 단위 {n}")
        print(f"  cp {o['cp']} = agn {o['agn']}(S {o['S']} · D {o['D']} · I {o['I']}) + 귀속 {o['attr']} · slot {o['slot']} · 가설/참조 {o['hyp_ref_ratio']}")
        print(f"  군말: 참조의 {o['fill_ref_share']} % · 가설 {o['fill_hyp_per_ref']} % · 군말 단위 S+D {o['fill_err_rate']} % → 군말 뺀 agn {o['agn_nofill']} · cp {o['cp_nofill']}")
        print(f"  겹침: 참조의 {o['overlap_share']} % · S+D 겹침 {o['err_rate_overlap']} % 대 홀로 {o['err_rate_solo']} % · S+D 의 {o['err_share_overlap']} % 가 겹침")
        print(f"  가장자리(처음·끝 {EDGE:g} s): 참조의 {o['edge_share']} % · S+D {o['err_rate_edge']} % 대 가운데 {o['err_rate_mid']} %")
        print(f"  슬롯: 순도 {o['purity']} · 포괄 {o['coverage']} · 짝지은 단위 중 오배정 {o['mis_share_of_pairs']}")
        print("  창 화자 수: " + " · ".join(f"{s}인 {v['word_share']}% cp {v['cp']} agn {v['agn']} 귀속 {v['attr']}" for s, v in o["by_nspk"].items()))
        print("  슬롯 쓰임(참조 화자·가설 슬롯): " + " · ".join(f"{c_} 단어 {v['word_share']}% cp {v['cp']} 귀속 {v['attr']}" for c_, v in o["slot_cells"].items() if v["word_share"] >= 1))
        print("  화자 말 몫: " + " · ".join(f"{b} 단어 {v['word_share']}% 짝 오류율 {v['cp_err_rate']} (cp 오류의 {v['err_share']}%)" for b, v in o["by_speaker_share"].items()))
        print("  귀속 오류 유형(pt): " + " · ".join(f"{t} {v}" for t, v in list(o["attr_types"].items())[:8]))
        mm = list(o["meetings"].items()); print("  회의 cp 높은 순: " + " · ".join(f"{m} {v['cp']}/{v['agn']}" for m, v in mm[:5]) + " … 낮은 순: " + " · ".join(f"{m} {v['cp']}/{v['agn']}" for m, v in mm[-3:]))
    if a.out: json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
