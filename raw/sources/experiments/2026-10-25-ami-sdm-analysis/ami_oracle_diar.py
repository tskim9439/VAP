# E0 — 정답 화자 구간으로 다시 귀속(2026-10-25, 화자 분리 통합 계획 [[output-diarization-integration-20261025]] 의 첫 시험, 학습 없음): 창 평가 원출력의
# 가설 단어를 시각으로 정답 화자 구간(busy_of — 발화 구간)에 붙여 슬롯을 다시 정하고 같은 규칙(score_window)으로 잰다.
# 한 스트림 형식에서 화자 분리가 완벽할 때 남는 귀속 오류(겹침 모호성·시각 어긋남)를 어림한다. agn 은 슬롯과 무관해 그대로다. 읽기만 한다.
#   PYTHONPATH=/soundai/users/tskim/VAPKT python ami_oracle_diar.py <결과 폴더> [--deltas 10] [--tol 0.2] [--workers 16] [--out <json>]
# 다시 붙이기: 지연 lag = 창의 정확히 일치한 단어쌍 (가설 − 참조) 시각 중앙값(일치가 5 개 미만이면 전체 중앙값). 가설 단어 시각 t − lag 에
#   말하던 참조 화자(구간 ± tol)가 하나면 그 화자, 여럿(겹침)이면 모델이 고른 화자(score_window 순열로 짝지은 참조)가 그중에 있으면 그대로, 없으면
#   구간 시작이 가장 늦은 화자, 아무도 없으면 구간까지 시각 거리가 가장 가까운 화자.
import argparse, collections, glob, json, os, statistics, sys

sys.path.insert(0, "/soundai/users/tskim/VAPKT")
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
G = {}


def prep(r):
    from experiments.cst_stage3_eval import window_ref, words_of
    from experiments.cst_attr_errors import busy_of
    from vapasr.data.cst_eval import align, parse_emitted, score_window, units
    c, i = G["index"][r["id"]]; ds = G["sets"][c]; tok = G["tok"]; cid, t0, L = ds.items[i]; lang = ds.dlgs[cid].lang
    ref_w, _, mask, sp = window_ref(ds, i, r["delta"], G["eot"], tok)
    hs, _ = parse_emitted([tuple(e) for e in r["emitted"]], G["spk"], G["eot"], drop_chunks=mask); hyp_w = [words_of(tok, h) for h in hs]
    sc = score_window(ref_w, hyp_w, lang); assert all(sc[k] == r[k] for k in ("n_ref", "cp", "agn")), r["id"]
    busy, _, _ = busy_of(ds.eps[cid], sp, t0, L)
    RW = sorted([(w, t) for ws in ref_w for w, t in ws], key=lambda x: x[1]); HW = sorted([(w, t) for ws in hyp_w for w, t in ws], key=lambda x: x[1])
    dts = [HW[b][1] - RW[a][1] for a, b in align([w for w, _ in RW], [w for w, _ in HW]) if a is not None and b is not None and RW[a][0] == HW[b][0]]
    return dict(key=(r["corpus"], SUB[cid.split(":", 1)[1]], r["delta"]), ref_w=ref_w, hyp_w=hyp_w, busy=busy, perm=sc["perm"], lang=lang,
                n_ref=sc["n_ref"], cp=sc["cp"], agn=sc["agn"], dts=dts)


def reassign(x, lag, tol):
    from vapasr.data.cst_eval import score_window
    busy = x["busy"]; n = len(busy); new = [[] for _ in range(n)]; cnt = collections.Counter()
    dist = lambda s, t: min((max(a - t, 0.0, t - b) for a, b in busy[s]), default=1e9)
    for h, ws in enumerate(x["hyp_w"]):
        own = x["perm"][h] if h < len(x["perm"]) and x["perm"][h] < n else None
        for w, t in ws:
            tt = t - lag; cand = [s for s in range(n) if any(a - tol <= tt <= b + tol for a, b in busy[s])]
            if len(cand) == 1: s = cand[0]; cnt["one"] += 1
            elif cand:
                cnt["overlap"] += 1
                if own in cand: s = own; cnt["overlap_own"] += 1
                else: s = max(cand, key=lambda s: max(a for a, b in busy[s] if a - tol <= tt <= b + tol))
            else: s = min(range(n), key=lambda s: dist(s, tt)) if n else 0; cnt["none"] += 1
            new[s].append((w, t))
    sc = score_window(x["ref_w"], new, x["lang"])
    return sc["cp"], cnt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("--deltas", type=lambda s: [int(v) for v in s.split(",")], default=[10])
    ap.add_argument("--tol", type=float, default=0.2); ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--out", default=None); a = ap.parse_args()
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
    with mp.get_context("fork").Pool(a.workers) as pool: X = list(pool.imap_unordered(prep, rows, chunksize=4))
    glag = {k: statistics.median([d for x in X if x["key"] == k for d in x["dts"]] or [0.0]) for k in {x["key"] for x in X}}
    agg = collections.defaultdict(collections.Counter); out = {}
    for x in X:
        lag = statistics.median(x["dts"]) if len(x["dts"]) >= 5 else glag[x["key"]]
        cpo, cnt = reassign(x, lag, a.tol); g = agg[x["key"]]
        g.update(n_ref=x["n_ref"], cp=x["cp"], agn=x["agn"], cp_oracle=cpo, **cnt)
    for k in sorted(agg):
        g = agg[k]; n = g["n_ref"]; hw = g["one"] + g["overlap"] + g["none"]; pc = lambda v, m=n: round(100 * v / max(1, m), 2)
        o = dict(n_ref=n, lag_median=round(glag[k], 3), cp=pc(g["cp"]), agn=pc(g["agn"]), attr=pc(g["cp"] - g["agn"]), cp_oracle_diar=pc(g["cp_oracle"]),
                 attr_oracle_diar=pc(g["cp_oracle"] - g["agn"]), hyp_words=hw, hyp_one=pc(g["one"], hw), hyp_overlap=pc(g["overlap"], hw),
                 hyp_overlap_kept_own=pc(g["overlap_own"], max(1, g["overlap"])), hyp_none=pc(g["none"], hw))
        out["/".join(map(str, k))] = o
        print(f"{k[0]:11s} {k[1]:4s} δ{k[2]:<2d} lag {o['lag_median']:.2f} s · cp {o['cp']} = agn {o['agn']} + 귀속 {o['attr']} → 정답 구간으로 다시 귀속 cp {o['cp_oracle_diar']}"
              f"(귀속 {o['attr_oracle_diar']}) · 가설 단어: 한 화자 {o['hyp_one']} % · 겹침 {o['hyp_overlap']} %(모델 화자 유지 {o['hyp_overlap_kept_own']} %) · 구간 밖 {o['hyp_none']} %", flush=True)
    if a.out: json.dump(dict(tol=a.tol, results=out), open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
