#!/usr/bin/env python
"""Stage 3 화자 귀속 오류 유형 분해 — cst_stage3_eval 의 원출력(predictions-rank*.jsonl)을 디코드 없이 다시 읽어 오배정 단위를 유형·겹침별로 센다.

  python experiments/cst_attr_errors.py --pred <eval out dir> --data <phase2 dir> --split <v2.json> --out <dir> [--deltas 4,10] [--workers 16]

창·참조는 그 평가와 같다(config-rank0.json 의 corpora·limit_windows·backchannel, cst_stage3_eval.window_ref). 창마다 다시 낸 n_ref·cp·agn 이
원출력의 값과 같은지 대조하고, 하나라도 다르면 요약 없이 끝낸다.
유형(vapasr/data/cst_eval.attr_errors): whole_short·whole_long(턴 통째) · late(교대 늦음) · early(교대 이름) · inner(헛교대) × 겹침 ov / 단독 solo.
출력: <out>/windows.jsonl(창·δ 별), <out>/summary.json, 표준 출력 표(오배정 단위 = 참조 단위 대비 pt)."""
import argparse, collections, glob, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.cst_stage3_eval import windows_of, window_ref, words_of

LANG = {"Korean": "KO", "English": "EN", "German": "DE"}
G = {}                                                    # fork 워커가 물려받는 전역(셋·tokenizer·특수 토큰)


def busy_of(eps, speakers, t0, L):
    """슬롯별 말하는 구간(창 기준 s)과 창 특성(겹침 비율, 분당 화자 교대)."""
    t1 = t0 + L; E = sorted([e for e in eps if e.end > t0 and e.start < t1 and e.speaker in speakers], key=lambda e: e.start)
    busy = [[(max(e.start, t0) - t0, min(e.end, t1) - t0) for e in E if e.speaker == s] for s in speakers] + [[]] * (2 - len(speakers))
    ov = sum(max(0.0, min(b0, b1) - max(a0, a1)) for a0, b0 in busy[0] for a1, b1 in busy[1])
    return busy, ov / L, sum(a.speaker != b.speaker for a, b in zip(E, E[1:])) / (L / 60)


def one(job):
    from vapasr.data.cst_eval import parse_emitted, score_window, attr_errors
    r = job; c, i = G["index"][r["id"]]; ds = G["sets"][c]; tok = G["tok"]; d = ds.dlgs[ds.items[i][0]]
    ref_w, _, mask, sp_order = window_ref(ds, i, r["delta"], G["eot"], tok)
    hs, _ = parse_emitted([tuple(e) for e in r["emitted"]], G["spk"], G["eot"], drop_chunks=mask)
    hyp_w = [words_of(tok, hs[0]), words_of(tok, hs[1])]
    sc = score_window(ref_w, hyp_w, d.lang); same = all(sc[k] == r[k] for k in ("n_ref", "cp", "agn"))
    busy, ov, chg = busy_of(ds.eps[ds.items[i][0]], sp_order, ds.items[i][1], ds.items[i][2])
    ae = attr_errors(ref_w, hyp_w, d.lang, sc["perm"], busy, G["short_words"]); same = same and ae.pop("agn") == sc["agn"] and ae["n_ref"] == sc["n_ref"]
    return dict(id=r["id"], delta=r["delta"], corpus=r["corpus"], lang=LANG[d.lang], n_spk=r["n_spk"], cp=sc["cp"], agn=sc["agn"], same=same,
                stored=[r["n_ref"], r["cp"], r["agn"]], overlap=round(ov, 4), changes_per_min=round(chg, 2), **ae)


def bucket(w):
    c = w["changes_per_min"]; o = w["overlap"]
    return ("교대/분 " + ("<5" if c < 5 else "5–10" if c < 10 else "10–20" if c < 20 else "≥20"),
            "겹침 " + ("0 %" if o == 0 else "<5 %" if o < .05 else "≥5 %"))


def table(rows, title):
    from vapasr.data.cst_eval import ATTR_TYPES
    n = sum(w["n_ref"] for w in rows) or 1; mis = sum(w["n_mis"] for w in rows) or 1; u = collections.Counter()
    for w in rows: u.update(w["units"])
    by_t = {t: u[f"{t}/ov"] + u[f"{t}/solo"] for t in ATTR_TYPES}
    rs = sum(w["runs_short"] for w in rows); rl = sum(w["runs"] - w["runs_short"] for w in rows); lw = [x for w in rows for x in w["late_words"]]
    res = dict(windows=len(rows), ref_units=n, cp=100 * sum(w["cp"] for w in rows) / n, agn=100 * sum(w["agn"] for w in rows) / n,
               mislabel=100 * sum(w["n_mis"] for w in rows) / n, ov_share=sum(v for k, v in u.items() if k.endswith("/ov")) / mis,
               types={t: dict(pt=100 * by_t[t] / n, share=by_t[t] / mis, ov=u[f"{t}/ov"] / max(1, by_t[t])) for t in ATTR_TYPES},
               whole_short_rate=sum(w["whole_short_runs"] for w in rows) / max(1, rs), whole_long_rate=sum(w["whole_long_runs"] for w in rows) / max(1, rl),
               short_run_share=rs / max(1, rs + rl), whole_iso_pt=100 * sum(w["whole_iso_units"] for w in rows) / n, whole_streak_pt=100 * sum(w["whole_streak_units"] for w in rows) / n,
               whole_streak_edge=sum(w["whole_streak_edge_units"] for w in rows) / max(1, sum(w["whole_streak_units"] for w in rows)), change_ratio=sum(w["hyp_changes"] for w in rows) / max(1, sum(w["ref_changes"] for w in rows)),
               late_words_median=statistics.median(lw) if lw else None, late_words_p90=sorted(lw)[int(0.9 * (len(lw) - 1))] if lw else None,
               late_one_word=sum(x == 1 for x in lw) / max(1, len(lw)))
    print(f"[{title}] 창 {len(rows)} · cp {res['cp']:.1f} · agn {res['agn']:.1f} · cp−agn {res['cp'] - res['agn']:.1f} · 오배정 {res['mislabel']:.1f} pt"
          f"(겹침 안 {100 * res['ov_share']:.0f} %) · 가설/참조 교대 {res['change_ratio']:.2f}")
    print("   " + " · ".join(f"{t} {v['pt']:.1f} pt({100 * v['share']:.0f} %, 겹침 {100 * v['ov']:.0f} %)" for t, v in res["types"].items()))
    print(f"   짧은 run(≤{G['short_words']} 단어) 통째 오배정 {100 * res['whole_short_rate']:.1f} %(run 의 {100 * res['short_run_share']:.0f} %) · "
          f"긴 run 통째 {100 * res['whole_long_rate']:.1f} % · 통째 홀로 {res['whole_iso_pt']:.1f} pt · 이어짐 {res['whole_streak_pt']:.1f} pt(창 끝에 닿음 {100 * res['whole_streak_edge']:.0f} %) · 늦은 교대 단어 중앙값 {res['late_words_median']} · p90 {res['late_words_p90']} · 1 단어 {100 * res['late_one_word']:.0f} %")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--data", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 10])
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--short-words", type=int, default=3); a = ap.parse_args()
    import multiprocessing as mp
    from transformers import AutoTokenizer
    conf = json.load(open(os.path.join(a.pred, "config-rank0.json")))
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.pred, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
    rows = [r for r in rows if r["delta"] in a.deltas]
    tok = AutoTokenizer.from_pretrained(conf["checkpoint"]); sp = json.load(open(os.path.join(conf["checkpoint"], "config.json")))["sp_ids"]
    ns = argparse.Namespace(corpora=conf["corpora"], data=a.data, split=a.split, deltas=a.deltas, limit_windows=conf["limit_windows"],
                            backchannel=tuple(conf["backchannel"]) if conf.get("backchannel") else None)
    sets, items = windows_of(ns, tok)
    G.update(sets=sets, tok=tok, eot=sp["<EOT>"], spk=(sp["<SPK_A>"], sp["<SPK_B>"]), short_words=a.short_words,
             index={f"{sets[c].items[i][0]}@{sets[c].items[i][1]}": (c, i) for c, i in items})
    miss = [r["id"] for r in rows if r["id"] not in G["index"]]; assert not miss, f"평가 창에 없는 원출력 {len(miss)} 개: {miss[:3]}"
    print(f"원출력 {len(rows)} 행(δ {a.deltas}) · 워커 {a.workers}", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool: out = list(pool.imap_unordered(one, rows, chunksize=4))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "windows.jsonl"), "w", encoding="utf-8") as f:
        for w in sorted(out, key=lambda w: (w["delta"], w["corpus"], w["id"])): f.write(json.dumps(w, ensure_ascii=False) + "\n")
    bad = [w for w in out if not w["same"]]
    if bad:
        for w in bad[:5]: print(f"  !! 불일치 {w['id']} δ{w['delta']}: 다시 냄 {[w['n_ref'], w['cp'], w['agn']]} / 원출력 {w['stored']}")
        sys.exit(f"원출력과 점수가 다른 창 {len(bad)} 개 — 참조 규칙이 평가와 다르다")
    print(f"대조: {len(out)} 창·δ 모두 n_ref·cp·agn 이 원출력과 같다", flush=True)
    summary = {}
    for dl in a.deltas:
        R = [w for w in out if w["delta"] == dl]; summary[f"d{dl}"] = table(R, f"δ{dl} 전체")
        for g in ("KO", "EN", "DE"): summary[f"d{dl}/{g}"] = table([w for w in R if w["lang"] == g], f"δ{dl} {g}")
        for k in sorted({b for w in R for b in bucket(w)}): summary[f"d{dl}/{k}"] = table([w for w in R if k in bucket(w)], f"δ{dl} {k}")
    json.dump(dict(pred=os.path.abspath(a.pred), short_words=a.short_words, results=summary), open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
