"""받아쓰기 오류(agn — 화자 무시) 분해(2026-11-04, 사용자 "agn 을 없애려면 어떻게 개선할 지 고민해봐. 종단간 모델구조를 사용할거야").
서버 호스트 CPU, 디코드 없음, 원 결과 폴더는 읽기만. 회의 통째(또는 창) 원출력을 참조와 화자 무시로 짝짓는다(gate_diag 와 같은 시각 — 가설 프레임 = 방출 청크 − 1 − δ,
같은 글자 ±tol 가장 가까운 것부터 하나씩). Whisper 정규화가 지우는 채움말(hmm·mm·mhm·mmm·uh·um)은 양쪽에서 뺀다.
  참조 단어 → 맞음 | 대치꼴(그 근처 ±tol 에 짝 없는 가설 단어가 있다) | 삭제꼴. 짝 없는 가설 단어 중 대치꼴에 안 쓰인 것 → 삽입꼴(참조 무음 · 말하는 중).
  참조 단어를 나누는 축: 겹침(그 프레임에 말하는 참조 화자 수 — episode 구간, 자기 포함) · episode 길이 · episode 안 위치(첫 단어 · 2–3 · 나머지)
    · 그 화자의 회의 안 말 몫(주 화자 · 나머지). episode 통째 놓침(그 episode 단어를 하나도 못 냄)도 센다.
  python agn_diag.py <결과 폴더>… [--tol 0.6] [--json 출력.json]   (같은 평가 설정의 폴더 여럿 — 참조는 한 번만 만든다)
"""
import argparse, bisect, collections, glob, json, math, os, re, sys
import numpy as np

REPO = "/soundai/users/tskim/VAPKT"
CHUNK = 0.08
FILL = {"hmm", "mm", "mhm", "mmm", "uh", "um"}
OV = ("1", "2", "3+")
ULEN = ((1.0, "<1s"), (3.0, "1-3s"), (10.0, "3-10s"), (1e9, ">=10s"))
POS = ("첫", "2-3", "나머지")
AXES = (("ov", OV), ("ulen", tuple(n for _, n in ULEN)), ("pos", POS), ("spk", ("주 화자", "나머지")))
norm = lambda w: re.sub(r"[^\w']+", "", w.lower())


def keep(w):
    n = norm(w); return n if n and n not in FILL else None


def match(ref, hyp, tol):
    """ref [(글자, 프레임)], hyp [(글자, 프레임)] → (ref 마다 짝 hyp index 또는 None, hyp 마다 짝 여부). 같은 글자 ±tol, 가장 가까운 것부터."""
    pool = collections.defaultdict(list)
    for j, (w, f) in enumerate(hyp): pool[w].append((f, j))
    for v in pool.values(): v.sort()
    used = set(); out = []
    for w, f in ref:
        v = pool.get(w, []); i = bisect.bisect_left(v, (f - tol, -1)); best = None
        while i < len(v) and v[i][0] <= f + tol:
            if v[i][1] not in used and (best is None or abs(v[i][0] - f) < abs(v[best][0] - f)): best = i
            i += 1
        if best is None: out.append(None)
        else: used.add(v[best][1]); out.append(v[best][1])
    return out, [j in used for j in range(len(hyp))]


def build_ref(E, ds, i, delta, eot, tok):
    """참조 단어(채움말 뺌)와 축 값, episode 길이, 말하는 프레임(±0.32 s 넓힘), 마스크."""
    cid, t0w, L = ds.items[i]
    ref_w, _, mask, sp_order = E.window_ref(ds, i, delta, eot, tok)
    eps = [e for e in ds.eps[cid] if e.end > t0w and e.start < t0w + L]
    K = int(math.ceil(L / CHUNK)) + 2; names = sorted({e.speaker for e in eps}); col = {n: k for k, n in enumerate(names)}
    A = np.zeros((K, max(1, len(names))), bool); per = collections.defaultdict(list)
    for j, e in enumerate(eps):
        a_, b_ = max(0, int((e.start - t0w) / CHUNK)), min(K, int(math.ceil((e.end - t0w) / CHUNK)) + 1)
        A[a_:b_, col[e.speaker]] = True; per[e.speaker].append((e.start - t0w, e.end - t0w, j))
    for v in per.values(): v.sort()
    starts = {n: [s_ for s_, _, _ in v] for n, v in per.items()}
    R = []
    for s, sl in enumerate(ref_w):
        name = sp_order[s] if s < len(sp_order) else None
        for w, t in sl:
            n = keep(w)
            if n is None: continue
            f = min(int(t / CHUNK), K - 1); ep = -1
            if name in per:
                k = bisect.bisect_right(starts[name], t + 0.05) - 1
                if k >= 0 and per[name][k][1] >= t - 0.3: ep = per[name][k][2]
            own = col.get(name); ov = int(A[f].sum()) + (0 if own is not None and A[f, own] else 1)
            R.append(dict(w=n, f=f, s=s, ep=ep, ov=ov, t=t))
    pos = collections.Counter()
    for q in sorted(R, key=lambda q: q["t"]):
        q["pos"] = pos[q["ep"]] if q["ep"] >= 0 else -1; pos[q["ep"]] += 1
    share = collections.Counter(q["s"] for q in R); top = share.most_common(1)[0][0] if share else -1
    sp = np.convolve(A.any(1).astype(np.int32), np.ones(9, np.int32), "same") > 0          # 말하는 프레임 ±4(0.32 s)
    return R, {j: e.end - e.start for j, e in enumerate(eps)}, top, mask, sp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("preds", nargs="+"); ap.add_argument("--tol", type=float, default=0.6); ap.add_argument("--json")
    x = ap.parse_args()
    sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
    import cst_stage3_eval as E
    from transformers import AutoTokenizer
    from vapasr.data.cst_eval import parse_emitted
    from vapasr.data.cst_dialogue import SLOTS
    man = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
    SUB = {**{m: "dev" for m in man["dev"]}, **{m: "eval" for m in man["eval"]}}
    cfg = json.load(open(os.path.join(x.preds[0], "config-rank0.json"))); tok = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    a = argparse.Namespace(split=None, corpora=cfg["corpora"], data="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1", deltas=cfg["deltas"],
                           backchannel=tuple(cfg["backchannel"]), max_speakers=cfg.get("max_speakers"), declare=cfg.get("declare"),
                           window=cfg["window"] if cfg["window"] == "full" else tuple(cfg["window"]), hop=cfg.get("hop", 10.0),
                           window_draws=cfg.get("window_draws", 0), context_s=cfg.get("context_s", 0.0), limit_windows=0, xl_order="lang")
    sets, todo = E.windows_of(a, tok)
    tol_f = int(round(x.tol / CHUNK)); out = {}; refs = {}
    pc_ = lambda u, v: 100.0 * u / max(1, v)
    for pred in x.preds:
        pc = json.load(open(os.path.join(pred, "config-rank0.json"))); cc = json.load(open(os.path.join(pc["checkpoint"], "config.json"))); sp = cc["sp_ids"]
        assert (pc["corpora"], pc["deltas"], pc["window"]) == (cfg["corpora"], cfg["deltas"], cfg["window"]), f"평가 설정이 다르다: {pred}"
        spk = tuple(sp[t] for t in SLOTS[:4]); EOT = sp["<EOT>"]; BEGIN = sp.get("<ONSET>"); XI = int(sp["<SPK_5>"]) if cc.get("x_slot") else -1
        spk_p = spk + ((XI,) if XI >= 0 else ())
        rows = {}
        for f in glob.glob(os.path.join(pred, "predictions-rank*.jsonl")):
            for l in open(f, encoding="utf-8"): r = json.loads(l); rows[(r["id"], r["delta"])] = r
        agg = collections.defaultdict(collections.Counter); meet = []
        for c, i in todo:
            ds = sets[c]; cid, t0w, L = ds.items[i]
            for delta in cfg["deltas"]:
                r = rows.get((f"{cid}@{t0w}", delta))
                if r is None: continue
                C = agg[(c, SUB[cid.split(":", 1)[1].split("@")[0]])]
                if (c, i, delta) not in refs: refs[(c, i, delta)] = build_ref(E, ds, i, delta, EOT, tok)
                R, epl, top, mask, speech = refs[(c, i, delta)]
                hs, _ = parse_emitted(r["emitted"], spk_p, EOT, drop_chunks=mask, begin_id=BEGIN)
                H = [(n, int(round(t / CHUNK)) - 1 - int(delta)) for h in hs for w, t in E.words_of(tok, h) for n in [keep(w)] if n]
                mref, mhyp = match([(q["w"], q["f"]) for q in R], H, tol_f)
                free = sorted(H[j][1] for j in range(len(H)) if not mhyp[j]); taken = [False] * len(free); kinds = []
                for q, m in zip(R, mref):
                    kind = "ok"
                    if m is None:
                        k = bisect.bisect_left(free, q["f"] - tol_f)
                        while k < len(free) and free[k] <= q["f"] + tol_f and taken[k]: k += 1
                        if k < len(free) and free[k] <= q["f"] + tol_f: taken[k] = True; kind = "sub"
                        else: kind = "del"
                    kinds.append(kind)
                    el = epl.get(q["ep"])
                    vals = dict(ov=OV[min(q["ov"], 3) - 1], ulen=next(n for b, n in ULEN if el < b) if el is not None else None,
                                pos=None if q["pos"] < 0 else POS[0] if q["pos"] == 0 else POS[1] if q["pos"] <= 2 else POS[2], spk="주 화자" if q["s"] == top else "나머지")
                    for ax, v in vals.items():
                        if v is not None: C[f"{ax}:{v}:n"] += 1; C[f"{ax}:{v}:{kind}"] += 1
                    if vals["ulen"] is not None:                                                    # 겹침 × episode 길이(2026-11-04 — 겹친 말 놓침이 짧은 맞장구인가)
                        jk = f"j:{vals['ov']}|{vals['ulen']}"; C[jk + ":n"] += 1; C[jk + ":" + kind] += 1
                    C["ref"] += 1; C[kind] += 1
                byep = collections.defaultdict(list)
                for q, kd in zip(R, kinds): byep[q["ep"]].append(kd == "ok")
                for j, v in byep.items():
                    if j < 0: continue
                    ul = next(n for b, n in ULEN if epl[j] < b); C[f"ep:{ul}:n"] += 1
                    if not any(v): C[f"ep:{ul}:lost"] += 1; C[f"ep:{ul}:lost_words"] += len(v)
                ins = [fr for fr, tk in zip(free, taken) if not tk]
                C["hyp"] += len(H); C["ins"] += len(ins); C["ins_sil"] += sum(1 for fr in ins if not speech[min(max(fr, 0), len(speech) - 1)])
                n_del = kinds.count("del"); n_sub = kinds.count("sub")
                meet.append(dict(corpus=c, id=cid, ref=len(R), dele=n_del, sub=n_sub, ins=len(ins)))
        out[pred] = dict(agg={f"{c}/{s}": dict(v) for (c, s), v in agg.items()}, meetings=meet)
        print(f"== {pred} — 화자 무시 짝(같은 글자 ±{x.tol} s, 채움말 뺌)")
        for (c, sub), v in sorted(agg.items()):
            n = v["ref"]; miss = n - v["ok"]
            print(f"  {c}/{sub}: 참조 {n} · 가설 {v['hyp']} · 근사 오류 {pc_(v['del'] + v['sub'] + v['ins'], n):.2f} %"
                  f"(삭제꼴 {pc_(v['del'], n):.2f} · 대치꼴 {pc_(v['sub'], n):.2f} · 삽입꼴 {pc_(v['ins'], n):.2f} — 그중 참조 무음 {pc_(v['ins_sil'], n):.2f})")
            for ax, vals in AXES:
                parts = []
                for val in vals:
                    nk = v.get(f"{ax}:{val}:n", 0)
                    if nk:
                        mk = nk - v.get(f"{ax}:{val}:ok", 0)
                        parts.append(f"{val} {pc_(nk, n):.1f} %(놓침 {pc_(mk, nk):.1f} · 삭제꼴 {pc_(v.get(f'{ax}:{val}:del', 0), nk):.1f} · 놓친 단어의 {pc_(mk, miss):.0f} %)")
                print(f"    {ax:4s} " + " · ".join(parts))
            parts = []
            for _, ul in ULEN:
                ne = v.get(f"ep:{ul}:n", 0)
                if ne: parts.append(f"{ul} {ne} 개 중 통째 놓침 {pc_(v.get(f'ep:{ul}:lost', 0), ne):.1f} %(그 단어 = 참조의 {pc_(v.get(f'ep:{ul}:lost_words', 0), n):.2f} %)")
            print("    episode " + " · ".join(parts))
            print("    겹침×길이(놓침 % · 놓친 단어의 몫 %): " + " · ".join(f"{o}|{u} {pc_(v.get(f'j:{o}|{u}:n', 0) - v.get(f'j:{o}|{u}:ok', 0), v.get(f'j:{o}|{u}:n', 0)):.0f}·{pc_(v.get(f'j:{o}|{u}:n', 0) - v.get(f'j:{o}|{u}:ok', 0), miss):.1f}"
                                                         for o in OV for _, u in ULEN if v.get(f"j:{o}|{u}:n", 0)))
        worst = sorted(meet, key=lambda m: -(m["dele"] + m["sub"] + m["ins"]) / max(1, m["ref"]))[:6]
        print("  근사 오류율 높은 회의: " + " · ".join(f"{m['id']} {100 * (m['dele'] + m['sub'] + m['ins']) / max(1, m['ref']):.1f}" for m in worst))
    if x.json: json.dump(out, open(x.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
