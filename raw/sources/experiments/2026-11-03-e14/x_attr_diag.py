"""E14 X 칸 단어 귀속 진단(2026-11-03 — 회의 통째 IHM-mix eval 에서 X 칸이 놓친 말을 받아썼는데(agn 23.09 → 21.74) cp 가 24.39 → 26.32 로 나빠졌다).
서버 호스트 CPU, 디코드 없음, 원 결과 폴더는 읽기만. gate_diag.py 와 같은 참조·시각(방출 청크 − 1 − δ = 소리 프레임, 같은 글자 ±tol).
  X 단어 → 참조 단어 짝: 되찾음(슬롯 칸이 못 낸 참조 단어) · 겹침(슬롯 칸도 낸 단어) · 헛(짝 없음).
  되찾은 단어마다 정답 슬롯 h = perm⁻¹(참조 화자)와 귀속 규칙(모두 그 시각 앞 활동만 — 스트리밍 그대로)을 견준다.
    cur = 원출력 x_slots(확률 최대, 근거 없으면 직전 활성) · recent = 가장 최근 활성 · never = 아직 한 번도 안 켜진 슬롯(없으면 cur) · stale = 마지막 활성이 가장 오래된 슬롯
  h 의 상태: 그 앞 활동 없음(never) · 마지막 활성에서 지난 시간. 회의마다 X 단어·되찾음·cur 맞음·끝까지 안 켜진 슬롯 수.
  python x_attr_diag.py <결과 폴더> [--tol 0.6] [--json 출력.json]
"""
import argparse, bisect, collections, glob, json, os, re, sys
import numpy as np

REPO = "/soundai/users/tskim/VAPKT"
CHUNK = 0.08
RULES = ("cur", "recent", "never", "stale")
GAPS = ((5.0, "<5s"), (30.0, "5-30s"), (120.0, "30-120s"), (1e9, ">120s"))
norm = lambda w: re.sub(r"[^\w']+", "", w.lower())


def last_on(A):
    """A (K, N) bool → (K, N) int: 프레임 k 앞(< k)에서 그 슬롯이 마지막으로 켜진 프레임(없으면 −1)."""
    K, N = A.shape; out = np.full((K, N), -1, np.int64); cur = np.full(N, -1, np.int64)
    for k in range(K):
        out[k] = cur; cur = np.where(A[k], k, cur)
    return out


class Pool:
    """같은 글자 참조 단어를 프레임 순으로 — 가장 가까운 안 쓴 것을 하나씩 짝짓는다."""
    def __init__(self, ref):
        self.d = collections.defaultdict(list)
        for j, (w, f, _) in enumerate(ref): self.d[w].append((f, j))
        for v in self.d.values(): v.sort()

    def take(self, w, f, tol, used):
        v = self.d.get(w, []); i = bisect.bisect_left(v, (f - tol, -1)); best = None
        while i < len(v) and v[i][0] <= f + tol:
            if v[i][1] not in used and (best is None or abs(v[i][0] - f) < abs(v[best][0] - f)): best = i
            i += 1
        if best is None: return None
        used.add(v[best][1]); return v[best][1]


def rules(lo, cur):
    has = lo >= 0
    recent = int(np.argmax(lo)) if has.any() else 0
    never = int(np.flatnonzero(~has)[0]) if (~has).any() else cur
    return dict(cur=cur, recent=recent, never=never, stale=int(np.argmin(lo)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("--tol", type=float, default=0.6); ap.add_argument("--json")
    x = ap.parse_args()
    sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
    import cst_stage3_eval as E
    from transformers import AutoTokenizer
    from vapasr.data.cst_eval import parse_emitted, act_from_runs
    from vapasr.data.cst_dialogue import SLOTS
    man = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
    SUB = {**{m: "dev" for m in man["dev"]}, **{m: "eval" for m in man["eval"]}}
    cfg = json.load(open(os.path.join(x.pred, "config-rank0.json"))); rows = {}
    for f in glob.glob(os.path.join(x.pred, "predictions-rank*.jsonl")):
        for l in open(f, encoding="utf-8"): r = json.loads(l); rows[(r["id"], r["delta"])] = r
    ck = cfg["checkpoint"]; tok = AutoTokenizer.from_pretrained(ck); cc = json.load(open(os.path.join(ck, "config.json"))); sp = cc["sp_ids"]
    assert cc.get("x_slot"), f"X 칸 모델이 아니다: {ck}"
    spk = tuple(sp[t] for t in SLOTS[:4]); EOT = sp["<EOT>"]; BEGIN = sp.get("<ONSET>"); XI = int(sp["<SPK_5>"])
    a = argparse.Namespace(split=None, corpora=cfg["corpora"], data="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1", deltas=cfg["deltas"],
                           backchannel=tuple(cfg["backchannel"]), max_speakers=cfg.get("max_speakers"), declare=cfg.get("declare"),
                           window=cfg["window"] if cfg["window"] == "full" else tuple(cfg["window"]), hop=cfg.get("hop", 10.0),
                           window_draws=cfg.get("window_draws", 0), context_s=cfg.get("context_s", 0.0), limit_windows=0, xl_order="lang")
    sets, todo = E.windows_of(a, tok)
    tol_f = int(round(x.tol / CHUNK)); agg = collections.defaultdict(collections.Counter); meet = []
    for c, i in todo:
        ds = sets[c]; cid, t0w, L = ds.items[i]
        for delta in cfg["deltas"]:
            r = rows.get((f"{cid}@{t0w}", delta))
            if r is None or "act_runs" not in r: continue
            K = int(round(L / CHUNK)); sub = SUB[cid.split(":", 1)[1].split("@")[0]]; C = agg[(c, sub)]
            ref_w, _, mask, _ = E.window_ref(ds, i, delta, EOT, tok)
            hs, _ = parse_emitted(r["emitted"], spk + (XI,), EOT, drop_chunks=mask, begin_id=BEGIN)
            xw = E.words_of(tok, hs[-1]); xs = r.get("x_slots") or []
            if len(xs) != len(xw): C["x_len_mismatch"] += 1; continue
            A = np.asarray(act_from_runs(r["act_runs"], K), bool); LO = last_on(A)
            inv = {p: h for h, p in enumerate(r["perm"])}
            ref = [(norm(w), int(t / CHUNK), inv.get(s)) for s, sl in enumerate(ref_w) for w, t in sl]
            pool = Pool(ref); by_slot = set()
            for h in hs[: len(spk)]:
                for w, t in E.words_of(tok, h): pool.take(norm(w), int(round(t / CHUNK)) - 1 - int(delta), tol_f, by_slot)
            by_x = set(); M = collections.Counter()
            for (w, t), cur in zip(xw, xs):
                f = min(max(int(round(t / CHUNK)) - 1 - int(delta), 0), K - 1)
                j = pool.take(norm(w), f, tol_f, by_x | by_slot)                                     # 슬롯 칸이 못 낸 참조 단어(쓴 표시는 아래에서 by_x 에)
                if j is None:
                    j = pool.take(norm(w), f, tol_f, set(by_x))                                       # 슬롯 칸이 이미 낸 단어와 겹치나
                    M["dup" if j is not None else "ins"] += 1
                    if j is not None: by_x.add(j)
                    continue
                by_x.add(j); h = ref[j][2]; M["rec"] += 1
                if h is None: M["rec_unmatched_spk"] += 1; continue
                lo = LO[f]; R = rules(lo, int(cur))
                for k, s_ in R.items(): M[f"ok_{k}"] += int(s_ == h)
                if lo[h] < 0: M["h_never"] += 1; M["ok_cur_h_never"] += int(R["cur"] == h); M["ok_never_h_never"] += int(R["never"] == h)
                else:
                    g = (f - lo[h]) * CHUNK; M["h_gap_" + next(n for b, n in GAPS if g < b)] += 1
            M["x"] = len(xw); M["ref"] = len(ref); M["slot_hit"] = len(by_slot)
            never_m = int((~A.any(0)).sum()); M["never_slots"] = never_m
            C.update(M); C["rows"] += 1
            meet.append(dict(corpus=c, sub=sub, id=cid, ref=len(ref), x=len(xw), rec=M["rec"], ok_cur=M["ok_cur"], ok_never=M["ok_never"], h_never=M["h_never"],
                             never_slots=never_m, spk_ref=sum(1 for sl in ref_w if sl)))
    pc = lambda p, q: 100.0 * p / max(1, q)
    print(f"== {x.pred} — 소리 프레임 ±{x.tol} s · 규칙 {RULES}")
    for (c, sub), v in sorted(agg.items()):
        n = v["rec"]
        print(f"  {c}/{sub}: 참조 {v['ref']} · X 단어 {v['x']}(되찾음 {v['rec']} · 겹침 {v['dup']} · 헛 {v['ins']}) · 되찾음 = 참조의 {pc(n, v['ref']):.2f} %"
              + (f" · 길이 어긋남 {v['x_len_mismatch']}" if v["x_len_mismatch"] else ""))
        print("    맞는 슬롯: " + " · ".join(f"{k} {pc(v['ok_' + k], n):.1f} %" for k in RULES)
              + f" · 정답 슬롯이 그 앞에 한 번도 안 켜짐 {pc(v['h_never'], n):.1f} %(cur 맞음 {pc(v['ok_cur_h_never'], v['h_never']):.0f} · never 맞음 {pc(v['ok_never_h_never'], v['h_never']):.0f})")
        print("    정답 슬롯 마지막 활성에서: " + " · ".join(f"{nm} {pc(v['h_gap_' + nm], n):.1f} %" for _, nm in GAPS) + f" · 끝까지 안 켜진 슬롯 {v['never_slots']}(행 {v['rows']})")
    print("  회의(X 단어 많은 순 12)")
    for m in sorted(meet, key=lambda m: -m["x"])[:12]:
        print(f"    {m['corpus']}/{m['sub']} {m['id']}: 참조 화자 {m['spk_ref']} · 끝까지 안 켜진 슬롯 {m['never_slots']} · X {m['x']} · 되찾음 {m['rec']}(cur 맞음 {m['ok_cur']} · never 맞음 {m['ok_never']} · 정답 슬롯 안 켜짐 {m['h_never']}) · 참조 {m['ref']}")
    if x.json: json.dump(dict(agg={f"{c}/{s}": dict(v) for (c, s), v in agg.items()}, meetings=meet), open(x.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
