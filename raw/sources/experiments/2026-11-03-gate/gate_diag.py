"""E13 활동 게이트 진단(2026-11-03, 사용자 "N3D 가 놓친 말을 받아쓰도록 게이트를 고칠 방법을 찾아줘") — 서버 호스트 CPU, 디코드 없음, 원 결과 폴더는 읽기만.
원출력(predictions-rank*.jsonl 의 emitted · act_runs · perm)과 참조(cst_stage3_eval.window_ref — 채점과 같은 단어·시각)로 참조 단어를 방출 라운드의 게이트 상태로 나눈다.
  방출 라운드 e = ⌊t/0.08⌋ + δ(t = 단어 마지막 토큰 시각, cst_dialogue.block_streams 글자 버킷). 화자 r 의 짝 슬롯 h = perm⁻¹(r)(cpWER 짝 — 없으면 unmatched).
  gate(E13 디코드 — batch_decode.BlockState.open_slots): 활동 창 [e − 11, e] 에 h 활동 ∪ 열린 발화(e 앞 라운드까지 h 의 <ONSET> 수 > <EOT> 수).
  win: 활동 창만(출력과 무관 — E11·E13 을 같은 기준으로 나눈다).
  범주 own(h 열림) · other(h 닫힘, 다른 슬롯 열림) · none(모두 닫힘 — 그 라운드는 [오디오] 하나) · unmatched.
  범주마다 가설이 그 단어를 냈나(같은 글자, 소리 프레임 ±tol): any(아무 슬롯 — 받아쓰기) · own(짝 슬롯 — 귀속까지).
  --windows W…: 활동 창을 W 로 넓히면 짝 슬롯이 열리는 단어 몫(cover)과 라운드당 열린 슬롯 수(open).
  python gate_diag.py <결과 폴더> [--windows 12,25,50,100,200] [--tol 0.6] [--json 출력.json]
"""
import argparse, bisect, collections, glob, json, os, re, sys
import numpy as np

REPO = "/soundai/users/tskim/VAPKT"
CHUNK = 0.08
norm = lambda w: re.sub(r"[^\w']+", "", w.lower())


def windows_open(A, W: int):
    """A (K, N) bool → (K, N) bool: 라운드 k 의 활동 창 [k − W + 1, k] 에 활동이 있나(cst_dialogue.gate_rounds 와 같은 창)."""
    A = np.asarray(A, bool); K = A.shape[0]
    cs = np.concatenate([np.zeros((1, A.shape[1]), np.int64), np.cumsum(A, 0)], 0)
    k = np.arange(K); return (cs[k + 1] - cs[np.maximum(0, k - int(W) + 1)]) > 0


def open_utt(events, K: int, N: int):
    """parse_emitted events [(B|E|W, 슬롯, 라운드)] → (K, N) bool: 라운드 k 앞(< k)까지 그 슬롯이 낸 <ONSET> 수 > <EOT> 수(BlockState.open_)."""
    d = np.zeros((K + 1, N), np.int64)
    for typ, s, k in events:
        if typ in ("B", "E") and 0 <= s < N and 0 <= k < K: d[k + 1, s] += 1 if typ == "B" else -1
    return np.cumsum(d, 0)[:K] > 0


def classify(op, e: int, h):
    """op (K, N) 열림 → 'own' | 'other' | 'none' | 'unmatched'(h None). 창 끝 뒤 라운드(flush)는 마지막 라운드로 본다."""
    if h is None: return "unmatched"
    row = op[min(e, op.shape[0] - 1)]
    return "own" if h < len(row) and row[h] else "other" if row.any() else "none"


def match(ref, hyp, tol_f: int):
    """ref [(단어, 프레임, 짝 슬롯)], hyp 슬롯별 [(단어, 프레임)] → (any, own) bool 목록: 같은 글자·프레임 ±tol_f 인 가설 단어를 하나씩(가까운 것부터) 짝짓는다."""
    def run(slots_of):
        pool = collections.defaultdict(list)
        for s, sl in enumerate(hyp):
            for w, f in sl: pool[(norm(w), s if slots_of else -1)].append(f)
        for v in pool.values(): v.sort()
        used = collections.defaultdict(set); out = []
        for w, f, h in ref:
            key = (norm(w), h if slots_of else -1); v = pool.get(key, []); i = bisect.bisect_left(v, f - tol_f); best = None
            while i < len(v) and v[i] <= f + tol_f:
                if i not in used[key] and (best is None or abs(v[i] - f) < abs(v[best] - f)): best = i
                i += 1
            if best is not None: used[key].add(best)
            out.append(best is not None)
        return out
    return run(False), run(True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("--windows", default="12,25,50,100,200")
    ap.add_argument("--tol", type=float, default=0.6); ap.add_argument("--json")
    x = ap.parse_args(); Ws = [int(w) for w in x.windows.split(",")]
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
    ck = cfg["checkpoint"]; tok = AutoTokenizer.from_pretrained(ck); sp = json.load(open(os.path.join(ck, "config.json")))["sp_ids"]
    spk = tuple(sp[t] for t in SLOTS[:4]); EOT = sp["<EOT>"]; BEGIN = sp.get("<ONSET>")
    a = argparse.Namespace(split=None, corpora=cfg["corpora"], data="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1", deltas=cfg["deltas"],
                           backchannel=tuple(cfg["backchannel"]), max_speakers=cfg.get("max_speakers"), declare=cfg.get("declare"),
                           window=cfg["window"] if cfg["window"] == "full" else tuple(cfg["window"]), hop=cfg.get("hop", 10.0),
                           window_draws=cfg.get("window_draws", 0), context_s=cfg.get("context_s", 0.0), limit_windows=0, xl_order="lang")
    sets, todo = E.windows_of(a, tok)
    tol_f = int(round(x.tol / CHUNK)); agg = collections.defaultdict(lambda: collections.Counter()); n_rows = 0
    for c, i in todo:
        ds = sets[c]; cid, t0w, L = ds.items[i]
        for delta in cfg["deltas"]:
            r = rows.get((f"{cid}@{t0w}", delta))
            if r is None or "act_runs" not in r: continue
            K = int(round(L / CHUNK)); g = (c, SUB[cid.split(":", 1)[1]]); C = agg[g]
            ref_w, _, mask, sp_order = E.window_ref(ds, i, delta, EOT, tok)
            ev = []; hs, _ = parse_emitted(r["emitted"], spk, EOT, drop_chunks=mask, begin_id=BEGIN, events=ev)
            hw = [E.words_of(tok, h) for h in hs]
            A = np.asarray(act_from_runs(r["act_runs"], K), bool); N = A.shape[1]
            win = windows_open(A, 12); gate = win | open_utt(ev, K, N)
            inv = {p: h for h, p in enumerate(r["perm"])}
            ref = [(w, int(t / CHUNK), inv.get(s)) for s, sl in enumerate(ref_w) for w, t in sl]
            hyp = [[(w, int(round(t / CHUNK)) - 1 - int(delta)) for w, t in sl] for sl in hw]
            m_any, m_own = match(ref, hyp, tol_f)
            covers = {W: windows_open(A, W) for W in Ws}
            for (w, b, h), ma, mo in zip(ref, m_any, m_own):
                e = b + int(delta)
                for name, op in (("gate", gate), ("win", win)):
                    k = classify(op, e, h); C[f"{name}:{k}:n"] += 1; C[f"{name}:{k}:any"] += int(ma); C[f"{name}:{k}:own"] += int(mo)
                if h is not None:
                    for W, op in covers.items(): C[f"cover{W}"] += int(op[min(e, K - 1), h] if h < N else 0)
                C["n_ref"] += 1; C["n_matched_spk"] += int(h is not None); C["any"] += int(ma); C["own"] += int(mo)
            for W, op in covers.items(): C[f"open{W}"] += int(op.sum()); C["rounds"] += K if W == Ws[0] else 0
            C["n_hyp"] += sum(map(len, hw)); n_rows += 1
    out = {f"{c}/{sub}": dict(v) for (c, sub), v in sorted(agg.items())}
    pc = lambda p, q: 100.0 * p / max(1, q)
    print(f"== {x.pred} — 행 {n_rows} · 소리 프레임 ±{x.tol} s · 창 {Ws}")
    for g, v in out.items():
        n = v["n_ref"]; m = v["n_matched_spk"]
        print(f"  {g}: 참조 단어 {n}(짝 화자 {m}) · 가설 {v['n_hyp']} · 받아씀(아무 슬롯) {pc(v['any'], n):.1f} % · 짝 슬롯 {pc(v['own'], n):.1f} %")
        for name in ("gate", "win"):
            parts = []
            for k in ("own", "other", "none", "unmatched"):
                nk = v.get(f"{name}:{k}:n", 0)
                if nk: parts.append(f"{k} {pc(nk, n):.1f} %(받아씀 {pc(v.get(f'{name}:{k}:any', 0), nk):.0f} · 짝 {pc(v.get(f'{name}:{k}:own', 0), nk):.0f})")
            print(f"    {name:4s} " + " · ".join(parts))
        print("    창 W: " + " · ".join(f"W{W} 덮음 {pc(v.get(f'cover{W}', 0), m):.1f} %·열린 슬롯 {v.get(f'open{W}', 0) / max(1, v['rounds']):.2f}" for W in Ws))
    if x.json: json.dump(out, open(x.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
