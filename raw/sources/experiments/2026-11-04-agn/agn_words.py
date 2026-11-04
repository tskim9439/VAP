"""agn_diag.py 의 짝으로 자주 놓친 단어·대치 쌍·삽입 단어를 센다(2026-11-04). 서버 호스트 CPU, 디코드 없음, 원 결과 폴더는 읽기만.
  python agn_words.py <결과 폴더> [--tol 0.6] [--top 25] [--sub eval]
"""
import argparse, bisect, collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agn_diag import REPO, CHUNK, keep, match, build_ref   # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("--tol", type=float, default=0.6); ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--sub", default="eval"); x = ap.parse_args()
    sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
    import cst_stage3_eval as E
    from transformers import AutoTokenizer
    from vapasr.data.cst_eval import parse_emitted
    from vapasr.data.cst_dialogue import SLOTS
    man = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json")); SUB = {**{m: "dev" for m in man["dev"]}, **{m: "eval" for m in man["eval"]}}
    cfg = json.load(open(os.path.join(x.pred, "config-rank0.json"))); tok = AutoTokenizer.from_pretrained(cfg["checkpoint"])
    cc = json.load(open(os.path.join(cfg["checkpoint"], "config.json"))); sp = cc["sp_ids"]
    spk = tuple(sp[t] for t in SLOTS[:4]); EOT = sp["<EOT>"]; BEGIN = sp.get("<ONSET>"); XI = int(sp["<SPK_5>"]) if cc.get("x_slot") else -1
    spk_p = spk + ((XI,) if XI >= 0 else ())
    a = argparse.Namespace(split=None, corpora=cfg["corpora"], data="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1", deltas=cfg["deltas"],
                           backchannel=tuple(cfg["backchannel"]), max_speakers=cfg.get("max_speakers"), declare=cfg.get("declare"),
                           window=cfg["window"] if cfg["window"] == "full" else tuple(cfg["window"]), hop=cfg.get("hop", 10.0),
                           window_draws=cfg.get("window_draws", 0), context_s=cfg.get("context_s", 0.0), limit_windows=0, xl_order="lang")
    sets, todo = E.windows_of(a, tok); rows = {}
    for f in glob.glob(os.path.join(x.pred, "predictions-rank*.jsonl")):
        for l in open(f, encoding="utf-8"): r = json.loads(l); rows[(r["id"], r["delta"])] = r
    tol_f = int(round(x.tol / CHUNK)); D = collections.defaultdict(collections.Counter); S = collections.defaultdict(collections.Counter)
    I = collections.defaultdict(collections.Counter); N = collections.Counter()
    for c, i in todo:
        ds = sets[c]; cid, t0w, L = ds.items[i]
        if SUB[cid.split(":", 1)[1].split("@")[0]] != x.sub: continue
        for delta in cfg["deltas"]:
            r = rows.get((f"{cid}@{t0w}", delta))
            if r is None: continue
            R, _, _, mask, _ = build_ref(E, ds, i, delta, EOT, tok)
            hs, _ = parse_emitted(r["emitted"], spk_p, EOT, drop_chunks=mask, begin_id=BEGIN)
            H = [(n, int(round(t / CHUNK)) - 1 - int(delta)) for h in hs for w, t in E.words_of(tok, h) for n in [keep(w)] if n]
            mref, mhyp = match([(q["w"], q["f"]) for q in R], H, tol_f)
            free = sorted((H[j][1], H[j][0]) for j in range(len(H)) if not mhyp[j]); taken = [False] * len(free); fr = [f_ for f_, _ in free]
            for q, m in zip(R, mref):
                if m is not None: continue
                k = bisect.bisect_left(fr, q["f"] - tol_f)
                while k < len(free) and fr[k] <= q["f"] + tol_f and taken[k]: k += 1
                if k < len(free) and fr[k] <= q["f"] + tol_f: taken[k] = True; S[c][(q["w"], free[k][1])] += 1
                else: D[c][q["w"]] += 1
            for (f_, w), tk in zip(free, taken):
                if not tk: I[c][w] += 1
            N[c] += len(R)
    for c in sorted(N):
        n = N[c]; dn, sn, inn = sum(D[c].values()), sum(S[c].values()), sum(I[c].values())
        print(f"== {c}/{x.sub}: 참조 {n} · 삭제꼴 {dn}({100 * dn / n:.2f} %) · 대치꼴 {sn}({100 * sn / n:.2f} %) · 삽입꼴 {inn}({100 * inn / n:.2f} %)")
        top = D[c].most_common(x.top); print(f"  삭제꼴 상위 {x.top}(삭제의 {100 * sum(v for _, v in top) / max(1, dn):.0f} %): " + " · ".join(f"{w} {v}" for w, v in top))
        top = S[c].most_common(x.top); print(f"  대치꼴 상위 {x.top}(대치의 {100 * sum(v for _, v in top) / max(1, sn):.0f} %): " + " · ".join(f"{a_}→{b_} {v}" for (a_, b_), v in top))
        top = I[c].most_common(x.top); print(f"  삽입꼴 상위 {x.top}(삽입의 {100 * sum(v for _, v in top) / max(1, inn):.0f} %): " + " · ".join(f"{w} {v}" for w, v in top))


if __name__ == "__main__":
    main()
