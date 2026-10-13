"""q06-s4-ast vs q06-s3-ast — 같은 평가셋·같은 행(eval_single_turn_asr.py 예측).
(1) 받아쓰기 오류율 짝 bootstrap 2,000 회(LibriSpeech 는 화자 군집, 나머지는 발화). Δ = s4 − s3 (%p).
(2) 번역 BLEU 짝 bootstrap 1,000 회(문장 재표본, sacrebleu 문장 통계 합산 — ast_eval 과 같은 토큰화: 한국어 목표 char, 그 외 13a). Δ = s4 − s3.
    + chrF·LAAL(평가 규약, (k+1)·80 ms)·빈 번역(침묵 / 전사만) 수.
usage: python cmp_s4_s3.py asr|mt"""
import collections, glob, json, re, sys
import numpy as np
R = "/soundai/users/tskim/VAPKT-data/results"
ORDER = {"4": 0, "6": 1, "8": 2, "10": 3, "final": 4}


def rows(d):
    out = {}
    for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l); out[(r["dataset"], str(r["delta"]), r["id"])] = r
    return out


def asr():
    rng = np.random.default_rng(7); B = 2000
    print(f"{'set':26s} {'δ':6s} {'n':>5s} {'s3':>7s} {'s4':>7s} {'Δ%p':>7s} {'95% CI':>17s}")
    for name in ("vpaa-kspon", "yodas-de", "librispeech-clean", "librispeech-other"):
        a, b = rows(f"single-turn-{name}-q06-s3-ast"), rows(f"single-turn-{name}-q06-s4-ast")
        for ds, dl in sorted({k[:2] for k in b if k in a}, key=lambda x: (x[0], ORDER[x[1]])):
            ids = sorted(k[2] for k in b if k[:2] == (ds, dl) and k in a)
            m = lambda r: r["metrics"][r["primary_metric"]]
            ea = np.array([m(a[(ds, dl, i)])["errors"] for i in ids], float); eb = np.array([m(b[(ds, dl, i)])["errors"] for i in ids], float)
            n = np.array([m(b[(ds, dl, i)])["n_ref"] for i in ids], float)
            if ds.startswith("librispeech"):
                spk = [re.match(r"(\d+)-\d+-\d+$", i.rsplit("/", 1)[-1]).group(1) for i in ids]; keys = sorted(set(spk)); ix = {k: j for j, k in enumerate(keys)}
                g = np.array([ix[s] for s in spk])
            else:
                g = np.arange(len(ids))
            G = g.max() + 1; EA, EB, N = np.bincount(g, ea, G), np.bincount(g, eb, G), np.bincount(g, n, G)
            s = rng.integers(0, G, (B, G)); cnt = np.stack([np.bincount(x, minlength=G) for x in s])
            d = (cnt @ EB - cnt @ EA) / (cnt @ N) * 100; lo, hi = np.percentile(d, [2.5, 97.5]); base = (eb.sum() - ea.sum()) / n.sum() * 100
            print(f"{ds:26s} δ{dl:5s} {len(ids):5d} {ea.sum() / n.sum() * 100:7.2f} {eb.sum() / n.sum() * 100:7.2f} {base:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]{' *' if lo > 0 or hi < 0 else ''}")


def mt():
    from sacrebleu.metrics import BLEU, CHRF
    rng = np.random.default_rng(7); B = 1000
    print(f"{'set':22s} {'δ':6s} {'n':>5s} {'BLEU s3':>8s} {'s4':>6s} {'Δ':>6s} {'95% CI':>15s}  {'chrF s3→s4':>13s}  {'LAAL s3→s4':>11s}  빈 번역 s3→s4 (침묵/전사만)")
    for name in ("fleurs", "fleurs-en-peak05", "covost2-sub7"):
        a, b = rows(f"ast-{name}-q06-s3-ast"), rows(f"ast-{name}-q06-s4-ast")
        for ds, dl in sorted({k[:2] for k in b if k in a and b[k].get("translation_reference") is not None}, key=lambda x: (x[0], ORDER[x[1]])):
            if dl not in ("4", "10", "final"): continue
            ids = sorted(k[2] for k in b if k[:2] == (ds, dl) and k in a)
            ra, rb = [a[(ds, dl, i)] for i in ids], [b[(ds, dl, i)] for i in ids]
            tok = "char" if rb[0]["tgt_lang"] == "Korean" else "13a"; bleu = BLEU(tokenize=tok)
            refs = [r["translation_reference"] for r in rb]
            sa = np.array(bleu._extract_corpus_statistics([r["translation_hypothesis"] for r in ra], [refs]), float)
            sb = np.array(bleu._extract_corpus_statistics([r["translation_hypothesis"] for r in rb], [refs]), float)
            score = lambda st: bleu._compute_score_from_stats(list(st)).score
            A, Bs = score(sa.sum(0)), score(sb.sum(0)); idx = rng.integers(0, len(ids), (B, len(ids)))
            diffs = np.array([score(sb[j].sum(0)) - score(sa[j].sum(0)) for j in idx]); lo, hi = np.percentile(diffs, [2.5, 97.5])
            chrf = lambda rr: CHRF().corpus_score([r["translation_hypothesis"] for r in rr], [refs]).score
            laal = lambda rr: np.mean([r["translation_laal"] for r in rr if r.get("translation_laal") is not None]) if dl != "final" else float("nan")
            empt = lambda rr: (sum(not r["token_ids"] for r in rr), sum(bool(r["token_ids"]) and not r["translation_hypothesis"] for r in rr))
            ea, eb = empt(ra), empt(rb)
            print(f"{ds:22s} δ{dl:5s} {len(ids):5d} {A:8.2f} {Bs:6.2f} {Bs - A:+6.2f} [{lo:+5.2f}, {hi:+5.2f}]{' *' if lo > 0 or hi < 0 else '  '}  {chrf(ra):5.2f}→{chrf(rb):5.2f}  "
                  f"{laal(ra):5.2f}→{laal(rb):5.2f}  {sum(ea)}→{sum(eb)} ({ea[0]}/{ea[1]} → {eb[0]}/{eb[1]})", flush=True)


{"asr": asr, "mt": mt}[sys.argv[1]]()
