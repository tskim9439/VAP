# V3 평가 묶음 run 비교 요약(2026-10-25, noeot_summary.py 를 run 인자로 — 계산은 같다): dev(선택)·TAXI(보고)·가드. mxc results 를 읽기만.
#   python3 v3_runs_summary.py --base <기준 결과 꼬리> 이름=<결과 꼬리> …   (결과 꼬리 = <run>-<태그>, 예 q17-v3-p1-tel-xlw-final · q17-v3-p1-tel-xlw-avg3-final)
#   단일 화자 최대 +Δ 는 --base 대비다(noeot_summary.py 는 V1 st4-m2a-ck2000 고정). v3_summary.py(10-22, pull 폴더용)와 다른 스크립트다.
import json, os, sys
R = "/soundai/users/tskim/VAPKT-data/results"; J = lambda p: json.load(open(p)) if os.path.exists(p) else None
A = sys.argv[1:]; assert len(A) >= 3 and A[0] == "--base", "usage: v3_runs_summary.py --base <꼬리> 이름=<꼬리> …"
BASE = A[1]; RUNS = [tuple(x.split("=", 1)) for x in A[2:]]
f = lambda x, k=2: "–" if x is None else f"{x:.{k}f}"
def dev(n, s):
    d = f"{R}/v3dev-{n}-{s}-sessions"; a, t = J(f"{d}/asr-report-d10.json"), J(f"{d}/turn-metrics-d10.json")
    return (dict(cp=a["split"]["cp"], agn=a["split"]["agn"], attr=a["split"]["attr"], tog=(t or {}).get("toggle_recall", {}).get("pct")) if a else {})
def taxi(n, d):
    a = J(f"{R}/stage4-eval-{n}/taxi90/asr-report-d{d}.json"); t = J(f"{R}/stage4-eval-{n}/taxi90/turn-metrics-d{d}.json")
    return dict(a["split"], tog=(t or {}).get("toggle_recall", {}).get("pct")) if a and "split" in a else {}
def pooled(sm, d):
    rows = [v for k, v in sm.items() if k.endswith(f"/d{d}")]; n = sum(r["ref_units"] for r in rows); w = lambda key: sum(r[key] * r["ref_units"] for r in rows) / n
    return dict(cp=w("err_cp"), agn=w("err_agn"), attr=w("err_cp") - w("err_agn"))
def bleu(sm, d):
    b = [x["bleu"] for k, v in sm.items() if k.endswith(f"/d{d}") for x in (v.get("translation") or {}).values()]; return sum(b) / len(b) if b else None
def regress(n):
    out = {}
    for s in ("fleurs", "vpaa-kspon", "yodas-de", "libri-other", "callhome-de"):
        x = J(f"{R}/stage3-regress-{n}/{s}/summary.json")
        if not x: continue
        for g, v in x["groups"].items():
            m = v["metrics"]; k = "cer_nospace" if "cer_nospace" in m else "wer"; out[f"{s}:{g}"] = 100 * m[k]["rate"]
        for g, v in (x.get("translation") or {}).items(): out[f"{s}:{g}:bleu"] = v["bleu"]
    return out
print("## dev(선택) xl-tel-dev δ10 · 짝 비교 xl-eval-v1")
print("| run | tel cp | agn | 귀속 | toggle | xl-eval cp | 귀속 |"); print("|---|---:|---:|---:|---:|---:|---:|")
for name, n in RUNS:
    a, b = dev(n, "xl-tel-dev"), dev(n, "xl-eval-v1")
    print(f"| {name} | {f(a.get('cp'))} | {f(a.get('agn'))} | {f(a.get('attr'))} | {f(a.get('tog'), 1)} | {f(b.get('cp'))} | {f(b.get('attr'))} |")
print("\n## TAXI(보고) 세션 통째 δ10 · δ4")
print("| run | δ10 cp | agn | 귀속 | toggle | δ4 cp | 귀속 |"); print("|---|---:|---:|---:|---:|---:|---:|")
for name, n in RUNS:
    a, b = taxi(n, 10), taxi(n, 4)
    print(f"| {name} | {f(a.get('cp'), 1)} | {f(a.get('agn'), 1)} | {f(a.get('attr'), 1)} | {f(a.get('tog'), 1)} | {f(b.get('cp'), 1)} | {f(b.get('attr'), 1)} |")
print(f"\n## 가드 — 같은 언어 대화(Stage 3 7 코퍼스), 대화 번역 BLEU, 합성 xl BLEU, 단일 화자 최대 악화({BASE} 대비)")
reg = {n: regress(n) for n in {BASE} | {n for _, n in RUNS}}
print("| run | δ | cp | agn | 귀속 | 대화 BLEU | xl BLEU | 단일 화자 최대 +Δ |"); print("|---|---|---:|---:|---:|---:|---:|---|")
for d in (10, 4):
    for name, n in RUNS:
        st = J(f"{R}/stage3-eval-{n}/summary.json"); dl = J(f"{R}/stage4-eval-{n}/dlg/summary.json"); xl = J(f"{R}/stage4-eval-{n}/xl/summary.json")
        if not st: print(f"| {name} | {d} | 결과 없음 |"); continue
        p = pooled(st, d); worst = "–"
        if n != BASE:
            ds = [(reg[n][k] - reg[BASE][k], k) for k in reg[n] if k in reg[BASE] and not k.endswith(":bleu") and f"delta-{d}" in k]
            if ds: m = max(ds); worst = f"{m[0]:+.2f}({m[1]})"
        print(f"| {name} | {d} | {f(p['cp'])} | {f(p['agn'])} | {f(p['attr'])} | {f(bleu(dl, d) if dl else None)} | {f(bleu(xl, d) if xl else None)} | {worst} |")
