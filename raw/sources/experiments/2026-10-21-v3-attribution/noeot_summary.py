# <EOT> 없는 R1x(v3-p1-tel-xlw-noeot) 대 R1x·V1 — dev(선택)·TAXI(보고)·가드(V1 대비). mxc results 를 읽기만.
import json, os, sys
R = "/soundai/users/tskim/VAPKT-data/results"; J = lambda p: json.load(open(p)) if os.path.exists(p) else None
RUNS = [("V1", "st4-m2a-ck2000"), ("R1x final", "v3-p1-tel-xlw-final"), ("R1x avg3", "v3-p1-tel-xlw-avg3-final")] + \
       [(f"noEOT {t}", f"v3-p1-tel-xlw-noeot-{t}") for t in ("ck1500", "ck2000", "final")] + [("noEOT avg3", "v3-p1-tel-xlw-noeot-avg3-final")]
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
print("\n## 가드(V1 대비) — 같은 언어 대화(Stage 3 7 코퍼스), 대화 번역 BLEU, 합성 xl BLEU, 단일 화자 최대 악화")
reg = {n: regress(n) for _, n in RUNS}
print("| run | δ | cp | agn | 귀속 | 대화 BLEU | xl BLEU | 단일 화자 최대 +Δ |"); print("|---|---|---:|---:|---:|---:|---:|---|")
for d in (10, 4):
    for name, n in RUNS:
        st = J(f"{R}/stage3-eval-{n}/summary.json"); dl = J(f"{R}/stage4-eval-{n}/dlg/summary.json"); xl = J(f"{R}/stage4-eval-{n}/xl/summary.json")
        if not st: print(f"| {name} | {d} | 결과 없음 |"); continue
        p = pooled(st, d); worst = "–"
        if n != "st4-m2a-ck2000":
            ds = [(reg[n][k] - reg["st4-m2a-ck2000"][k], k) for k in reg[n] if k in reg["st4-m2a-ck2000"] and not k.endswith(":bleu") and f"delta-{d}" in k]
            if ds: m = max(ds); worst = f"{m[0]:+.2f}({m[1]})"
        print(f"| {name} | {d} | {f(p['cp'])} | {f(p['agn'])} | {f(p['attr'])} | {f(bleu(dl, d) if dl else None)} | {f(bleu(xl, d) if xl else None)} | {worst} |")
st_a, st_b = J(f"{R}/stage3-eval-st3-p11-revive-ck3000/summary.json"), J(f"{R}/stage3-eval-st3-p11-revive-noeot-ck3000/summary.json")
print("\n## 대화 Stage 3 ck3000 같은 언어(δ10): p11 " + (f"{pooled(st_a, 10)}" if st_a else "없음") + " · p11-noEOT " + (f"{pooled(st_b, 10)}" if st_b else "없음"))
