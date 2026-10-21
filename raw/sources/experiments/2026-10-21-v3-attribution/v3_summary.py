# V3 R1·R2 체크포인트별 요약 — dev(선택)·TAXI(보고)·가드(V1 대비). python3 v3_summary.py <pull 폴더> <V1 폴더>
import json, os, sys
D, V1 = sys.argv[1], sys.argv[2]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
RUNS = [("V1", None)] + [(f"{r}-{t}", (r, t)) for r in ("v3-p1", "v3-p1-nodata", "v3-p1-xlw", "v3-p1-xlw-arrival") for t in ("ck1500", "ck2000", "final")]
def dev(name, s):
    d = f"{D}/v3dev-{'st4-m2a-ck2000' if name == 'V1' else name}-{s}-sessions"; a, t = J(f"{d}/asr-report-d10.json"), J(f"{d}/turn-metrics-d10.json")
    if not a: return None
    sm = J(f"{d}/slot-map.json"); sm = (sm or {}).get("d10") or {}
    return dict(cp=a["split"]["cp"], agn=a["split"]["agn"], attr=a["split"]["attr"], rec=(t or {}).get("toggle_recall", {}).get("pct"), swap=f"{sm.get('swap', 0)}/{sm.get('rule', 0) + sm.get('swap', 0)}" if sm else "")
def taxi(name, w, d):
    p = f"{V1}/{w}-asr-report-d{d}.json" if name == "V1" else f"{D}/stage4-eval-{name}/{w}/asr-report-d{d}.json"; a = J(p)
    tm = None if name == "V1" else J(f"{D}/stage4-eval-{name}/{w}/turn-metrics-d{d}.json")
    sm = None if name == "V1" else J(f"{D}/stage4-eval-{name}/{w}/slot-map.json")
    return (a or {}).get("split"), (tm or {}).get("toggle_recall", {}).get("pct"), ((sm or {}).get(f"d{d}") or {})
def pooled(sm, d):
    rows = [v for k, v in sm.items() if k.endswith(f"/d{d}")]; n = sum(r["ref_units"] for r in rows)
    w = lambda key: sum(r[key] * r["ref_units"] for r in rows) / n
    rec = sum(r["eot_recall"] * r["eot_pos"] for r in rows) / sum(r["eot_pos"] for r in rows); pre = sum(r["eot_precision"] * r["eot_hyp"] for r in rows) / sum(r["eot_hyp"] for r in rows)
    return dict(cp=w("err_cp"), agn=w("err_agn"), attr=w("err_cp") - w("err_agn"), eot_r=rec, eot_p=pre)
def bleu(sm, d):
    b = [x["bleu"] for k, v in sm.items() if k.endswith(f"/d{d}") for x in (v.get("translation") or {}).values()]
    return sum(b) / len(b) if b else None
def regress(name):
    out = {}
    for s in ("fleurs", "vpaa-kspon", "yodas-de", "libri-other", "callhome-de"):
        x = J(f"{V1}/regress-{s}.json") if name == "V1" else J(f"{D}/stage3-regress-{name}/{s}/summary.json")
        if not x: continue
        for g, v in x["groups"].items():
            m = v["metrics"]; k = "cer_nospace" if "cer_nospace" in m else "wer"; out[f"{s}:{g}"] = 100 * m[k]["rate"]
        for g, v in (x.get("translation") or {}).items(): out[f"{s}:{g}:bleu"] = v["bleu"]
    return out
st3 = {n: J(f"{V1}/stage3-summary.json") if n == "V1" else J(f"{D}/stage3-eval-{n}/summary.json") for n, _ in RUNS}
dlg = {n: J(f"{V1}/dlg-summary.json") if n == "V1" else J(f"{D}/stage4-eval-{n}/dlg/summary.json") for n, _ in RUNS}
xl = {n: J(f"{V1}/xl-summary.json") if n == "V1" else J(f"{D}/stage4-eval-{n}/xl/summary.json") for n, _ in RUNS}
reg = {n: regress(n) for n, _ in RUNS}
f = lambda x, k=2: "–" if x is None else f"{x:.{k}f}"
print("## dev(선택 기준) — xl-tel-dev 160 · 짝 비교 xl-eval-v1 160, δ10, bias 없음")
print("| run | tel cp | agn | 귀속 | toggle | swap | xl-eval cp | agn | 귀속 |"); print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for n, _ in RUNS:
    a, b = dev(n, "xl-tel-dev"), dev(n, "xl-eval-v1"); a = a or {}; b = b or {}
    print(f"| {n} | {f(a.get('cp'))} | {f(a.get('agn'))} | {f(a.get('attr'))} | {f(a.get('rec'), 1)} | {a.get('swap', '')} | {f(b.get('cp'))} | {f(b.get('agn'))} | {f(b.get('attr'))} |")
print("\n## TAXI(보고 전용) — 세션 통째(90 s) δ10·δ4, 창 40 s δ10")
print("| run | 통째 δ10 cp | agn | 귀속 | toggle | swap | 통째 δ4 cp | 귀속 | 40 s δ10 cp | 귀속 | 40 s swap |"); print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for n, _ in RUNS:
    (s10, r10, m10), (s4, _, _), (w10, _, mw) = taxi(n, "taxi90", 10), taxi(n, "taxi90", 4), taxi(n, "taxi", 10); s10 = s10 or {}; s4 = s4 or {}; w10 = w10 or {}
    sw = lambda m: f"{m.get('swap', 0)}/{m.get('rule', 0) + m.get('swap', 0)}" if m else ""
    print(f"| {n} | {f(s10.get('cp'), 1)} | {f(s10.get('agn'), 1)} | {f(s10.get('attr'), 1)} | {f(r10, 1)} | {sw(m10)} | {f(s4.get('cp'), 1)} | {f(s4.get('attr'), 1)} | {f(w10.get('cp'), 1)} | {f(w10.get('attr'), 1)} | {sw(mw)} |")
print("\n## 가드(V1 ck2000 대비) — Stage 3 같은 언어 대화(7 코퍼스 풀링), 대화 번역 BLEU 평균, 합성 xl, 단일 화자 최대 악화")
print("| run | δ | cp | agn | 귀속 | EOT R | EOT P | 대화 BLEU | xl BLEU | 단일 화자 최대 +Δ(항목) |"); print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for d in (10, 4):
    for n, _ in RUNS:
        if not st3[n]: print(f"| {n} | {d} | 결과 없음 |"); continue
        p = pooled(st3[n], d); db = bleu(dlg[n], d) if dlg[n] else None; xb = bleu(xl[n], d) if xl[n] else None
        worst = "–"
        if n != "V1":
            ds = [(reg[n][k] - reg["V1"][k], k) for k in reg[n] if k in reg["V1"] and not k.endswith(":bleu") and f"delta-{d}" in k]
            if ds: m = max(ds); worst = f"{m[0]:+.2f}({m[1]})"
        print(f"| {n} | {d} | {f(p['cp'])} | {f(p['agn'])} | {f(p['attr'])} | {f(p['eot_r'], 3)} | {f(p['eot_p'], 3)} | {f(db)} | {f(xb)} | {worst} |")
