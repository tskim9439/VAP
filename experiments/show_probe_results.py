#!/usr/bin/env python
"""probe 실행 결과 요약. 고정 FP 예산(≤0.045 / 0.06 / 0.08 / 0.10)에서 sweep 상 **최대 recall** 과 그 지점의 p50 latency — 임계값 이웃(nearest) 이 아닌 예산 기준.
python experiments/show_probe_results.py [--task eot|int] [run 디렉토리 ...]
"""
import os, sys, json, glob, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--task", default="eot"); ap.add_argument("runs", nargs="*"); a = ap.parse_args()
root = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "probe"); runs = a.runs or sorted(glob.glob(os.path.join(root, "*")))
BUDGETS = [0.045, 0.06, 0.08, 0.10]
def best_at(sweep, budget):
    rows = [r for r in sweep if r["fp_rate"] <= budget and r["recall"] is not None]
    return max(rows, key=lambda r: (r["recall"], -r["lat_p50"])) if rows else None
hdr = f"{'run':32s} {'valCE':>6s} {'acc':>5s} | " + " | ".join(f"{a.task.upper()}@fp≤{b:.3f}: R / p50".rjust(22) for b in BUDGETS)
print(hdr); print("-" * len(hdr)); seen = set()
for d in runs:
    p = os.path.join(d, "results.json")
    if not os.path.exists(p): continue
    r = json.load(open(p)); v = (r.get("val") or [{}])[-1]; tb = r.get("turnbench_dev")
    key = (r["encoder"], round(r["frame_hz"], 2), r["run"].split("-d")[1] if "-d" in r["run"] else "")
    cells = []
    if tb:
        sw = tb[a.task]["sweep"]
        for b in BUDGETS:
            o = best_at(sw, b); cells.append(f"{o['recall']:.3f} / {o['lat_p50']:5.0f}ms" if o else "   —   ")
    else: cells = ["-"] * len(BUDGETS)
    tag = " (dup)" if key in seen else ""; seen.add(key)
    print(f"{(r['run'][:26] + tag):32s} {v.get('ce', float('nan')):6.3f} {v.get('acc', float('nan')):5.3f} | " + " | ".join(c.rjust(22) for c in cells))
print(f"\n기준 VAP(oto ckpt, dev): EOT 0.841 @ 0.045 / 463 ms · INT 0.957 @ 0.100 / 896 ms · 사전학습 원본 EOT 0.793 @ 0.094 / 613")
