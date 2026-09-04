#!/usr/bin/env python
"""특징 캐시 stats.json 요약표 (인코더 × 매니페스트: n, 시간, RTF, peak GPU, 용량)."""
import os, glob, json
root = os.environ.get("DATA_FEATURE_CACHE_DIR", "/data3/tskim/features"); tot = {}
print(f"{'encoder':16s} {'manifest':14s} {'n':>4s} {'hours':>6s} {'RTF':>7s} {'peakGB':>6s} {'GB':>6s}")
for f in sorted(glob.glob(os.path.join(root, "*", "*", "stats.json"))):
    s = json.load(open(f)); gb = sum(os.path.getsize(p) for p in glob.glob(os.path.join(os.path.dirname(f), "*.npy"))) / 1e9
    n = len(open(os.path.join(os.path.dirname(f), "index.jsonl")).read().splitlines())
    print(f"{s['encoder']:16s} {s['manifest']:14s} {n:4d} {s['hours']:6.1f} {(s['rtf_encode'] or 0):7.4f} {s['peak_gpu_gb']:6.1f} {gb:6.1f}")
    t = tot.setdefault(s["encoder"], [0, 0.0, 0.0]); t[0] += n; t[1] += s["hours"]; t[2] += gb
print("\n합계:"); [print(f"  {k:16s} n={v[0]:4d} {v[1]:6.1f} h {v[2]:6.1f} GB") for k, v in tot.items()]; print(f"  전체 {sum(v[2] for v in tot.values()):.0f} GB")
