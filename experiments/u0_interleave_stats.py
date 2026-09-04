#!/usr/bin/env python
"""정렬 결과(jsonl) → interleaved target 생성 → chunk 당 방출 분포·overflow 통계 (δ, M 별). U0 의 M 예산 확정용.
python experiments/u0_interleave_stats.py <align dir> [--delay 2] [--M 1,2,3,4]
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.interleave import build_interleaved, Specials
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--delay", type=int, default=2); ap.add_argument("--M", default="1,2,3,4"); a = ap.parse_args()
sp = Specials(next_audio=-1, empty_audio=-2, spk=(-3, -4))
files = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
res = {}
for M in [int(x) for x in a.M.split(",")]:
    tot = dict(chunks=0, tokens=0, overflow=0, max_backlog=0, hist={})
    for f in files:
        streams = [[], []]; dur = 0.0
        for l in open(f):
            u = json.loads(l); dur = max(dur, u["end"])
            for t in u["tokens"]: streams[u["speaker"]].append((t["id"], t["end_time"]))
        for s in streams: s.sort(key=lambda x: x[1])
        _, st = build_interleaved(streams, dur + 1.0, sp, delay_frames=a.delay, max_per_chunk=M)
        tot["chunks"] += st.chunks; tot["tokens"] += st.tokens; tot["overflow"] += st.overflow_tokens; tot["max_backlog"] = max(tot["max_backlog"], st.max_backlog)
        for k, v in st.per_chunk_hist.items(): tot["hist"][k] = tot["hist"].get(k, 0) + v
    h = tot["hist"]; n = sum(h.values()); res[M] = dict(overflow_frac=tot["overflow"] / max(1, tot["tokens"]), max_backlog=tot["max_backlog"],
                                                       chunk_hist={str(k): round(v / n, 4) for k, v in sorted(h.items())}, tokens=tot["tokens"], chunks=tot["chunks"])
    print(f"M={M}: 토큰 {tot['tokens']} chunk {tot['chunks']} | 이월(추가 지연) 토큰 비율 {res[M]['overflow_frac']*100:.2f} % | max backlog {tot['max_backlog']} | chunk 당 방출 분포 {res[M]['chunk_hist']}")
json.dump(res, open(os.path.join(a.dir, f"interleave-stats-d{a.delay}.json"), "w"), indent=1)
