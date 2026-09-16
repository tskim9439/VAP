#!/usr/bin/env python
"""Phase 2 mono 혼합 캐시 사전 생성 — 학습 중 첫 접근 때 대화 채널 전체를 섞는 비용(D1 첫 epoch 지연)을 없앤다. CPU 작업이라 SLURM 없이 mxc 로그인 노드에서 돌린다(사용자 지시 2026-09-16).
  python experiments/p2_build_mono.py --data /soundai/users/tskim/VAPKT-data/data/phase2 [--corpora all] [--procs 16]
캐시 규약은 vapasr.data.dialogue_dataset.build_mono_cache(<mono>/<corpus>/<conv>.npy, float16, 원자적 저장) 그대로라 학습(DialogueWindowDataset.mono) 과 같은 파일을 쓴다. 이미 있는 파일은 건너뛴다."""
import os, sys, time, json, argparse, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--mono", default=None, help="캐시 디렉토리(기본 <data>/mono)")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,ami,notsofar,icsi"); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--nice", type=int, default=10)
a = ap.parse_args(); os.nice(a.nice); MONO = a.mono or os.path.join(a.data, "mono")
from vapasr.data.dialogue import Dialogue
from vapasr.data.dialogue_dataset import build_mono_cache, mono_cache_path

def one(line):
    d = Dialogue.from_json(line); f = mono_cache_path(MONO, d)
    if os.path.exists(f): return ("skip", d.conv_id, os.path.getsize(f), 0.0)
    t = time.time()
    try: build_mono_cache(d, MONO); return ("done", d.conv_id, os.path.getsize(f), time.time() - t)
    except Exception as e: return ("fail", d.conv_id, 0, f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    tot = dict(done=0, skip=0, fail=0, bytes=0); T0 = time.time()
    for c in a.corpora.split(","):
        p = os.path.join(a.data, f"{c}.refined.dialogues.jsonl")
        if not os.path.exists(p): print(f"!! {p} 없음 — 건너뜀", flush=True); continue
        lines = [l for l in open(p, encoding="utf-8") if l.strip()]; lines.sort(key=len, reverse=True)      # 긴 대화(채널 많음) 먼저 → 꼬리 지연 축소
        t0 = time.time(); n = dict(done=0, skip=0, fail=0, bytes=0); fails = []
        with mp.get_context("fork").Pool(a.procs) as pool:
            for i, (st, cid, sz, info) in enumerate(pool.imap_unordered(one, lines, chunksize=1), 1):
                n[st] += 1; n["bytes"] += sz
                if st == "fail": fails.append((cid, info))
                if i % 200 == 0 or i == len(lines): print(f"  {c}: {i}/{len(lines)} done {n['done']} skip {n['skip']} fail {n['fail']} {n['bytes']/2**30:.1f} GiB {time.time()-t0:.0f}s", flush=True)
        for cid, info in fails[:20]: print(f"  !! {c} {cid}: {info}", flush=True)
        print(f"mono {c} done: {json.dumps(n)} {time.time()-t0:.0f}s", flush=True)
        for k in tot: tot[k] += n[k]
    print(f"mono all done: {json.dumps(tot)} {tot['bytes']/2**30:.1f} GiB {time.time()-T0:.0f}s", flush=True); print("EXIT=0")
