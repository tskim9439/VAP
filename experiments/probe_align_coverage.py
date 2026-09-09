#!/usr/bin/env python
"""정렬 레코드로 코퍼스별 '라벨이 덮지 못한 오디오' 를 잰다: 발화 안 토큰 간 최대 공백, 첫 토큰 앞·마지막 토큰 뒤 여백, 토큰 구간 합/발화 길이.
토큰에는 end_time 만 있으므로 공백 = 연속 토큰 end_time 차이(긴 단어도 포함되지만 코퍼스 간 비교에는 충분). 자막형 라벨(간투사·반복 생략)이면 내부 공백이 커진다. 2026-09-09: 031/033 라벨이 '이제·인제·그' 를 생략하는 것을 확인 → 코퍼스별 정량화."""
import os, sys, json, glob, random, argparse, statistics
ap = argparse.ArgumentParser(); ap.add_argument("--manifests", required=True); ap.add_argument("--shards", type=int, default=3); ap.add_argument("--max", type=int, default=3000); a = ap.parse_args()
ROOT = os.environ.get("VAPASR_ALIGN_ROOT", "/soundai/users/tskim/VAPKT-data/data/manifests/align-asr-tn-v1")
for m in a.manifests.split(","):
    parts = sorted(glob.glob(os.path.join(ROOT, m, "parts", "*.jsonl"))); random.Random(0).shuffle(parts); gaps, lead, trail, cov, n = [], [], [], [], 0
    for p in parts[: a.shards]:
        for line in open(p):
            try: r = json.loads(line)
            except Exception: continue
            for u in r.get("utts", []) or []:
                toks = u.get("tokens") or []; et = [t.get("end_time") for t in toks]
                if len(et) < 2 or not all(isinstance(x, (int, float)) for x in et): continue
                us, ue = u.get("start", et[0]), u.get("end", et[-1]); dur = ue - us
                if dur <= 0.5: continue
                g = max(et[i + 1] - et[i] for i in range(len(et) - 1)); gaps.append(g); lead.append(et[0] - us); trail.append(ue - et[-1]); cov.append(0.0); n += 1
                if n >= a.max: break
            if n >= a.max: break
        if n >= a.max: break
    if not n: print(f"{m}: 레코드 형식을 못 읽음 — 첫 줄 키: {list(json.loads(open(parts[0]).readline()).keys())[:8]}"); continue
    q = lambda v, p: sorted(v)[int(len(v) * p)]
    print(f"{m:20s} n={n:5d} 내부 최대공백 p50={q(gaps,.5):.2f}s p90={q(gaps,.9):.2f}s ≥1s={sum(g>=1 for g in gaps)/n:.1%} ≥2s={sum(g>=2 for g in gaps)/n:.1%} | 선행 p50={q(lead,.5):.2f}s 후행 p50={q(trail,.5):.2f}s", flush=True)
