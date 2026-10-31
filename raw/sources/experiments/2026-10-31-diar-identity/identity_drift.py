# 화자 분리 혼동 나누기(2026-10-31, 사용자 "고정 화자 분리기를 대신할 수 있는 방법을 한번 모색해봐"): 회의 통째 프레임 확률(diar_eval --save-probs 의 probs/<id>.npz)에서
# 혼동을 "구간 안 짝(구간마다 가장 잘 맞는 화자 짝)"과 "회의 전체 짝(회의 내내 한 짝)"으로 나눠 잰다. 읽기만 한다.
#   구간 안 혼동이 크면 화자를 가르는 특징 자체가 약하고, 회의 전체 혼동만 크면(차이 = 정체 표류) 시간이 지나며 슬롯의 화자가 바뀐다.
#   python3 identity_drift.py <probs 폴더> [--threshold 0.5] [--blocks 30,120,600] [--sub eval|dev|all]
import argparse, glob, itertools, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("probs"); ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--blocks", default="30,120,600")
ap.add_argument("--sub", default="eval", choices=("eval", "dev", "all"))
a = ap.parse_args()
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
FS = 0.08


def best_map(H, Rf):
    """가설 열·참조 열 겹침을 가장 크게 하는 일대일 짝(작은 쪽 수만큼) → {가설 열: 참조 열}."""
    O = H.T.astype(np.int64) @ Rf.astype(np.int64)                                             # (Nh, Nr)
    nh, nr = O.shape; best, arg = -1, None
    if nh <= nr:
        for p in itertools.permutations(range(nr), nh):
            s = sum(O[i, p[i]] for i in range(nh))
            if s > best: best, arg = s, {i: p[i] for i in range(nh)}
    else:
        for p in itertools.permutations(range(nh), nr):
            s = sum(O[p[j], j] for j in range(nr))
            if s > best: best, arg = s, {p[j]: j for j in range(nr)}
    return arg


def counts(H, Rf, mp):
    """프레임 DER 성분 합(NIST 꼴): 참조 말 수, 놓침, 헛, 혼동 — 짝 mp 로."""
    nr, nh = Rf.sum(1), H.sum(1)
    corr = np.zeros(len(H), np.int64)
    for i, j in mp.items(): corr += (H[:, i] & Rf[:, j]).astype(np.int64)
    return int(nr.sum()), int(np.maximum(nr - nh, 0).sum()), int(np.maximum(nh - nr, 0).sum()), int((np.minimum(nr, nh) - corr).sum())


blocks = [float(x) for x in a.blocks.split(",") if x]
agg = {}
for p in sorted(glob.glob(os.path.join(a.probs, "*.npz"))):
    wid = os.path.basename(p)[:-4]; corpus, rest = wid.split(":", 1); sub = SUB.get(rest.split("@")[0])
    if a.sub != "all" and sub != a.sub: continue
    z = np.load(p); prob, ref = z["prob"].astype(np.float32), z["ref"] > 0
    m = z["mask"].astype(bool) if "mask" in z.files else np.ones(len(prob), bool)
    K = min(len(prob), len(ref), len(m)); H = (prob[:K] > a.threshold)[m[:K]]; Rf = ref[:K][m[:K]]
    g = agg.setdefault(corpus, dict(ref=0, miss=0, fa=0, conf_g=0, **{f"conf_{b:g}": 0 for b in blocks}, early=[0, 0], late=[0, 0], n=0))
    mp = best_map(H, Rf); r_, mi, fa, cg = counts(H, Rf, mp)
    g["ref"] += r_; g["miss"] += mi; g["fa"] += fa; g["conf_g"] += cg; g["n"] += 1
    for b in blocks:
        L = int(round(b / FS)); cl = 0
        for s in range(0, len(H), L):
            hb, rb = H[s:s + L], Rf[s:s + L]
            if rb.any() or hb.any(): cl += counts(hb, rb, best_map(hb, rb))[3]
        g[f"conf_{b:g}"] += cl
    half = len(H) // 2                                                                           # 회의 앞·뒤 반의 전체 짝 혼동(표류가 쌓이나)
    for k, sl in (("early", slice(0, half)), ("late", slice(half, None))):
        r2, _, _, c2 = counts(H[sl], Rf[sl], mp); g[k][0] += c2; g[k][1] += r2
print(f"== {a.probs} — {a.sub}, 임계 {a.threshold}, 혼동 % (참조 말 프레임 대비)")
for c, g in agg.items():
    R = max(1, g["ref"]); loc = " · ".join(f"{b:g} s 구간 짝 {100 * g[f'conf_{b:g}'] / R:.2f}" for b in blocks)
    print(f"  {c}({g['n']} 회의): DER {100 * (g['miss'] + g['fa'] + g['conf_g']) / R:.2f} = 놓침 {100 * g['miss'] / R:.2f} + 헛 {100 * g['fa'] / R:.2f} + 혼동(회의 전체 짝) {100 * g['conf_g'] / R:.2f}"
          f" | {loc} | 전체 짝 혼동 앞 반 {100 * g['early'][0] / max(1, g['early'][1]):.2f} · 뒤 반 {100 * g['late'][0] / max(1, g['late'][1]):.2f}")
