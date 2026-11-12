"""N3D 가 참조 단어 시각에 그 화자 슬롯을 켰는가 — 겹침 수(1·2·3+)별. 단어 시각 기준이라 발화 구간 안 쉼의 영향이 없다. 읽기만."""
import json, sys, collections, numpy as np
from scipy.optimize import linear_sum_assignment
dump, probs = sys.argv[1], sys.argv[2]; FR = 0.08
W = collections.defaultdict(list)
for l in open(dump, encoding="utf-8"):
    q = json.loads(l)
    if q["sys"] == "E13": W[(q["corpus"], q["cid"])].append(q)
res = collections.defaultdict(lambda: collections.Counter())
for (c, cid), ws in sorted(W.items()):
    name = cid if ":" in cid else f"{c}:{cid}"
    try: z = np.load(f"{probs}/{name}@full.npz")
    except FileNotFoundError: z = np.load(f"{probs}/{c}:{cid.split(':')[-1]}@full.npz")
    pr = z["prob"].astype(np.float32) > 0.5; T = len(pr); ns = max(q["s"] for q in ws) + 1
    def on(t, k, lo, hi):                                                    # [t − lo, t + hi] 안에 슬롯 k 가 켜진 프레임이 있는가
        a = max(0, int((t - lo) / FR)); b = min(T, int((t + hi) / FR) + 1); return bool(pr[a:b, k].any())
    C = np.zeros((ns, pr.shape[1]))
    for q in ws:
        f = min(T - 1, int(q["t"] / FR)); C[q["s"]] += pr[max(0, f - 2): f + 1].any(0)
    r, cc = linear_sum_assignment(-C); mp = dict(zip(r, cc))
    for q in ws:
        k = mp.get(q["s"]); ov = "1" if q["ov"] <= 1 else "2" if q["ov"] == 2 else "3+"; R = res[(c, ov)]
        R["n"] += 1; R["miss"] += q["orc"] != "ok"; R["del"] += q["orc"] == "del"
        strict = k is not None and on(q["t"], k, 0.0, 0.0); loose = k is not None and on(q["t"], k, 0.32, 0.16); wide = k is not None and on(q["t"], k, 1.0, 0.5)
        R["strict"] += strict; R["loose"] += loose; R["wide"] += wide
        if not loose: R["off_n"] += 1; R["off_miss"] += q["orc"] != "ok"
        else: R["on_n"] += 1; R["on_miss"] += q["orc"] != "ok"
for c in ("ami-ihmmix", "ami-sdm"):
    print(f"== {c}")
    for ov in ("1", "2", "3+"):
        R = res[(c, ov)]; n = R["n"]
        print(f"  겹침 {ov:2s} 단어 {n:6d} · N3D 가 그 화자를 켠 몫: 그 프레임 {100 * R['strict'] / n:.1f} % · −0.32 ~ +0.16 s {100 * R['loose'] / n:.1f} % · −1.0 ~ +0.5 s {100 * R['wide'] / n:.1f} %"
              f" | 우리 놓침: 켜진 단어 {100 * R['on_miss'] / max(1, R['on_n']):.1f} % · 꺼진 단어 {100 * R['off_miss'] / max(1, R['off_n']):.1f} %(꺼진 단어가 놓침에서 차지하는 몫 {100 * R['off_miss'] / max(1, R['miss']):.1f} %)")
