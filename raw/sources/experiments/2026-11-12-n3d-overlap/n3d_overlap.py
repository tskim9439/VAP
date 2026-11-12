"""N3D 가 겹친 구간에서 화자를 얼마나 잡는가(2026-11-12, 사용자 "혹시 N3D 자체가 겹침 구간 탐지를 잘 못하나?").
저장한 N3D 확률(스트리밍, 80 ms 프레임, > 0.5)과 같은 파일의 정답 활동을 AMI eval 회의마다 최적 짝으로 맞춘 뒤, 그 프레임에 말하는 정답 화자 수(1·2·3+)별로 센다. 읽기만."""
import glob, json, numpy as np
from scipy.optimize import linear_sum_assignment
D = "/soundai/users/tskim/VAPKT-data"; P = f"{D}/tmp/e11-check/diar-probs/n3d/n3d/probs"
ev = set(json.load(open(f"{D}/data/ami-bench-v1/manifest.json"))["eval"])
for cond in ("ami-ihmmix", "ami-sdm"):
    A = dict(sf=np.zeros(4), hit=np.zeros(4), fr=np.zeros(4), both=0, one=0, none=0, new_miss=0, old_miss=0, one_n=0, fa1=0, tp=0, pp=0, rp=0, n=0)
    dur = {b: [0, 0] for b in ("<0.5", "0.5-1", "1-2", "2+")}; nm = 0
    for f in sorted(glob.glob(f"{P}/{cond}:*@full.npz")):
        cid = f.split("/")[-1].split("@")[0].split(":")[1]
        if cid not in ev: continue
        z = np.load(f); m = z["mask"].astype(bool); ref = z["ref"].astype(bool); pr = z["prob"].astype(np.float32) > 0.5; nm += 1
        C = np.array([[(ref[:, s] & pr[:, k]).sum() for k in range(pr.shape[1])] for s in range(ref.shape[1])])
        r, c = linear_sum_assignment(-C); hyp = np.zeros_like(ref); hyp[:, r] = pr[:, c]                      # 최적 짝(정답 화자 → N3D 슬롯)
        if m.mean() < 0.5: m = ~m                                                                            # mask 극성(참 = 채점 프레임)
        nr = ref.sum(1); nh = hyp.sum(1); A["n"] += int(m.sum())
        for i, sel in enumerate((nr == 1, nr == 2, nr >= 3)):
            s_ = sel & m; A["fr"][i] += s_.sum(); A["sf"][i] += ref[s_].sum(); A["hit"][i] += (ref[s_] & hyp[s_]).sum()
        s1 = (nr == 1) & m; A["fa1"] += int((nh[s1] >= 2).sum())
        A["tp"] += int(((nr >= 2) & (nh >= 2) & m).sum()); A["pp"] += int(((nh >= 2) & m).sum()); A["rp"] += int(((nr >= 2) & m).sum())
        # 둘 겹침: 둘 다·하나·없음, 하나만 잡았을 때 놓친 쪽이 나중에 말을 시작한 화자인가
        onset = np.zeros(ref.shape, np.int64)                                                               # 지금 발화 구간의 시작 프레임
        for s in range(ref.shape[1]):
            st = 0
            for t in range(ref.shape[0]):
                if ref[t, s] and (t == 0 or not ref[t - 1, s]): st = t
                onset[t, s] = st
        s2 = np.where((nr == 2) & m)[0]
        for t in s2:
            sp = np.where(ref[t])[0]; d = hyp[t, sp]
            if d.all(): A["both"] += 1
            elif d.any():
                A["one"] += 1; miss = sp[~d][0]; newer = sp[np.argmax(onset[t, sp])]; A["one_n"] += 1
                if miss == newer: A["new_miss"] += 1
                else: A["old_miss"] += 1
            else: A["none"] += 1
        # 겹침 구간 길이별: 구간 프레임 중 겹친 화자가 모두 잡힌 몫
        ov = (nr >= 2) & m; t = 0; T = len(ov)
        while t < T:
            if ov[t]:
                e = t
                while e < T and ov[e]: e += 1
                L = (e - t) * 0.08; b = "<0.5" if L < 0.5 else "0.5-1" if L < 1 else "1-2" if L < 2 else "2+"
                dur[b][0] += e - t; dur[b][1] += int((ref[t:e] & hyp[t:e]).sum(1).__eq__(ref[t:e].sum(1)).sum()); t = e
            else: t += 1
    n2 = A["both"] + A["one"] + A["none"]
    print(f"== {cond} · eval 회의 {nm} · 채점 프레임 {A['n']}")
    print("  말하는 화자를 N3D 가 잡은 몫(화자·프레임): 혼자 {:.1f} % · 둘 겹침 {:.1f} % · 셋+ {:.1f} %".format(*[100 * A['hit'][i] / max(1, A['sf'][i]) for i in range(3)]))
    print("  프레임 몫: 혼자 {:.1f} % · 둘 겹침 {:.1f} % · 셋+ {:.1f} %".format(*[100 * A['fr'][i] / max(1, A['fr'][:3].sum()) for i in range(3)]))
    print(f"  둘 겹침 프레임: 둘 다 잡음 {100 * A['both'] / n2:.1f} % · 한 명만 {100 * A['one'] / n2:.1f} % · 아무도 {100 * A['none'] / n2:.1f} %")
    print(f"  한 명만 잡았을 때 놓친 쪽: 나중에 말을 시작한 화자 {100 * A['new_miss'] / max(1, A['one_n']):.1f} % · 먼저 말하던 화자 {100 * A['old_miss'] / max(1, A['one_n']):.1f} %")
    print(f"  겹침 탐지(둘 이상 켜짐): 재현 {100 * A['tp'] / max(1, A['rp']):.1f} % · 정밀 {100 * A['tp'] / max(1, A['pp']):.1f} % · 혼자 말 프레임에서 둘 이상 켠 몫 {100 * A['fa1'] / max(1, A['fr'][0]):.1f} %")
    print("  겹침 구간 길이별 '겹친 화자 모두 잡음' 프레임 몫: " + " · ".join(f"{b} s {100 * v[1] / max(1, v[0]):.1f} %(프레임 몫 {100 * v[0] / max(1, sum(x[0] for x in dur.values())):.0f} %)" for b, v in dur.items()))
