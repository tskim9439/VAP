# E1 miss 분해(mxc 호스트, 2026-10-25): diar_eval --save-probs 의 창 확률(probs/<id>.npz)로, 임계 θ 에서 놓친 음성 중 겹친 프레임 몫과
#   참조 화자별 검출률(창마다 혼동이 가장 적은 슬롯 순열로 짝)을 창 말 몫 구간별로 낸다. ssh mxc python3 - θ < e1_miss.py(numpy 만 — 시스템 python 으로 돈다).
import glob, itertools, json, os, sys
import numpy as np
O = "/soundai/users/tskim/VAPKT-data/results/diar-eval-ami-q17-diar-e1-final"; TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.45
meta = {r["id"]: r for f in glob.glob(O + "/windows-rank*.jsonl") for r in map(json.loads, open(f))}
for cs in (("ami-sdm", "eval"), ("ami-sdm", "dev"), ("ami-ihmmix", "eval")):
    spk = []; tot_ref = tot_miss = 0.0; ov_ref = ov_miss = 0.0
    for wid, r in meta.items():
        if (r["corpus"], r["sub"]) != cs: continue
        z = np.load(f"{O}/probs/{wid}.npz"); m = z["mask"]; p = (z["prob"].astype(np.float32) > TH)[m].astype(float); g = z["ref"][m].astype(float)
        n_ref, n_hyp = g.sum(1), p.sum(1); tot_ref += n_ref.sum(); tot_miss += np.clip(n_ref - n_hyp, 0, None).sum()
        ov = n_ref >= 2; ov_ref += n_ref[ov].sum(); ov_miss += np.clip(n_ref - n_hyp, 0, None)[ov].sum()
        best = max(itertools.permutations(range(4)), key=lambda q: (g * p[:, list(q)]).sum())
        L = g.sum()
        for s in range(4):
            if g[:, s].sum() == 0: continue
            hit = (g[:, s] * p[:, best[s]]).sum(); spk.append((g[:, s].sum() / max(1, L), g[:, s].sum(), hit))
    a = np.array(spk); share, fr, hit = a[:, 0], a[:, 1], a[:, 2]; lost = hit / fr < 0.1
    print(f"== {cs} 임계 {TH} · 참조 화자 {len(a)} · miss {100 * tot_miss / tot_ref:.1f} %(겹친 프레임 몫: 참조 {100 * ov_ref / tot_ref:.1f} %, miss {100 * ov_miss / max(1, tot_miss):.1f} %)")
    print(f"   통째로 놓친 화자(검출 < 10 %): {lost.sum()} 명({100 * lost.mean():.1f} %) · 그 화자 참조 프레임 = 전체 참조의 {100 * fr[lost].sum() / fr.sum():.1f} % · 창 말 몫 중앙값 {np.median(share[lost]) * 100:.1f} %(나머지 {np.median(share[~lost]) * 100:.1f} %)")
    for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 1.01)):
        k = (share >= lo) & (share < hi)
        print(f"   창 말 몫 {lo:.2f}–{min(hi, 1):.2f}: 화자 {k.sum():4d} · 검출률(프레임) {100 * hit[k].sum() / max(1, fr[k].sum()):.1f} % · 통째로 놓침 {100 * lost[k].mean():.1f} %")
