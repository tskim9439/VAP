# 코퍼스별 발화 길이 구간의 마스크(전사 없는 음성) 비율과, 화자 교대 중 한쪽이 마스크인 비율 — DialogueWindowDataset._bad_spans 규칙(allow_unaligned=False).
import json, sys
B = [(0, .5), (.5, 1), (1, 2), (2, 5), (5, 1e9)]
print(f"{'코퍼스':<12}" + "".join(f"{f'{lo:g}–{hi:g}s' if hi < 1e8 else f'≥{lo:g}s':>13}" for lo, hi in B) + f"{'전체(시간)':>12}{'교대 손실':>10}{'교대 수':>9}")
for path in sys.argv[1:]:
    name = path.rsplit("/", 1)[-1].split(".")[0]; n = [[0, 0] for _ in B]; dur = [0.0, 0.0]; chg = [0, 0]
    for line in open(path):
        d = json.loads(line); us = d["utterances"]; miss = set((d.get("meta") or {}).get("missing", []))
        bad = [bool(u.get("flags")) or not u.get("tokens") or i in miss for i, u in enumerate(us, 1)]
        for u, b in zip(us, bad):
            L = u["end"] - u["start"]; k = next(j for j, (lo, hi) in enumerate(B) if lo <= L < hi)
            n[k][0] += 1; n[k][1] += b; dur[0] += L; dur[1] += L * b
        order = sorted(range(len(us)), key=lambda i: us[i]["start"])
        for a, c in zip(order, order[1:]):
            if us[a]["speaker"] != us[c]["speaker"]: chg[0] += 1; chg[1] += bad[a] or bad[c]
    print(f"{name:<12}" + "".join(f"{f'{100*b/max(1,t):.0f}% ({t//1000}k)':>13}" for t, b in n) + f"{100*dur[1]/max(1e-9,dur[0]):>11.0f}%{100*chg[1]/max(1,chg[0]):>9.0f}%{chg[0]:>9}")
