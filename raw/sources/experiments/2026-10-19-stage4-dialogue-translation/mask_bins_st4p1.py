import json, glob, collections
R = "/soundai/users/tskim/VAPKT-data/results"
rows = [json.loads(l) for p in sorted(glob.glob(f"{R}/stage3-eval-st4-p1-final/predictions-rank*.jsonl")) for l in open(p, encoding="utf-8")]
bins = ((0, .2, "마스크 <20 %"), (.2, .4, "20–40 %"), (.4, 1.01, "≥40 %"))
print("st4-p1 final, Stage 3 회귀 창(δ10): 코퍼스 | 마스크 청크 비율 구간별 창 수 · cp · agn · hyp/ref")
for c in ("aihub134-1", "aihub134-2", "aihub71631", "otoSpeech", "kec", "nsc", "ztc"):
    rr = [r for r in rows if r["delta"] == 10 and r["corpus"] == c]; out = []
    for lo, hi, name in bins:
        xs = [r for r in rr if lo <= r["masked_chunks"] / r["K"] < hi]; n = sum(r["n_ref"] for r in xs)
        out.append(f"{name} {len(xs):3d}창 cp {100*sum(r['cp'] for r in xs)/max(1,n):5.1f} agn {100*sum(r['agn'] for r in xs)/max(1,n):5.1f} h/r {sum(r['n_hyp'] for r in xs)/max(1,n):.2f}")
    print(f"  {c:11s} | " + " | ".join(out))
