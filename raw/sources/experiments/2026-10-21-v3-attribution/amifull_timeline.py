# 회의 통째 평가(cst_stage3_eval --window full)의 시간 구간별 점수(2026-10-24): 원출력(predictions-rank*.jsonl)을 평가와 같은 규칙으로 다시 풀어
# 회의마다 B 분 구간으로 나누고, 구간마다 화자 무시 agn · 세션 순열 고정 cp(회의 전체에서 고른 짝을 그대로) · 구간 최적 cp 를 낸다.
# 긴 위치(학습 창은 수천 위치, max_position_embeddings 65,536 — AMI 회의는 2.3 만–8.7 만)에서 받아쓰기·귀속이 무너지는지, 회의 중간에 짝이 바뀌는지 본다.
# 구간 경계 근처 단어는 지연만큼 이웃 구간으로 갈 수 있다(B 분에 비해 작다). 위치 = prefix + 2 × 청크 + 그때까지 낸 토큰(디코드 규칙의 어림). 읽기만 한다.
#   PYTHONPATH=/soundai/users/tskim/VAPKT python amifull_timeline.py <결과 폴더> [--data <ami-bench-v1>] [--bin-min 5] [--deltas 10]
import argparse, collections, glob, json, os, sys
from itertools import permutations

sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from experiments.cst_stage3_eval import window_ref, windows_of, words_of
from vapasr.data.cst_dialogue import SLOTS
from vapasr.data.cst_eval import ed, parse_emitted, units

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1")
ap.add_argument("--bin-min", type=float, default=5.0); ap.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10]); a = ap.parse_args()
from transformers import AutoTokenizer
conf = json.load(open(os.path.join(a.out, "config-rank0.json"))); assert conf.get("window") == "full", "--window full 결과 폴더가 아니다"
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.out, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
rows = [r for r in rows if r["delta"] in a.deltas]
tok = AutoTokenizer.from_pretrained(conf["checkpoint"]); ck = json.load(open(os.path.join(conf["checkpoint"], "config.json"))); sp = ck["sp_ids"]
ns = argparse.Namespace(corpora=conf["corpora"], data=a.data, split=None, deltas=a.deltas, limit_windows=conf["limit_windows"], window="full",
                        backchannel=tuple(conf["backchannel"]) if conf.get("backchannel") else None, max_speakers=conf.get("max_speakers", 2), declare=conf.get("declare"))
sets, items = windows_of(ns, tok); index = {f"{sets[c].items[i][0]}@{sets[c].items[i][1]}": (c, i) for c, i in items}
spk = tuple(sp[t] for t in SLOTS[: len(ck.get("slot_tokens") or SLOTS[:2])]); EOT = sp["<EOT>"]; B = a.bin_min * 60
acc = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    c, i = index[r["id"]]; ds = sets[c]
    ref_w, _, mask, _ = window_ref(ds, i, r["delta"], EOT, tok)
    hs, _ = parse_emitted([tuple(e) for e in r["emitted"]], spk, EOT, drop_chunks=mask); hyp_w = [words_of(tok, h) for h in hs]
    n = max(2, len(ref_w), len(hyp_w)); R = [ref_w[k] if k < len(ref_w) else [] for k in range(n)]; H = [hyp_w[k] if k < len(hyp_w) else [] for k in range(n)]
    perm = r["perm"] + [k for k in range(n) if k not in r["perm"]]                             # 세션 순열(가설 슬롯 → 참조 슬롯)
    pre = len(ds.cst_prefix(ds.slots(i)[1][: len(spk)], r["delta"])); emk = sorted(k for k, _ in r["emitted"])
    for b in range(int(r["L"] // B) + 1):
        t0, t1 = b * B, (b + 1) * B
        U = lambda ws: [u for w, t in ws if t0 <= t < t1 for u in units(w, r["lang"])]
        Rb, Hb = [U(x) for x in R], [U(x) for x in H]
        if not sum(map(len, Rb)): continue
        cost = [[ed(Rb[p], Hb[h]) for h in range(n)] for p in range(n)]
        ra = [u for w, t in sorted([x for x in sum(R, []) if t0 <= x[1] < t1], key=lambda x: x[1]) for u in units(w, r["lang"])]
        ha = [u for w, t in sorted([x for x in sum(H, []) if t0 <= x[1] < t1], key=lambda x: x[1]) for u in units(w, r["lang"])]
        k1 = int(t1 / 0.08); pos = pre + 2 * min(k1, r["K"]) + sum(1 for k in emk if k < k1)
        g = acc[(c, r["delta"], b)]; g.update(n_ref=sum(map(len, Rb)), agn=ed(ra, ha), cp_fix=sum(cost[perm[h]][h] for h in range(n)),
                                              cp_loc=min(sum(cost[p[h]][h] for h in range(n)) for p in permutations(range(n))), meet=1, pos=pos)
for c in sorted({k[0] for k in acc}):
    for dl in a.deltas:
        print(f"== {c} δ{dl} — {a.bin_min:g} 분 구간(회의 합): agn · cp(세션 짝 고정) · cp(구간 최적 짝) · 귀속 몫(고정 − agn) · 회의 수 · 끝 위치 평균")
        for b in sorted(k[2] for k in acc if k[0] == c and k[1] == dl):
            g = acc[(c, dl, b)]; n = g["n_ref"] or 1
            print(f"  {b * a.bin_min:4.0f}–{(b + 1) * a.bin_min:3.0f} 분  agn {100 * g['agn'] / n:6.2f} · cp 고정 {100 * g['cp_fix'] / n:6.2f} · cp 구간 {100 * g['cp_loc'] / n:6.2f} · "
                  f"귀속 {100 * (g['cp_fix'] - g['agn']) / n:6.2f} · 회의 {g['meet']:2d} · 위치 {g['pos'] / g['meet']:8.0f} · 단어 {g['n_ref']}")
