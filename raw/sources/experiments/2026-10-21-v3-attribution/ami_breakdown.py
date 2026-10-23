# AMI(δ10) 오류 분해: 창 화자 수별 cp·agn·귀속, 2 슬롯 구조 하한(창마다 말 많은 두 화자 밖의 참조 단어 비율), 군말 비율과 군말을 뺀 cp. 읽기만.
import json, glob, sys, collections
from itertools import permutations
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.cst_eval import ed, units
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json")); SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
FILL = {"um", "uh", "hmm", "mm", "mmm", "mhm", "hm", "er", "erm", "ah", "eh", "huh", "uhm", "mm-hmm", "uh-huh", "uh-oh"}
R = "/soundai/users/tskim/VAPKT-data/results/stage3-eval-ami-"
def U(s): return [u for w in s.split() for u in units(w, "English")]
def cp_of(refs, hyps, drop=False):
    f = (lambda xs: [x for x in xs if x not in FILL]) if drop else (lambda xs: xs)
    Rs, Hs = [f(U(x)) for x in refs], [f(U(x)) for x in hyps]; n = max(2, len(Rs), len(Hs)); Rs += [[]] * (n - len(Rs)); Hs += [[]] * (n - len(Hs))
    c = [[ed(Rs[r], Hs[h]) for h in range(n)] for r in range(n)]
    return min(sum(c[p[h]][h] for h in range(n)) for p in permutations(range(n))), sum(map(len, Rs))
for name in ("st4-m2a-ck2000", "v3-p1-tel-xlw-final", "v2-p1-final"):
    rows = [json.loads(l) for f in glob.glob(R + name + "/predictions-rank*.jsonl") for l in open(f)]
    rows = [r for r in rows if int(r["delta"]) == 10 and SUB[r["id"].split(":", 1)[1].split("@")[0]] == "eval"]
    for cond in ("ami-ihmmix", "ami-sdm"):
        rr = [r for r in rows if r["corpus"] == cond]; b = collections.defaultdict(lambda: [0, 0, 0, 0])
        floor = tot = fr = fh = nh = 0; cpd = nrd = 0
        for r in rr:
            k = min(int(r["n_spk"]), 4); x = b[k]; x[0] += 1; x[1] += r["cp"]; x[2] += r["agn"]; x[3] += r["n_ref"]
            wc = sorted((len(U(t)) for t in r["ref_text"]), reverse=True); floor += sum(wc[2:]); tot += sum(wc)
            ru = [u for t in r["ref_text"] for u in U(t)]; hu = [u for t in r["hyp_text"] for u in U(t)]
            fr += sum(u in FILL for u in ru); fh += sum(u in FILL for u in hu); nh += len(hu)
            c, n = cp_of(r["ref_text"], r["hyp_text"], drop=True); cpd += c; nrd += n
        n_all = sum(x[3] for x in b.values())
        print(f"== {name} · {cond} eval δ10: cp {100 * sum(x[1] for x in b.values()) / n_all:.1f} · agn {100 * sum(x[2] for x in b.values()) / n_all:.1f} · "
              f"참조 단어 {n_all} · 가설 단어 {nh}({100 * nh / n_all:.0f} %) · 2 슬롯 구조 하한 {100 * floor / tot:.1f} % · 군말 참조 {100 * fr / tot:.1f} % · 가설 {100 * fh / max(1, nh):.1f} % · 군말 뺀 cp {100 * cpd / nrd:.1f}")
        for k in sorted(b): x = b[k]; print(f"   화자 {k}: 창 {x[0]:4d} · 참조 {100 * x[3] / n_all:4.1f} % · cp {100 * x[1] / x[3]:5.1f} · agn {100 * x[2] / x[3]:5.1f} · 귀속 {100 * (x[1] - x[2]) / x[3]:5.1f}")
