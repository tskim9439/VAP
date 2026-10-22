# AMI 공식 분할 비교(2026-10-23): cst_stage3_eval 원출력(ami-bench-v1 — ami-ihmmix·ami-sdm)으로 모델마다 공식 eval·dev × δ 의 cpWER·agn·귀속 몫(cp − agn)을 내고,
# 기준 모델 대비 차이를 회의 단위 짝 bootstrap(회의를 복원 추출, 같은 창끼리 짝 — 창 id 가 다르면 멈춘다)의 95 % 구간으로 낸다. 읽기만 한다.
#   python3 ami_compare.py --base <기준 결과 폴더> <비교 폴더> [<폴더> …] [--n 1000] [--seed 0]   (vapasr 가 import 되는 python)
import argparse, glob, json, os, random, sys
from collections import defaultdict

MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
meeting = lambda rid: rid.split(":", 1)[1].split("@")[0]


def load(d):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "predictions-rank*.jsonl"))) for l in open(f, encoding="utf-8")]
    if not rows: sys.exit(f"원출력 없음: {d}")
    out = defaultdict(lambda: defaultdict(dict))                                              # (corpus, sub, δ) → 회의 → 창 id → (cp, agn, n_ref)
    for r in rows: out[(r["corpus"], SUB[meeting(r["id"])], r["delta"])][meeting(r["id"])][r["id"]] = (r["cp"], r["agn"], r["n_ref"])
    return out


def rate(groups, ms):
    cp = agn = n = 0
    for m in ms:
        for c, a, k in groups[m].values(): cp += c; agn += a; n += k
    return 100 * cp / max(1, n), 100 * agn / max(1, n)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    base = load(a.base); name = lambda d: os.path.basename(d.rstrip("/")).replace("stage3-eval-ami-", "")
    print(f"기준 {name(a.base)} · 차이 = 비교 − 기준, 회의 짝 bootstrap {a.n} 회 95 % 구간(* = 0 을 포함하지 않음)")
    for key in sorted(base):
        b = base[key]; ms = sorted(b); bcp, bagn = rate(b, ms)
        print(f"\n{key[0]:10s} {key[1]:4s} δ{key[2]:<2d} 회의 {len(ms)} · 창 {sum(len(b[m]) for m in ms)}")
        print(f"  {name(a.base):32s} cp {bcp:6.2f} · agn {bagn:6.2f} · 귀속 {bcp - bagn:5.2f}")
        for d in a.dirs:
            o = load(d).get(key)
            if not o: print(f"  {name(d):32s} (이 칸 없음)"); continue
            if sorted(o) != ms or any(sorted(o[m]) != sorted(b[m]) for m in ms): sys.exit(f"창 구성이 다르다: {d} {key}")
            ocp, oagn = rate(o, ms); rnd = random.Random(a.seed); dcp, dat = [], []
            for _ in range(a.n):
                s = [rnd.choice(ms) for _ in ms]; x, y = rate(o, s), rate(b, s); dcp.append(x[0] - y[0]); dat.append((x[0] - x[1]) - (y[0] - y[1]))
            ci = lambda v: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1])
            (c0, c1), (t0, t1) = ci(dcp), ci(dat); st = lambda lo, hi: "*" if lo > 0 or hi < 0 else ""
            print(f"  {name(d):32s} cp {ocp:6.2f} · agn {oagn:6.2f} · 귀속 {ocp - oagn:5.2f} │ Δcp {ocp - bcp:+6.2f} [{c0:+.2f}, {c1:+.2f}]{st(c0, c1)}"
                  f" · Δ귀속 {(ocp - oagn) - (bcp - bagn):+6.2f} [{t0:+.2f}, {t1:+.2f}]{st(t0, t1)}")


if __name__ == "__main__":
    main()
