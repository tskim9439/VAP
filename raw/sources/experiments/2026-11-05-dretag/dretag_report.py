#!/usr/bin/env python3
"""디코드 때 재귀속 보고(2026-11-05): 판마다 원 디코드 · 사후 재귀속(-relabel3-1-wn) · 디코드 때 재귀속(-dr<B>-<F>)의 회의 통째 표준 cpWER(Whisper + meeteval,
화자 통째)·Whisper agn(단어마다)·활동 어김·재귀속 셈을 AMI 공식 분할별로 한 표에 쓴다. 읽기만 한다.
  python3 dretag_report.py [B-F(기본 3-1)] [산출 이름의 분할 꼬리(기본 -subeval)]"""
import glob, json, os, re, sys

R = "/soundai/users/tskim/VAPKT-data/results"
W = sys.argv[1] if len(sys.argv) > 1 else "3-1"; ST = sys.argv[2] if len(sys.argv) > 2 else "-subeval"
G = "encq17-graft-g2b-gxq17-graft-g2b-"
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json")); SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
EXT = [("NVIDIA SSA-v2 + N3D(보고)", "13.38 / 20.33"), ("Multitalker Parakeet + N3D(직접, 1.12 s)", "15.86 / 20.50"),
       ("Parakeet + Streaming Sortformer v2(직접)", "17.59 / 34.59"), ("Parakeet HF 카드(보고)", "21.26 / 37.44"), ("NVIDIA SSA-v1(보고)", "22.10 / 39.91")]


def meeteval(path):
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"\s+(ami-\w+)\s+(eval|dev)\s+δ10 meeteval cpWER\s+([0-9.]+)", l)
            if m: out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def agn(path):
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"\s+(ami-\w+)\s+(eval|dev)\s+δ10 cpWER\s+([0-9.]+) · agn\s+([0-9.]+)", l)
            if m: out[(m.group(1), m.group(2))] = float(m.group(4))
    return out


def rows_stats(folder):
    """행에서 활동 어김(follow off_other / n)·디코드 때 재귀속 셈(dretag) — 분할별 합."""
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "predictions-rank*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); k = (r["corpus"], SUB.get(r["id"].split(":", 1)[1].split("@")[0], "?")); o = out.setdefault(k, dict(n=0, off=0, boxes=0, moved=0, tokens=0))
            fo = r.get("follow") or {}; o["n"] += fo.get("n", 0); o["off"] += fo.get("off_other", 0)
            for x in ("boxes", "moved", "tokens"): o[x] += (r.get("dretag") or {}).get(x, 0)
    return out


def cell(d, sub):
    a, b = d.get(("ami-ihmmix", sub)), d.get(("ami-sdm", sub))
    return "–" if a is None and b is None else f"{a if a is not None else float('nan'):.2f} / {b if b is not None else float('nan'):.2f}"


print(f"디코드 때 재귀속(창 {W}) — AMI 회의 통째 δ10, IHM-mix / SDM, 표준 = Whisper + meeteval(화자 통째), agn = Whisper 단어마다")
for sub in ("eval", "dev"):
    print(f"\n== {sub}")
    print("| 판 | 원 디코드 cp | 사후 재귀속 cp | 디코드 때 재귀속 cp | Whisper agn(원 → 디코드 때) | 활동 어김 %(원 → 디코드 때, IHM-mix / SDM) | 옮긴 칸 / 글자 칸(IHM-mix / SDM) |")
    print("|---|---|---|---|---|---|---|")
    for name, base, dr in (("E13 · N3D", "stage3-eval-amifull-c30-d10-q17-e13-final", f"stage3-eval-amifull-c30-dr{W}{ST}-d10-q17-e13-final"),
                           ("E13 + G2b(종단간)", f"stage3-eval-amifull-c30-{G}d10-q17-e13-final", f"stage3-eval-amifull-c30-{G}dr{W}{ST}-d10-q17-e13-final")):
        b, rl, d = meeteval(f"{R}/{base}-wn/meeteval-whisper.txt"), meeteval(f"{R}/{base}-relabel3-1-wn/meeteval-whisper.txt"), meeteval(f"{R}/{dr}-wn/meeteval-whisper.txt")
        ab, ad = agn(f"{R}/{base}-wn/ami-report.txt"), agn(f"{R}/{dr}-wn/ami-report.txt")
        sb, sd = rows_stats(f"{R}/{base}"), rows_stats(f"{R}/{dr}")
        off = lambda S, c: (100 * S[(c, sub)]["off"] / max(1, S[(c, sub)]["n"])) if (c, sub) in S else float("nan")
        mv = lambda c: f"{sd[(c, sub)]['moved']}/{sd[(c, sub)]['boxes']}" if (c, sub) in sd else "–"
        print(f"| {name} | {cell(b, sub)} | {cell(rl, sub)} | {cell(d, sub)} | {cell(ab, sub)} → {cell(ad, sub)} | "
              f"{off(sb, 'ami-ihmmix'):.1f} / {off(sb, 'ami-sdm'):.1f} → {off(sd, 'ami-ihmmix'):.1f} / {off(sd, 'ami-sdm'):.1f} | {mv('ami-ihmmix')} / {mv('ami-sdm')} |")
print("\n외부 기준(회의 통째 표준 cpWER eval, IHM-mix / SDM — 모두 외부 화자 분리기를 쓴다; 우리 δ10 평균 지연 0.92 s)")
for n, v in EXT: print(f"  {n}: {v}")
