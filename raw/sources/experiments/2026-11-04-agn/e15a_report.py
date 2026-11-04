# E15a(인코더 위층 공동 학습 + 겹침 많은 셋 가중, 2026-11-04 — wiki output-agn-reduction-20261104)의 AMI 결과를 E13 과 한 표로 + 대화 번역(dlg-summary δ10)을 E13 과.
#   종단간(활동 = G2b 학생) 비교가 중심이다: E13 + G2b 대 E15a + G2b. 표준 재채점·재귀속·원 채점·agn 분해(agn_diag json)를 읽기만 한다. 없는 칸은 –.
#   python3 e15a_report.py [RUN(기본 q17-e15a)] [GRAFT_RUN(기본 q17-graft-g2b)]   (vapasr 를 import 하지 않는다)
import json, os, re, sys
R = "/soundai/users/tskim/VAPKT-data/results"; T = "/soundai/users/tskim/VAPKT-data/tmp"
RUN = sys.argv[1] if len(sys.argv) > 1 else "q17-e15a"; GR = sys.argv[2] if len(sys.argv) > 2 else "q17-graft-g2b"
C = ("ami-ihmmix", "ami-sdm")
ROWS = [("E13 · N3D", "stage3-eval-amifull-c30-d10-q17-e13-final", "stage3-eval-ami-d10-q17-e13-final"),
        ("E13 + G2b(종단간)", f"stage3-eval-amifull-c30-enc{GR}-gx{GR}-d10-q17-e13-final", f"stage3-eval-ami-d10-enc{GR}-gx{GR}-q17-e13-final"),
        (f"E15a + G2b(종단간, {RUN})", f"stage3-eval-amifull-c30-gx{GR}-d10-{RUN}-final", f"stage3-eval-ami-d10-gx{GR}-{RUN}-final")]


def meeteval(path):
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"\s+(ami-\w+)\s+(eval|dev)\s+δ10 meeteval cpWER\s+([0-9.]+)", l)
            if m: out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def report(path):
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"\s+(ami-\w+)\s+(eval|dev)\s+δ10 cpWER\s+([0-9.]+) · agn\s+([0-9.]+)", l)
            if not m: continue
            d = dict(cp=float(m.group(3)), agn=float(m.group(4)))
            f = re.search(r"활동 추종 on ([0-9.]+) % · 다른 슬롯 ([0-9.]+) %", l)
            if f: d.update(on=float(f.group(1)), off=float(f.group(2)))
            g = re.search(r"LLM DER ([0-9.]+)\(놓침 ([0-9.]+) 헛 ([0-9.]+) 혼동 ([0-9.]+)\)", l)
            if g: d.update(miss=float(g.group(2)), fa=float(g.group(3)), conf=float(g.group(4)))
            out[(m.group(1), m.group(2))] = d
    return out


def agn_parts(path):
    """agn_diag --json → {폴더 이름: {(코퍼스, 분할): (혼자 놓침, 둘 놓침, 셋+ 놓침)}}."""
    if not os.path.exists(path): return {}
    out = {}
    for pred, v in json.load(open(path)).items():
        x = {}
        for k, a in v["agg"].items():
            c, s = k.split("/"); rate = lambda o: 100.0 * (a.get(f"ov:{o}:n", 0) - a.get(f"ov:{o}:ok", 0)) / max(1, a.get(f"ov:{o}:n", 0))
            x[(c, s)] = (rate("1"), rate("2"), rate("3+"))
        out[os.path.basename(pred.rstrip("/"))] = x
    return out


def cell(d, key=None, sub="eval", fmt="{:.2f}"):
    v = [d.get((c, sub)) for c in C]
    v = [x.get(key) if isinstance(x, dict) else x for x in v] if key else v
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


AG = agn_parts(f"{T}/agn/agn-e15a.json")
H = ("판", "회의 통째 표준 cp", "+ 재귀속", "회의 Whisper agn", "놓침 혼자·둘·셋+ %(IHM-mix)", "어김 %", "LLM DER 놓침·헛·혼동(IHM-mix)", "창 표준 cp", "창 agn(우리)")
for sub in ("eval", "dev"):
    print(f"AMI {sub}, IHM-mix / SDM — δ10(표준 = Whisper + meeteval). 놓침 = agn_diag(같은 글자 ±0.6 s, 겹침 = 그 시각 말하는 참조 화자 수)")
    print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
    for name, full, win in ROWS:
        st, rl, wn, rp = meeteval(f"{R}/{full}-wn/meeteval-whisper.txt"), meeteval(f"{R}/{full}-relabel3-1-wn/meeteval-whisper.txt"), report(f"{R}/{full}-wn/ami-report.txt"), report(f"{R}/{full}/ami-report.txt")
        ws, wr = meeteval(f"{R}/{win}-wn/meeteval-whisper.txt"), report(f"{R}/{win}/ami-report.txt")
        x = rp.get(("ami-ihmmix", sub)) or {}; dd = "–" if "miss" not in x else f"{x['miss']:.1f} · {x['fa']:.1f} · {x['conf']:.1f}"
        ov = (AG.get(full) or {}).get(("ami-ihmmix", sub)); ovs = "–" if ov is None else " · ".join(f"{v:.1f}" for v in ov)
        print("| " + " | ".join((name, cell(st, sub=sub), cell(rl, sub=sub), cell(wn, "agn", sub), ovs, cell(rp, "off", sub, "{:.1f}"), dd, cell(ws, sub=sub), cell(wr, "agn", sub))) + " |")
    print()
print("대화 번역(held-out 창 100 개, δ10, N3D 활동) — BLEU · 받아쓰기 CER/WER")
for run in ("q17-e13", RUN):
    p = f"{R}/stage4-eval-{run}-final/dlg-summary.log"
    if not os.path.exists(p): print(f"  {run}: –"); continue
    for l in open(p, encoding="utf-8"):
        if "/d10" in l: print(f"  {run}: " + re.sub(r" · EOT R [0-9.]+ P [0-9.]+", "", l.strip())[:200])
