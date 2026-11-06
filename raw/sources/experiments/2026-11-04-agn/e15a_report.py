# E15a(인코더 위층 공동 학습 + 겹침 많은 셋 가중, 2026-11-04 — wiki output-agn-reduction-20261104)의 AMI 결과를 E13 과 한 표로 + 대화 번역(dlg-summary δ10)을 E13 과.
#   종단간(활동 = G2b 학생) 비교가 중심이다: E13 + G2b 대 E15a + G2b. 표준 재채점·재귀속·원 채점·agn 분해(agn_diag json)를 읽기만 한다. 없는 칸은 –.
#   python3 e15a_report.py [RUN[,RUN…](기본 q17-e15a — 2026-11-05 절제 q17-e15a-enc 와 함께 쉼표로)] [GRAFT_RUN(기본 q17-graft-g2b)]   (vapasr 를 import 하지 않는다)
import glob, json, os, re, sys
R = "/soundai/users/tskim/VAPKT-data/results"; T = "/soundai/users/tskim/VAPKT-data/tmp"
RUNS = (sys.argv[1] if len(sys.argv) > 1 else "q17-e15a").split(","); GR = sys.argv[2] if len(sys.argv) > 2 else "q17-graft-g2b"
C = ("ami-ihmmix", "ami-sdm")
ROWS = [("E13 · N3D", "stage3-eval-amifull-c30-d10-q17-e13-final", "stage3-eval-ami-d10-q17-e13-final"),
        ("E13 + G2b(종단간)", f"stage3-eval-amifull-c30-enc{GR}-gx{GR}-d10-q17-e13-final", f"stage3-eval-ami-d10-enc{GR}-gx{GR}-q17-e13-final")] + \
       [(f"{run}{'@' + tag if tag != 'final' else ''} + G2b(종단간{', ' + sub + ' 분할만' if sub else ''})" if act != "n3d" else f"{run}{'@' + tag if tag != 'final' else ''} · N3D{'(' + sub + ' 분할만)' if sub else ''}",
         f"stage3-eval-amifull-c30{'' if act == 'n3d' else '-gx' + GR}{'-sub' + sub if sub else ''}-d10-{run}-{tag}",
         f"stage3-eval-ami-d10{'' if act == 'n3d' else '-gx' + GR}{'-sub' + sub if sub else ''}-{run}-{tag}") for run, sub, act, tag in ((x.split(":") + ["", "", ""])[:3] + [((x.split(":") + ["", "", ""])[3] or "final")] for x in RUNS)]   # RUN:eval = 빠른 평가(-sub<분할>), RUN:eval:n3d = 활동 N3D(2026-11-06 E15b — 산출 이름에 -gx 없음), RUN:eval:n3d:ck4250 = 체크포인트 태그(기본 final)


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


def agn_parts(paths):
    """agn_diag --json 들 → {폴더 이름: {(코퍼스, 분할): (혼자 놓침, 둘 놓침, 셋+ 놓침)}}."""
    out = {}
    for pred, v in (kv for path in paths for kv in json.load(open(path)).items()):
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


AG = agn_parts(sorted(glob.glob(f"{T}/agn/agn-*.json")))
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
for run, tag in [("q17-e13", "final")] + [(x.split(":")[0], (x.split(":") + ["", "", ""])[3] or "final") for x in RUNS]:
    p = f"{R}/stage4-eval-{run}-{tag}/dlg-summary.log"
    if not os.path.exists(p): print(f"  {run}@{tag}: –"); continue
    for l in open(p, encoding="utf-8"):
        if "/d10" in l: print(f"  {run}{'@' + tag if tag != 'final' else ''}: " + re.sub(r" · EOT R [0-9.]+ P [0-9.]+", "", l.strip())[:200])
