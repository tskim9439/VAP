# G2b 학생과 종단간 G3(2026-11-03 — 묶음 q17-graft·q17-u3-asr·q17-g3 G3_RUNS, wiki output-diarizer-replacement-20261031 13 절)의 결과를 한데.
#   ① 화자 분리 DER(회의 통째, 스트리밍 청크 4, 최적 짝): N3D 교사 · G2 · G2b  ② ASR 지킴(학생 인코더 + N3D 활동, 창)
#   ③ AMI 회의 통째·창: E11 · G3(E11 + G2) · G3b(E11 + G2b) · E13 · E13 + G2b — 표준 재채점·재귀속·추종·LLM DER 을 읽기만 한다. 없는 칸은 –.
#   python3 g3b_report.py [GRAFT_RUN(기본 q17-graft-g2b)]   (vapasr 를 import 하지 않는다)
import os, re, sys
R = "/soundai/users/tskim/VAPKT-data/results"
GR = sys.argv[1] if len(sys.argv) > 1 else "q17-graft-g2b"; G2 = "q17-graft-g2-lora"
C = ("ami-ihmmix", "ami-sdm")


def der(path):
    """diar_eval summary.log → {(코퍼스, 분할): (DER 최적 짝, miss, fa, conf_best)}."""
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"(ami-\w+)/(eval|dev)\s.*최적 짝\s+([0-9.]+)\(miss ([0-9.]+) · fa ([0-9.]+) · conf [0-9.]+/([0-9.]+)\)", l)
            if m: out[(m.group(1), m.group(2))] = tuple(float(m.group(k)) for k in (3, 4, 5, 6))
    return out


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


def cell(d, key=None, sub="eval", fmt="{:.2f}"):
    v = [d.get((c, sub)) for c in C]
    v = [x.get(key) if isinstance(x, dict) else x for x in v] if key else v
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


print("① 화자 분리 DER — 회의 통째, 스트리밍 청크 4, 최적 짝(miss · fa · 혼동), IHM-mix / SDM")
print("| 화자 분리기 | eval | dev |"); print("|---|---|---|")
for name, d in (("N3D(교사)", "diar-n3d-c4-session"), ("G2 학생", f"diar-graft-{G2}-c4-session"), ("G2b 학생", f"diar-graft-{GR}-c4-session")):
    x = der(f"{R}/{d}/summary.log")
    f = lambda sub: " / ".join("–" if (c, sub) not in x else f"{x[(c, sub)][0]:.2f}({x[(c, sub)][1]:.1f} · {x[(c, sub)][2]:.1f} · {x[(c, sub)][3]:.1f})" for c in C)
    print(f"| {name} | {f('eval')} | {f('dev')} |")
print()
print("② ASR 지킴 — E11 에 학생 인코더 + N3D 활동, 창 δ10, 표준 cp · 창 agn(우리), IHM-mix / SDM")
print("| 인코더 | eval cp | eval agn | dev cp | dev agn |"); print("|---|---|---|---|---|")
for name, w in (("원 인코더(E11)", "stage3-eval-ami-d10-q17-e11-final"), ("G2 인코더", f"stage3-eval-ami-d10-enc{G2}-q17-e11-final"), ("G2b 인코더", f"stage3-eval-ami-d10-enc{GR}-q17-e11-final")):
    ws, wr = meeteval(f"{R}/{w}-wn/meeteval-whisper.txt"), report(f"{R}/{w}/ami-report.txt")
    print(f"| {name} | {cell(ws)} | {cell(wr, 'agn')} | {cell(ws, sub='dev')} | {cell(wr, 'agn', 'dev')} |")
print()
ROWS = [("E11 · N3D", "stage3-eval-amifull-c30-d10-q17-e11-final", "stage3-eval-ami-d10-q17-e11-final"),
        ("G3 · E11 + G2(인코더 하나)", f"stage3-eval-amifull-c30-enc{G2}-gx{G2}-d10-q17-e11-final", f"stage3-eval-ami-d10-enc{G2}-gx{G2}-q17-e11-final"),
        ("G3b · E11 + G2b(인코더 하나)", f"stage3-eval-amifull-c30-enc{GR}-gx{GR}-d10-q17-e11-final", f"stage3-eval-ami-d10-enc{GR}-gx{GR}-q17-e11-final"),
        ("E13 · N3D", "stage3-eval-amifull-c30-d10-q17-e13-final", "stage3-eval-ami-d10-q17-e13-final"),
        ("E13 + G2b(인코더 하나)", f"stage3-eval-amifull-c30-enc{GR}-gx{GR}-d10-q17-e13-final", f"stage3-eval-ami-d10-enc{GR}-gx{GR}-q17-e13-final")]
H = ("판", "회의 통째 표준 cp", "+ 재귀속", "회의 Whisper agn", "어김 %", "LLM DER 놓침·헛·혼동(회의, IHM-mix)", "창 표준 cp", "창 agn(우리)")
for sub in ("eval", "dev"):
    print(f"③ AMI {sub}, IHM-mix / SDM — δ10(표준 = Whisper + meeteval), 어김은 단어 대비 %")
    print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
    for name, full, win in ROWS:
        src = f"{R}/{full}-follow" if os.path.isdir(f"{R}/{full}-follow") else f"{R}/{full}"     # E11 은 활동을 -follow 에서 다시 계산했다
        st, rl, wn, rp = meeteval(f"{R}/{full}-wn/meeteval-whisper.txt"), meeteval(f"{src}-relabel3-1-wn/meeteval-whisper.txt"), report(f"{R}/{full}-wn/ami-report.txt"), report(f"{src}/ami-report.txt")
        ws, wr = meeteval(f"{R}/{win}-wn/meeteval-whisper.txt"), report(f"{R}/{win}/ami-report.txt")
        x = rp.get(("ami-ihmmix", sub)) or {}
        dd = "–" if "miss" not in x else f"{x['miss']:.1f} · {x['fa']:.1f} · {x['conf']:.1f}"
        print("| " + " | ".join((name, cell(st, sub=sub), cell(rl, sub=sub), cell(wn, "agn", sub), cell(rp, "off", sub, "{:.1f}"), dd, cell(ws, sub=sub), cell(wr, "agn", sub))) + " |")
    print()
