# E11 지연 완화 비교(2026-11-02, 묶음 q17-e11-relax — wiki output-diarizer-replacement-20261031 11 절): 학습 없이 지연만 늘린 E11 의 AMI eval 결과를 한 표로.
#   E11(N3D 청크 4 · 인코더 우측 문맥 3) · diar10(N3D 청크 124 ≈ 9.9 s) · enc13(인코더 우측 문맥 13) · both(둘 다 — 창과 회의 통째).
#   표준 재채점(<원>-wn)·출력 재귀속(<원 또는 -follow>-relabel3-1-wn)·활동 추종(ami-report)을 읽기만 한다. 없는 칸은 –.
#   python3 relax_report.py   (vapasr 를 import 하지 않는다)
import glob, json, os, re, sys
R = "/soundai/users/tskim/VAPKT-data/results"
ROWS = [("E11 · N3D 청크 4 · 인코더 rc 3", "stage3-eval-amifull-c30-d10-q17-e11-final", "stage3-eval-ami-d10-q17-e11-final"),
        ("diar10 · N3D 청크 124", None, "stage3-eval-ami-d10-n3dc124-q17-e11-final"),
        ("enc13 · 인코더 rc 13", None, "stage3-eval-ami-d10-rc13-q17-e11-final"),
        ("both · N3D 124 + rc 13", "stage3-eval-amifull-c30-n3dc124-rc13-d10-q17-e11-final", "stage3-eval-ami-d10-n3dc124-rc13-q17-e11-final")]
C = ("ami-ihmmix", "ami-sdm")


def meeteval(path):
    """meeteval-whisper.txt → {(코퍼스, 분할): cpWER}."""
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"\s+(ami-\w+)\s+(eval|dev)\s+δ10 meeteval cpWER\s+([0-9.]+)", l)
            if m: out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def report(path):
    """ami-report.txt → {(코퍼스, 분할): dict(cp, agn[, off])}."""
    out = {}
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            m = re.match(r"\s+(ami-\w+)\s+(eval|dev)\s+δ10 cpWER\s+([0-9.]+) · agn\s+([0-9.]+)", l)
            if not m: continue
            d = dict(cp=float(m.group(3)), agn=float(m.group(4)))
            f = re.search(r"활동 추종 on ([0-9.]+) % · 다른 슬롯 ([0-9.]+) %", l)
            if f: d.update(on=float(f.group(1)), off=float(f.group(2)))
            out[(m.group(1), m.group(2))] = d
    return out


def cell(d, key=None, sub="eval", fmt="{:.2f}"):
    v = [d.get((c, sub)) for c in C]
    v = [x.get(key) if isinstance(x, dict) else x for x in v] if key else v
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


H = ("판", "회의 통째 표준 cp", "+ 재귀속", "회의 Whisper agn", "어김 %", "창 표준 cp", "창 agn(우리)")
for sub in ("eval", "dev"):
    print(f"AMI {sub}, IHM-mix / SDM — δ10(표준 = Whisper + meeteval), 어김은 단어 대비 %, 활동 = 디코드가 받은 슬롯 활동")
    print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
    for name, full, win in ROWS:
        if full:
            src = f"{R}/{full}-follow" if os.path.isdir(f"{R}/{full}-follow") else f"{R}/{full}"   # E11 은 활동을 -follow 에서 다시 계산했다
            st, rl, wn, rp = meeteval(f"{R}/{full}-wn/meeteval-whisper.txt"), meeteval(f"{src}-relabel3-1-wn/meeteval-whisper.txt"), report(f"{R}/{full}-wn/ami-report.txt"), report(f"{src}/ami-report.txt")
        else: st = rl = wn = rp = {}
        ws, wr = meeteval(f"{R}/{win}-wn/meeteval-whisper.txt"), report(f"{R}/{win}/ami-report.txt")
        print("| " + " | ".join((name, cell(st, sub=sub), cell(rl, sub=sub), cell(wn, "agn", sub), cell(rp, "off", sub, "{:.1f}"), cell(ws, sub=sub), cell(wr, "agn", sub))) + " |")
    print()
