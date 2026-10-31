# E12 장치별 판 비교(2026-10-31, 사용자 "장치별로 나눠 학습하는 판 제출 명령어 알려줘"): E11·E12·섞기만·바꾸기만·잡음만·대조의 AMI eval 결과를 한 표로.
# 표준 재채점(<원>-wn)·출력 재귀속(<원 또는 -follow>-relabel3-1-wn)·활동 추종 세부(<…>-fdetail) 산출을 읽기만 한다. 없는 칸은 –.
#   python3 e12_ablation_report.py   (vapasr 를 import 하지 않는다)
import glob, json, os, re
R = "/soundai/users/tskim/VAPKT-data/results"
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
RUNS = [("E11", "q17-e11"), ("E12(셋 다)", "q17-e12"), ("섞기만", "q17-e12-shuf"), ("바꾸기만", "q17-e12-remap"), ("잡음만", "q17-e12-noise"), ("대조(이어 학습만)", "q17-e12-ctrl")]
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
    """ami-report.txt → {(코퍼스, 분할): dict(cp, agn[, on, off])}."""
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


def types(path):
    """-fdetail 행 → {(코퍼스, 분할): (직전 화자 슬롯 %, 30 s 넘게 말 없던 슬롯 %)} — 어긴 단어 유형을 전체 단어 대비로."""
    agg = {}
    for p in glob.glob(os.path.join(path, "predictions-rank*.jsonl")):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); key = (r["corpus"], SUB.get(r["id"].split(":", 1)[1].split("@")[0]))
            a = agg.setdefault(key, [0, 0, 0])
            for d in r.get("follow_detail", []):
                a[0] += 1
                if d[0] == 1: a[1] += d[7]; a[2] += int(d[5] >= 0 and d[5] * 0.08 >= 30.0)
    return {k: (100 * v[1] / max(1, v[0]), 100 * v[2] / max(1, v[0])) for k, v in agg.items()}


def cell(d, key, f=lambda x: f"{x:.2f}"):
    v = [d.get((c, "eval"), {}) if isinstance(d.get((c, "eval")), dict) else d.get((c, "eval")) for c in C]
    v = [x.get(key) if isinstance(x, dict) else x for x in v] if key else v
    return " / ".join("–" if x is None else f(x) for x in v)


rows = []
for name, run in RUNS:
    full, win = f"{R}/stage3-eval-amifull-c30-d10-{run}-final", f"{R}/stage3-eval-ami-d10-{run}-final"
    src = full + "-follow" if os.path.isdir(full + "-follow") else full                          # E11 은 활동을 -follow 에서 다시 계산했다
    st, rl, rp, wn = meeteval(f"{full}-wn/meeteval-whisper.txt"), meeteval(f"{src}-relabel3-1-wn/meeteval-whisper.txt"), report(f"{src}/ami-report.txt"), report(f"{full}-wn/ami-report.txt")
    ty = types(f"{src}-fdetail"); wst = meeteval(f"{win}-wn/meeteval-whisper.txt")
    tyc = {k: v for k, v in ty.items()}
    rows.append((name, cell(st, None), cell(rl, None), cell(wn, "agn"), cell(rp, "off", lambda x: f"{x:.1f}"),
                 " / ".join("–" if (c, "eval") not in tyc else f"{tyc[(c, 'eval')][0]:.1f}" for c in C),
                 " / ".join("–" if (c, "eval") not in tyc else f"{tyc[(c, 'eval')][1]:.1f}" for c in C), cell(wst, None)))
H = ("판", "회의 통째 표준 cp", "+ 재귀속", "Whisper agn", "어김 %", "직전 화자 슬롯 %", "30 s 빈 슬롯 %", "창 표준 cp")
print("AMI eval, IHM-mix / SDM — 회의 통째 δ10(표준 = Whisper + meeteval), 어김·유형은 단어 대비 %")
print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
for r in rows: print("| " + " | ".join(r) + " |")
