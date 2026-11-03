# E14(비귀속 칸 + 확률 활동, 2026-11-03 — wiki output-e14-xslot-20261103)의 AMI 결과를 E11·E13 과 한 표로 + 대화 번역(dlg-summary δ10 BLEU·받아쓰기)을 E6·E13 과.
#   표준 재채점(<원>-wn)·출력 재귀속(<원 또는 -follow>-relabel3-1-wn)·원 채점(ami-report)·원출력(X 칸 단어 몫)·게이트 진단(gate_diag json)을 읽기만 한다. 없는 칸은 –.
#   python3 e14_report.py   (vapasr 를 import 하지 않는다)
import glob, json, os, re
R = "/soundai/users/tskim/VAPKT-data/results"; T = "/soundai/users/tskim/VAPKT-data/tmp/e13-gate"
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json")); SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
ROWS = [("E11", "stage3-eval-amifull-c30-d10-q17-e11-final", "stage3-eval-ami-d10-q17-e11-final"),
        ("E13(게이트)", "stage3-eval-amifull-c30-d10-q17-e13-final", "stage3-eval-ami-d10-q17-e13-final"),
        ("E14(게이트 + X 칸 + 확률 활동)", "stage3-eval-amifull-c30-d10-q17-e14-final", "stage3-eval-ami-d10-q17-e14-final")]
C = ("ami-ihmmix", "ami-sdm")


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


def xshare(path):
    """원출력 → {(코퍼스, 분할): X 칸 단어 / 가설 단어 %}."""
    acc = {}
    for f in glob.glob(os.path.join(path, "predictions-rank*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); k = (r["corpus"], SUB[r["id"].split(":", 1)[1].split("@")[0]]); a = acc.setdefault(k, [0, 0]); a[0] += r.get("n_x", 0); a[1] += r["n_hyp"]
    return {k: 100.0 * a[0] / max(1, a[1]) for k, a in acc.items()}


def diag(path):
    if not os.path.exists(path): return {}
    d = json.load(open(path)); return {tuple(k.split("/")): (100 * v["any"] / v["n_ref"], 100 * v["own"] / v["n_ref"]) for k, v in d.items()}


def cell(d, key=None, sub="eval", fmt="{:.2f}"):
    v = [d.get((c, sub)) for c in C]
    v = [x.get(key) if isinstance(x, dict) else x for x in v] if key else v
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


H = ("판", "회의 통째 표준 cp", "+ 재귀속", "Whisper agn", "받아씀 · 짝 슬롯 %", "X 칸 단어 %", "어김 %", "LLM DER 놓침·헛·혼동(IHM-mix)", "창 표준 cp", "창 agn(우리)")
for sub in ("eval", "dev"):
    print(f"AMI {sub}, IHM-mix / SDM — δ10(표준 = Whisper + meeteval). 받아씀 = gate_diag(같은 글자·±0.6 s), X 칸 단어 % = 가설 단어 중 X 칸에서 나온 몫")
    print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
    for name, full, win in ROWS:
        src = f"{R}/{full}-follow" if os.path.isdir(f"{R}/{full}-follow") else f"{R}/{full}"
        st, rl, wn, rp = meeteval(f"{R}/{full}-wn/meeteval-whisper.txt"), meeteval(f"{src}-relabel3-1-wn/meeteval-whisper.txt"), report(f"{R}/{full}-wn/ami-report.txt"), report(f"{src}/ami-report.txt")
        ws, wr = meeteval(f"{R}/{win}-wn/meeteval-whisper.txt"), report(f"{R}/{win}/ami-report.txt")
        dg, xs = diag(f"{T}/diag-{os.path.basename(src)}.json"), xshare(f"{R}/{full}")
        rec = " / ".join("–" if (c, sub) not in dg else f"{dg[(c, sub)][0]:.1f} · {dg[(c, sub)][1]:.1f}" for c in C)
        x = rp.get(("ami-ihmmix", sub)) or {}
        der = "–" if "miss" not in x else f"{x['miss']:.1f} · {x['fa']:.1f} · {x['conf']:.1f}"
        print("| " + " | ".join((name, cell(st, sub=sub), cell(rl, sub=sub), cell(wn, "agn", sub), rec, cell(xs, sub=sub, fmt="{:.1f}"), cell(rp, "off", sub, "{:.1f}"), der,
                                 cell(ws, sub=sub), cell(wr, "agn", sub))) + " |")
    print()
print("대화 번역(held-out 창 100 개, δ10) — BLEU · 받아쓰기 CER/WER")
for run in ("q17-e6", "q17-e13", "q17-e14"):
    p = f"{R}/stage4-eval-{run}-final/dlg-summary.log"
    if not os.path.exists(p): print(f"  {run}: –"); continue
    for l in open(p, encoding="utf-8"):
        if "/d10" in l: print(f"  {run}: " + re.sub(r" · EOT R [0-9.]+ P [0-9.]+", "", l.strip())[:200])
