# E13 추론 게이트 판(2026-11-03 — 묶음 q17-e13-gate, wiki output-e13-gate-trans-20261102)의 AMI 회의 통째 결과를 E11·E13 과 한 표로.
#   표준 재채점(<원>-wn)·출력 재귀속(<원 또는 -follow>-relabel3-1-wn)·원 채점(ami-report — 추종·LLM DER)·원출력(청크당 위치·청크 ms)·게이트 진단(gate_diag json)을 읽기만 한다.
#   없는 칸은 –. python3 gate_report.py [판 태그 …(기본 gt03 gt01 gl0)]   (vapasr 를 import 하지 않는다)
import glob, json, os, re, sys
R = "/soundai/users/tskim/VAPKT-data/results"; T = "/soundai/users/tskim/VAPKT-data/tmp/e13-gate"
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json")); SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
TAGS = sys.argv[1:] or ["gt03", "gt01", "gl0"]
NAME = dict(gt03="게이트 확률 > 0.3", gt01="게이트 확률 > 0.1", gt015="게이트 확률 > 0.15", gt02="게이트 확률 > 0.2", gl0="게이트 끔(태그 4 개)", gl25="게이트 창 25", gl50="게이트 창 50")
ROWS = [("E11(게이트 없음, 태그 4 개)", "stage3-eval-amifull-c30-d10-q17-e11-final"), ("E13(게이트 창 12 · 확률 > 0.5)", "stage3-eval-amifull-c30-d10-q17-e13-final")] + \
       [(f"E13 · {NAME.get(t, t)}", f"stage3-eval-amifull-c30-{t}-d10-q17-e13-final") for t in TAGS]
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


def rows_cost(path):
    """원출력 → {(코퍼스, 분할): (청크당 위치, 청크 ms 평균)}."""
    acc = {}
    for f in glob.glob(os.path.join(path, "predictions-rank*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); k = (r["corpus"], SUB[r["id"].split(":", 1)[1].split("@")[0]]); a = acc.setdefault(k, [0, 0, 0.0, 0])
            a[0] += r.get("positions", 0); a[1] += r["K"]; a[2] += r.get("chunk_ms", 0.0) * r["K"]; a[3] += r["K"] if "chunk_ms" in r else 0
    return {k: (a[0] / max(1, a[1]), a[2] / max(1, a[3])) for k, a in acc.items()}


def diag(path):
    """gate_diag json → {(코퍼스, 분할): (받아씀 %, 짝 슬롯 %)}."""
    if not os.path.exists(path): return {}
    d = json.load(open(path)); return {tuple(k.split("/")): (100 * v["any"] / v["n_ref"], 100 * v["own"] / v["n_ref"]) for k, v in d.items()}


def cell(d, key=None, sub="eval", fmt="{:.2f}"):
    v = [d.get((c, sub)) for c in C]
    v = [x.get(key) if isinstance(x, dict) else x for x in v] if key else v
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


H = ("판", "표준 cp", "+ 재귀속", "Whisper agn", "받아씀 · 짝 슬롯 %", "어김 %", "LLM DER 놓침·헛·혼동(IHM-mix)", "청크당 위치 · ms")
for sub in ("eval", "dev"):
    print(f"AMI {sub} 회의 통째, IHM-mix / SDM — δ10(표준 = Whisper + meeteval). 받아씀 = 참조 단어를 아무 슬롯에서 같은 글자·±0.6 s 로 낸 몫(gate_diag), 짝 슬롯 = cpWER 짝 슬롯에서")
    print("| " + " | ".join(H) + " |"); print("|" + "---|" * len(H))
    for name, full in ROWS:
        src = f"{R}/{full}-follow" if os.path.isdir(f"{R}/{full}-follow") else f"{R}/{full}"        # E11 은 활동을 -follow 에서 다시 계산했다
        st, rl, wn, rp = meeteval(f"{R}/{full}-wn/meeteval-whisper.txt"), meeteval(f"{src}-relabel3-1-wn/meeteval-whisper.txt"), report(f"{R}/{full}-wn/ami-report.txt"), report(f"{src}/ami-report.txt")
        dg = diag(f"{T}/diag-{os.path.basename(src)}.json"); cs = rows_cost(f"{R}/{full}")
        rec = " / ".join("–" if (c, sub) not in dg else f"{dg[(c, sub)][0]:.1f} · {dg[(c, sub)][1]:.1f}" for c in C)
        x = rp.get(("ami-ihmmix", sub)) or {}
        der = "–" if "miss" not in x else f"{x['miss']:.1f} · {x['fa']:.1f} · {x['conf']:.1f}"
        cost = " / ".join("–" if (c, sub) not in cs else f"{cs[(c, sub)][0]:.2f} · {cs[(c, sub)][1]:.0f}" for c in C)
        print("| " + " | ".join((name, cell(st, sub=sub), cell(rl, sub=sub), cell(wn, "agn", sub), rec, cell(rp, "off", sub, "{:.1f}"), der, cost)) + " |")
    print()
