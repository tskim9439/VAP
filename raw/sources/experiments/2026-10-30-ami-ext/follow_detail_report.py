# 활동 추종 세부 보고(2026-10-30): cst_stage3_eval --follow-detail 행(follow_detail — cst_eval.follow_stats detail 꼴)을 코퍼스·공식 dev/eval 로 나눠,
# LLM 이 활동을 어긴 단어(다른 슬롯)가 말 차례의 어디에서·어떤 화자 상황에서 나는지 센다. 읽기만 한다.
#   python3 follow_detail_report.py <결과 폴더> [<폴더> …]   (vapasr 가 import 되지 않아도 된다)
# detail 칸: [범주(0 on·1 다른 슬롯·2 무음), f, 단어 슬롯 s, 활동 슬롯 e, 창 안 활성 슬롯 수, s 가 마지막 활성이던 뒤 프레임, e 의 차례 시작 뒤 프레임,
#             직전 화자 = s, LLM 맞음, 활동 맞음, e 가 그 차례 전에 쉰 프레임]
import glob, json, os, sys
MAN = json.load(open("/soundai/users/tskim/VAPKT-data/data/ami-bench-v1/manifest.json"))
SUB = {**{m: "dev" for m in MAN["dev"]}, **{m: "eval" for m in MAN["eval"]}}
FS = 0.08                                                                                          # 프레임 = 80 ms


def bins(x, edges):
    """초 x 의 구간 번호(edges 오름차순 경계, 마지막 칸은 경계 이상)."""
    for i, b in enumerate(edges):
        if x < b: return i
    return len(edges)


def rate_table(D, key, edges, names, title):
    """on·다른 슬롯 단어(활동이 있는 단어)를 key 구간별로 나눠 어긴 비율과 어긴 단어의 몫, 그 안에서 활동만 맞음 비율."""
    tot = [[0, 0, 0, 0] for _ in names]                                                            # on, 다른 슬롯, 다른 슬롯 중 활동만 맞음, 다른 슬롯 중 LLM 만 맞음
    for r in D:
        if r[0] not in (0, 1): continue
        k = key(r)
        if k is None: continue
        t = tot[k]; t[r[0]] += 1
        if r[0] == 1 and r[9] >= 0: t[2] += int(r[9] == 1 and r[8] == 0); t[3] += int(r[8] == 1 and r[9] == 0)
    n1 = sum(t[1] for t in tot) or 1
    cells = []
    for nm, (on, off, ao, lo) in zip(names, tot):
        n = on + off
        cells.append(f"{nm} 어김 {100 * off / max(1, n):4.1f} %(몫 {100 * off / n1:4.1f} %, 활동만 {100 * ao / max(1, off):3.0f} · LLM 만 {100 * lo / max(1, off):3.0f}, 단어 {n})")
    return f"    {title}: " + " | ".join(cells)


def report(rows):
    D = [d for r in rows for d in r.get("follow_detail", [])]
    if not D: return ["    follow_detail 없음"]
    n = len(D); off = [d for d in D if d[0] == 1]; no = max(1, len(off))
    out = [f"    단어 {n} · on {100 * sum(d[0] == 0 for d in D) / n:.1f} % · 다른 슬롯 {100 * len(off) / n:.1f} % · 무음 {100 * sum(d[0] == 2 for d in D) / n:.1f} %"]
    out.append(rate_table(D, lambda d: bins(d[6] * FS, (0.5, 1.0, 2.0, 5.0)) if d[6] >= 0 else None, (0.5, 1, 2, 5),
                          ["차례 0–0.5 s", "0.5–1 s", "1–2 s", "2–5 s", "5 s–"], "활동 화자의 차례 시작 뒤"))
    out.append(rate_table(D, lambda d: (0 if d[10] < 0 else 1 + bins(d[10] * FS, (5.0, 30.0))) if d[6] >= 0 else None, None,
                          ["첫 차례", "쉼 <5 s", "5–30 s", "30 s–"], "활동 화자가 그 차례 전에 쉰 시간"))
    out.append(rate_table(D, lambda d: (0 if d[4] == 1 else 1) if d[4] >= 1 else None, None, ["단독", "겹침"], "창 안 활성 화자 수"))
    ps = sum(d[7] for d in off)
    sb = [0] * 5
    for d in off: sb[0 if d[5] < 0 else 1 + bins(d[5] * FS, (1.0, 5.0, 30.0))] += 1
    out.append(f"    다른 슬롯 단어 중 LLM 슬롯 = 활동 화자 차례의 직전 화자 {100 * ps / no:.1f} % · LLM 슬롯이 마지막으로 말한 뒤: 처음 {100 * sb[0] / no:.1f} · <1 s {100 * sb[1] / no:.1f}"
               f" · 1–5 s {100 * sb[2] / no:.1f} · 5–30 s {100 * sb[3] / no:.1f} · 30 s– {100 * sb[4] / no:.1f} %")
    return out


for d in sys.argv[1:]:
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(d, "predictions-rank*.jsonl"))) for l in open(p, encoding="utf-8")]
    print(f"== {d} — 활동 추종 세부(창 [f − 3, f + 1], 다른 슬롯 = LLM 슬롯이 꺼지고 다른 슬롯이 켜짐)")
    for c in sorted({r["corpus"] for r in rows}):
        for sub in ("eval", "dev"):
            rr = [r for r in rows if r["corpus"] == c and SUB.get(r["id"].split(":", 1)[1].split("@")[0]) == sub]
            if not rr: continue
            print(f"  {c} {sub}(회의·창 {len(rr)})")
            for line in report(rr): print(line)
