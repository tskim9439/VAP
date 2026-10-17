# 실제 학습 대화(held-out 제외)를 합성 점검과 같은 overlap_stats 로 — 대화 전체 기준 분당 교대·겹침 비율·분당 맞장구(≤1 s·≤3 토큰). 읽기만 한다.
import sys, json, random
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue, build_episodes
from vapasr.data.sim_dialogue import overlap_stats
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; split = json.load(open(f"{D}/splits/v2.json"))
for lang, cs in (("English", ["otoSpeech"]), ("Korean", ["aihub71631", "aihub134-1", "aihub134-2"]), ("German", ["kec", "ztc", "nsc"])):
    rows = []
    for c in cs:
        held = set(split["corpora"].get(c, {}).get("heldout", [])); lines = open(f"{D}/{c}.refined.dialogues.jsonl", encoding="utf-8").readlines()
        for l in random.Random(0).sample(lines, min(150, len(lines))):
            d = Dialogue.from_json(l)
            if d.conv_id in held: continue
            o = overlap_stats(d); eps = build_episodes(d)
            o["bc"] = sum(1 for e in eps if e.end - e.start <= 1.0 and len(e.tokens) <= 3); o["corpus"] = c; rows.append(o)
    T = sum(r["duration_s"] for r in rows)
    print(f"{lang}: 대화 {len(rows)} · 분당 교대 {sum(r['changes_per_min'] * r['duration_s'] for r in rows) / T:.2f} · 겹침 {100 * sum(r['overlap_s'] for r in rows) / T:.1f} % · "
          f"말소리 {100 * sum(r['speech_s'] for r in rows) / T:.0f} % · 분당 맞장구 {60 * sum(r['bc'] for r in rows) / T:.2f}")
