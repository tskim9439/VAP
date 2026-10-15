import json, collections, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.corpora import parse_srt, SPEECH_LABELS
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; gaps = []; between = collections.Counter(); ex = []; n = 0
for l in open(f"{D}/otoSpeech.dialogues.jsonl", encoding="utf-8"):
    d = json.loads(l); prev = {}
    for u in sorted(d["utterances"], key=lambda u: (u["speaker"], u["start"])):
        t = u.get("raw") or ""
        p = prev.get(u["speaker"])
        if p and t == p["raw"] and len(t.split()) >= 4:
            g = u["start"] - p["end"]; gaps.append(g); n += 1
            if len(ex) < 4: ex.append((d["conv_id"], u["speaker"], round(p["start"], 2), round(p["end"], 2), round(u["start"], 2), round(u["end"], 2), t[:60]))
            srt = parse_srt(f"{d['meta']['src']}/speaker_{int(u['speaker']) + 1}_annotation_a.srt")
            for s, e, lab, _ in srt:
                if s >= p["end"] - 1e-3 and e <= u["start"] + 1e-3: between[lab] += 1
        prev[u["speaker"]] = u
import statistics
print("남은 긴(4 단어+) 반복", n, "· 간격 s p50/p90/max", round(statistics.median(gaps), 3), round(sorted(gaps)[int(0.9 * (len(gaps) - 1))], 3), round(max(gaps), 3))
print("사이에 낀 SRT 라벨", dict(between.most_common(8)))
for x in ex: print(" ", x)
