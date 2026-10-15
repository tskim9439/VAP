import json, collections, sys
D = "/soundai/users/tskim/VAPKT-data/data/phase2"
for c in sys.argv[1].split(","):
    n = dup = tok = duptok = dense = 0; convs = collections.Counter()
    for l in open(f"{D}/{c}.refined.dialogues.jsonl", encoding="utf-8"):
        d = json.loads(l); prev = {}
        for u in sorted(d["utterances"], key=lambda u: (u["speaker"], u["start"])):
            t = u.get("text") or ""; k = len(u.get("tokens") or [])
            if not t: continue
            n += 1; tok += k; dur = max(0.05, u["end"] - u["start"])
            if t == prev.get(u["speaker"]): dup += 1; duptok += k; convs[d["conv_id"]] += 1
            if k / dur > 12: dense += 1                                            # 초당 12 토큰 넘게(사람 발화로 불가능한 밀도)
            prev[u["speaker"]] = t
    print(c, f"utts {n} · 같은 화자 직전과 같은 전사 {dup} ({dup / max(1, n):.1%}) · 그 토큰 {duptok}/{tok} ({duptok / max(1, tok):.1%}) · 토큰 밀도 >12/s {dense} · 해당 대화 {len(convs)}")
