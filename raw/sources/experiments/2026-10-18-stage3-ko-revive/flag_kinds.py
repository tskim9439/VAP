# 마스크 발화의 종류: flags 사유별 개수, 토큰(정렬) 유무, raw 전사 유무, meta.missing.
import json, sys
from collections import Counter
for path in sys.argv[1:]:
    name = path.split(".")[0]; c = Counter(); why = Counter()
    for line in open(path):
        d = json.loads(line); miss = set((d.get("meta") or {}).get("missing", []))
        for i, u in enumerate(d["utterances"], 1):
            if i in miss: c["missing"] += 1; continue
            if u.get("flags"):
                c["flag"] += 1; c["flag+tokens"] += bool(u.get("tokens")); c["flag+raw"] += bool((u.get("raw") or "").strip()); c["flag+text"] += bool((u.get("text") or "").strip())
                for f in u["flags"]: why[f.split(":")[0]] += 1
            elif not u.get("tokens"): c["no_tokens"] += 1; c["no_tokens+text"] += bool((u.get("text") or "").strip()); c["no_tokens+raw"] += bool((u.get("raw") or "").strip())
            else: c["ok"] += 1
    print(name, dict(c)); print("   flags:", dict(why.most_common(8)))
