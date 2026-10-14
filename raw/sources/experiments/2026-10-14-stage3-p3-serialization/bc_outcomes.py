import sys, json, collections
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue, build_episodes
from vapasr.data.eot_soft import classify
D = "/soundai/users/tskim/VAPKT-data/data/phase2"
for c in sys.argv[1].split(","):
    base = collections.Counter(); nobc = collections.Counter(); n = 0
    for k, l in enumerate(open(f"{D}/{c}.refined.dialogues.jsonl", encoding="utf-8")):
        if k >= 150: break
        d = Dialogue.from_json(l); eps = build_episodes(d); obs = d.observed()
        short = lambda e: (e.end - e.start) <= 1.0 and len(e.tokens) <= 3
        big = [e for e in eps if not short(e)]
        for e in eps:
            if short(e): continue                                     # 맞장구성 episode 자체의 EOT 는 따로 본다
            n += 1; base[classify(e, eps, obs)] += 1
            nobc[classify(e, [o for o in eps if o.speaker == e.speaker or not short(o)], obs)] += 1
    tot = sum(base.values())
    f = lambda C: {k: round(C[k] / tot, 3) for k in ("shift", "silence", "both", "hold_long", "hold_short", "unobserved")}
    print(c, "episodes(>1 s or >3 tok)", tot, "\n   as-is      ", f(base), "\n   ignore-bc  ", f(nobc))
