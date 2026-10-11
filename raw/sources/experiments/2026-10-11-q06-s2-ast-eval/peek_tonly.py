import glob, json, collections
R = "/soundai/users/tskim/VAPKT-data/results/ast-covost2-q06-s2-ast"
rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/predictions-rank*.jsonl")) for l in open(f)]
by = collections.defaultdict(dict)
for r in rows: by[r["id"]][str(r["delta"])] = r
d4 = [v["4"] for k, v in by.items() if k.startswith("covost2-en-de/")]
ton = [r for r in d4 if r["token_ids"] and not r["translation_hypothesis"]]
b = collections.defaultdict(lambda: [0, 0])
for r in d4:
    k = min(int(r["audio_s"] // 2) * 2, 12); b[k][1] += 1; b[k][0] += int(r in ton)
print("transcript-only by duration (s):", {k: f"{v[0]}/{v[1]}" for k, v in sorted(b.items())})
nw = collections.Counter(min(len(r["hypothesis"].split()), 10) for r in ton); print("transcript words:", sorted(nw.items()))
same = sum(1 for r in ton if not by[r["id"]]["10"]["translation_hypothesis"]); print("also empty at δ10:", same, "/", len(ton))
for r in ton[:8]:
    print(f"  {r['audio_s']:.1f}s REF {r['reference'][:60]!r} | HYP {r['hypothesis'][:60]!r} | final TR {by[r['id']]['final']['translation_hypothesis'][:50]!r} | flush {r['flush_rounds']} post {r['post_audio_tokens']}")
