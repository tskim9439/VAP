import glob, json, collections
R = "/soundai/users/tskim/VAPKT-data/results"
rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/ast-fleurs-q06-s2-ast/predictions-rank*.jsonl")) for l in open(f)]
by = collections.defaultdict(dict)
for r in rows: by[r["id"]][str(r["delta"])] = r
for ds in ("fleurs-en-de", "fleurs-en-ko", "fleurs-de-en", "fleurs-ko-en"):
    for d in ("4", "6", "8", "10"):
        rr = [v[d] for k, v in by.items() if k.startswith(ds + "/") and d in v]
        allempty = [r for r in rr if not r["token_ids"]]
        noTrans = [r for r in rr if r["token_ids"] and not r["translation_hypothesis"]]
        print(f"{ds} δ{d}: n={len(rr)} silent(no tokens)={len(allempty)} transcript-only={len(noTrans)}")
# duration buckets for en-de d4 silent
rr = [v["4"] for k, v in by.items() if k.startswith("fleurs-en-de/")]
b = collections.defaultdict(lambda: [0, 0])
for r in rr:
    k = min(int(r["audio_s"] // 5) * 5, 25); b[k][1] += 1; b[k][0] += int(not r["token_ids"])
print("en-de δ4 silent by duration bucket(s):", {k: f"{v[0]}/{v[1]}" for k, v in sorted(b.items())})
# same audio across en-de / en-ko?
ide = {k.split("/")[1] for k, v in by.items() if k.startswith("fleurs-en-de/") and not v["4"]["token_ids"]}
iko = {k.split("/")[1] for k, v in by.items() if k.startswith("fleurs-en-ko/") and not v["4"]["token_ids"]}
allko = {k.split("/")[1] for k in by if k.startswith("fleurs-en-ko/")}
print("silent en-de∩en-ko ids:", len(ide & iko), "en-de silent among en-ko ids:", len(ide & allko), "en-ko silent:", len(iko))
# ASR-only progress so far
for d in ("single-turn-vpaa-kspon-q06-s2-ast", "single-turn-yodas-de-q06-s2-ast", "single-turn-librispeech-clean-q06-s2-ast", "single-turn-librispeech-other-q06-s2-ast"):
    rs = [json.loads(l) for f in glob.glob(f"{R}/{d}/predictions-rank*.jsonl") for l in open(f)]
    c = collections.Counter((r["dataset"], str(r["delta"])) for r in rs); s = collections.Counter((r["dataset"], str(r["delta"])) for r in rs if not r["token_ids"])
    print(d, {f"{k[0]}/δ{k[1]}": f"{s[k]}/{c[k]}" for k in sorted(c)})
