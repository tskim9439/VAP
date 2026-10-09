import gzip, json, random, sys
R = "/soundai/users/tskim/VAPKT-data/data/ast-mono-v1"
rng = random.Random(7)
for d in ["en-de", "en-ko", "ko-en"]:
    gen = {json.loads(l)["uid"]: json.loads(l) for l in gzip.open(f"{R}/gen-v3-bf16/{d}/u-00000.jsonl.gz", "rt", encoding="utf-8")}
    fx = [json.loads(l) for l in gzip.open(f"{R}/fix-val-gen-v3-bf16/{d}/u-00000.jsonl.gz", "rt", encoding="utf-8")]
    flips = [f for f in fx if not f["pass_v2"] and f["pass_v3"]]
    reps = [f for f in fx if f["final"] == "repair"]
    strict = [f for f in fx if f["pass_v2"] and not f["pass_v3"]]
    print(f"\n######## {d}: flips {len(flips)} repairs {len(reps)} v2pass_v3fail {len(strict)}")
    for f in rng.sample(flips, min(8, len(flips))):
        g = gen[f["uid"]]; j2 = g["judge"] or {}; j3 = f["judge_v3"] or {}
        print(f"- [{g['set']}] SRC: {g['source'][:160]}\n  TGT: {g['translation'][:160]}\n  v2: {j2.get('content')}/om={j2.get('omission')}/add={j2.get('addition')}/{j2.get('grammar')} {str(j2.get('note'))[:140]}\n  v3: {j3.get('content')}/{j3.get('grammar')} {str(j3.get('note'))[:140]}")
    print("  --- repairs")
    for f in rng.sample(reps, min(4, len(reps))):
        g = gen[f["uid"]]; j3 = f["repair_judge"] or {}
        print(f"- SRC: {g['source'][:150]}\n  OLD: {g['translation'][:150]}\n  NEW: {f['repair_translation'][:150]}\n  v3: {j3.get('content')}/{j3.get('grammar')} {str(j3.get('note'))[:120]}")
    print("  --- v2 pass, v3 fail")
    for f in rng.sample(strict, min(3, len(strict))):
        g = gen[f["uid"]]; j3 = f["judge_v3"] or {}
        print(f"- SRC: {g['source'][:150]}\n  TGT: {g['translation'][:150]}\n  v3: {j3.get('content')}/om={j3.get('omission')}/add={j3.get('addition')}/{j3.get('grammar')} {str(j3.get('note'))[:140]}")
