"""Rule negatives (semcommit_llm.rule_negatives) and a mid-utterance connective-ending rule vs gold v1 (Korean parts)."""
import json, sys, collections
sys.path.insert(0, "/home/tskim/VAP")
from experiments.semcommit_gold import split_half
from vapasr.data.semcommit_llm import rule_negatives, _tok, KO_CONNECTIVE_ENDINGS, KO_FINAL_GO, KO_NEUDE
GD = "/data4/tskim/semcommit/gold/v1"
for part in ("ks-eval", "ks-long", "gs-test", "ls-test"):
    W = {r["id"]: r for r in map(json.loads, open(f"{GD}/words-{part}.jsonl"))}; rn = collections.Counter()
    for g in map(json.loads, open(f"{GD}/gold-{part}.jsonl")):
        lab = lambda i: "C" if i in g["commit"] else "A" if i in g["ambig"] else "N"
        for i, r in rule_negatives(W[g["id"]]).items(): rn[(r, split_half(g["id"]), lab(i))] += 1
    print(part, {f"{r}/{h}": f"C {rn[(r, h, 'C')]} AMB {rn[(r, h, 'A')]} NO {rn[(r, h, 'N')]}" for r in ("reply_prefix", "conn_final", "conn_mid") for h in ("dev", "test") if any(rn[(r, h, x)] for x in "CAN")})
