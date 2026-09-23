"""Compact table of semcommit eval reports: per config — SEM (late2) P / P_exclB / R / F1 / latency p50 / PCR / B & hard-neg commit rate, TURN P/R/early, ASR WER|CER."""
import json, sys, os
def fm(x): return "    -" if x is None else f"{x:5.2f}"
for path in sorted(sys.argv[1:]):
    if path.endswith(".config.json") or not os.path.exists(path): continue
    d = json.load(open(path))
    if "configs" not in d: continue
    print(f"== {os.path.basename(path)[:-5]}")
    print(f"{'config':11s} {'nhyp':>4s} {'P':>5s} {'P-B':>5s} {'R':>5s} {'F1':>5s} {'lat50':>5s} {'PCR':>5s} {'Bcr':>5s} {'Ncr':>5s} | {'T.P':>5s} {'T.R':>5s} {'early':>5s} | asr")
    for k, c in d["configs"].items():
        o = c["overall"]; t = o["text"]; l2 = o["timing"]["late2"]; tu = o.get("turn", {}); tl = tu.get("late2", {}); a = o.get("asr", {})
        asr = " ".join(f"{m}={v['rate']*100:.2f}" for m, v in a.items() if isinstance(v, dict) and "rate" in v)
        print(f"{k:11s} {t['n_hyp']:4d} {fm(l2.get('precision'))} {fm(l2.get('precision_excl_B'))} {fm(l2.get('recall'))} {fm(l2.get('f1'))} {fm(l2.get('latency_s', {}).get('p50'))} "
              f"{fm(t.get('pcr'))} {fm(t.get('B_commit_rate'))} {fm(t.get('hardneg_commit_rate'))} | {fm(tl.get('precision'))} {fm(tl.get('recall'))} {tu.get('early_turn', 0):5d} | {asr}")
