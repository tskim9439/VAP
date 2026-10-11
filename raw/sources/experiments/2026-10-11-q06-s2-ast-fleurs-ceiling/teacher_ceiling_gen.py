"""Teacher ceiling on FLEURS (q06-s2-ast evaluation, 2026-10-11): Qwen3.8-27B translates FLEURS source text (no audio) with
  mono-raw : v3 monotonic prompt (the training-target prompt), gold raw transcript (cased, punctuated)
  mono-norm: v3 monotonic prompt, gold normalized transcript (lowercase, no punctuation — like our ASR output)
  free-raw : plain professional-translation prompt, gold raw transcript
  mono-asr : v3 monotonic prompt, our model's final-mode transcript (cascade: our ASR errors + teacher MT)
usage: <vllm python> teacher_ceiling_gen.py <out_dir> [gpu_memory_utilization]"""
import glob, json, sys, time
sys.path.insert(0, "/soundai/users/tskim/VAPKT/experiments")
import ast_mono_pilot as P

def main():
    out = sys.argv[1]; util = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    M = "/soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-fleurs.jsonl"
    rows = [json.loads(l) for l in open(M)]
    hyp = {}
    for f in glob.glob("/soundai/users/tskim/VAPKT-data/results/ast-fleurs-q06-s2-ast/predictions-rank*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["delta"] == "final": hyp[r["id"]] = r["hypothesis"]
    assert all(r["id"] in hyp for r in rows), "missing final-mode transcripts"
    def free(src, tgt, text):
        return (f"You are a professional translator. Translate the {src} text into natural, fluent {tgt}. The text is a transcript "
                f"of speech and may have no punctuation or capital letters. Output only the {tgt} translation."), text
    conds = {"mono-raw": lambda r: P.p_translate_v3(r["lang"], r["tgt_lang"], r["raw_reference"]),
             "mono-norm": lambda r: P.p_translate_v3(r["lang"], r["tgt_lang"], r["reference"]),
             "free-raw": lambda r: free(r["lang"], r["tgt_lang"], r["raw_reference"]),
             "mono-asr": lambda r: P.p_translate_v3(r["lang"], r["tgt_lang"], hyp[r["id"]])}
    eng = P.VLLMEngine("/soundai/Model/Qwen3.8-27B", max_num_seqs=256, gpu_memory_utilization=util)
    keys, prompts = [], []
    for c, f in conds.items():
        for r in rows:
            keys.append((c, r)); prompts.append(f(r))
    t0 = time.monotonic()
    outs = eng(prompts, 320, None)
    with open(f"{out}/translations.jsonl", "w") as w:
        for (c, r), p, o in zip(keys, prompts, outs):
            w.write(json.dumps(dict(cond=c, id=r["id"], dataset=r["dataset"], tgt_lang=r["tgt_lang"], source=p[1],
                                    hypothesis=P.clean(o), raw_output=o, translation_reference=r["translation_reference"]), ensure_ascii=False) + "\n")
    print(json.dumps(dict(prompts=len(prompts), secs=round(time.monotonic() - t0, 1), n_out=eng.n_out, n_prompt=eng.n_prompt,
                          n_cached=eng.n_cached, n_truncated=eng.n_truncated, gpu_memory_utilization=util)), flush=True)


if __name__ == "__main__":
    main()
