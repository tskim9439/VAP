#!/usr/bin/env python3
"""Semantic-commit LLM teacher (plan §7–11, §16, §23). One model per process, resumable append-only JSONL.

  stageA    full-context annotation, greedy JSON generation (index-based)      → A.jsonl
  stageB    causal prefix judge, label scoring SAFE/WAIT/UNCERTAIN            → B.<judge>.jsonl
  stageC    future stability judge, label scoring STABLE/REVISION (+type)     → C.jsonl
  tiebreak  Korean candidates whose primary judges disagree (Stage B prompt)  → T.jsonl
  grade     A/B/N labels.jsonl + stats.json (no model)

Each model stage refuses to resume onto an output whose <out>.fingerprint.json differs (model, kind, prompt hash,
chat-template hash, template kwargs, generation/scoring params, words.jsonl digest; stageC also the Stage A file digest,
because the next candidate bounds its future window — extending Stage A after Stage C needs a new Stage C --out).
Rows already present (key = stage, id, after_word, judge, prompt_version) are skipped; inference_error rows are retried.
A run that wrote any inference_error row (or stopped after --max-failed-blocks consecutive failed blocks) prints
INCOMPLETE and exits 2; rerun the same command to retry. grade refuses inputs whose fingerprints name another
words.jsonl or mix prompt versions.
Example (rack4, one A100-40GB):
  python experiments/semcommit_teacher.py stageA --words W.jsonl --model /data4/tskim/OpenSource/qwen3-8b \
      --kind qwen3 --out A.jsonl --gpu 0 --batch-size 8
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data import semcommit_llm as sc
from vapasr.data.selection import file_digest

LLM = sc.LLM   # module attribute so tests can inject a fake model adapter


def _common(p):
    p.add_argument("--words", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cuda (pinned to --gpu via CUDA_VISIBLE_DEVICES) or cpu")
    p.add_argument("--batch-size", type=int, default=None,
                   help="default 1 for --kind gptoss (dense MoE + eager attention + CPU offload on 40 GB), else 8")
    p.add_argument("--limit", type=int, default=None, help="first N streams after --lang-filter")
    p.add_argument("--max-memory", default=None, help="e.g. 0:37GiB,cpu:120GiB → device_map=auto (gptoss default)")
    p.add_argument("--lang-filter", choices=["Korean", "English", "all"], default="all")


def _model(p, stage_a=True):
    p.add_argument("--model", required=True)
    p.add_argument("--kind", choices=sorted(sc.KINDS), required=True)
    p.add_argument("--judge-name", default=None, help="row judge name (default: --kind)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--flush-every", type=int, default=None, help="items per write+fsync block (default 4×batch)")
    p.add_argument("--max-failed-blocks", type=int, default=3,
                   help="stop after this many consecutive all-error blocks (0 = never); the run then exits 2")
    if not stage_a:
        p.add_argument("--stageA", type=Path, required=True)
        p.add_argument("--extra-candidates", default="", help="recipe v0.3: also judge rule-based candidates "
                       f"(comma list of {','.join(sc.EXTRA_SOURCES)}); '' = Stage A only (v0.2)")
        p.add_argument("--only-extra", action="store_true", help="only the extra candidates Stage A did not propose "
                       "(write them to a new file next to the v0.2 rows; grade takes several B/C files)")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stageA"); _common(p); _model(p)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--retries", type=int, default=1)
    p = sub.add_parser("stageB"); _common(p); _model(p, False)
    p = sub.add_parser("stageC"); _common(p); _model(p, False)
    p.add_argument("--future-words", type=int, default=12)
    p.add_argument("--future-s", type=float, default=3.0)
    p = sub.add_parser("tiebreak"); _common(p); _model(p, False)
    p.add_argument("--stageB", type=Path, nargs="+", required=True)
    p.add_argument("--judges-ko", default=",".join(sc.PRIMARY_JUDGES["Korean"]))
    p = sub.add_parser("grade"); _common(p)
    p.add_argument("--stageA", type=Path, required=True)
    p.add_argument("--stageB", type=Path, nargs="+", required=True)
    p.add_argument("--stageC", type=Path, nargs="+", required=True)
    p.add_argument("--tiebreak", type=Path, nargs="*", default=[])
    p.add_argument("--recipe", choices=["v0.2", "v0.3"], default="v0.2",
                   help="v0.3: candidates = Stage A ∪ --extra-candidates, calibrated grade (mean P(SAFE), P(REVISION), punctuated "
                        "sentence ends), N from human disfluency tags and --neg-rules; thresholds from --thresholds or semcommit_llm.V3_THRESHOLDS")
    p.add_argument("--extra-candidates", default=",".join(sc.EXTRA_SOURCES), help="v0.3 extra candidate sources")
    p.add_argument("--thresholds", type=Path, default=None, help="v0.3 thresholds JSON {lang: {punct|punct_resp|other: {p_safe, p_rev} | null}} (semcommit_gold.py tune)")
    p.add_argument("--neg-rules", default=",".join(sc.NEG_RULES), help="v0.3 rule negatives (semcommit_llm.rule_negatives) graded N; '' = v0.3–v0.3.2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--judges-ko", default=",".join(sc.PRIMARY_JUDGES["Korean"]))
    p.add_argument("--judges-en", default=",".join(sc.PRIMARY_JUDGES["English"]))
    p.add_argument("--disfl-negatives", action=argparse.BooleanOptionalAction, default=False,
                   help="ablation: also emit stageA=false N rows at disfluency positions Stage A did not propose (the "
                        "dataset gives them hardneg_weight, overriding the default NEXT weight at those word ends)")
    p.add_argument("--c-mask-types", default=",".join(sc.C_MASK_TYPES),
                   help="Stage C REVISION types graded B (masked) instead of N ('' = every REVISION is N)")
    p.add_argument("--turn-end", choices=["all", "none"], default="all", help="labels row turn_end flag (used only when training with --turn-end: <EOT>)")
    a = ap.parse_args(argv)
    if a.batch_size is None:
        a.batch_size = 1 if getattr(a, "kind", None) == "gptoss" else 8
    if a.batch_size < 1 or (a.limit is not None and a.limit < 1):
        ap.error("invalid --batch-size/--limit")
    if a.cmd != "grade":
        a.judge_name = a.judge_name or a.kind
        a.flush_every = a.flush_every or 4 * a.batch_size
    return a


def _rows_of(paths):
    return [r for p in paths for r in sc.read_jsonl(p)]


def _si(stage, it):
    """(stream, after_word) of a work item: A → stream, B/T → (s, i), C → (s, i, next)."""
    return (it, None) if stage == "A" else (it[0], it[1])


def run_model_stage(a, stage, items, fn, params):
    """Fingerprint + resume + blockwise inference with fsync; a failed block becomes explicit inference_error rows."""
    if a.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    llm = LLM(a.model, a.kind, device=a.device, max_memory=sc.parse_max_memory(a.max_memory), dtype=a.dtype)
    pv = sc.prompt_version()
    fp = dict(llm.fingerprint(), stage=stage, judge=a.judge_name, prompt_version=pv, prompt_sha256=sc.prompt_sha256(),
              words_sha256=file_digest(a.words), params=params)
    store = sc.JsonlStore(a.out, fp)
    pending, seen = [], set()
    for it in items:   # skip finished keys and repeated items (duplicate stream ids in words.jsonl)
        k = sc.row_key(stage, _si(stage, it)[0]["id"], _si(stage, it)[1], a.judge_name, pv)
        if k not in store.done and k not in seen:
            seen.add(k)
            pending.append(it)
    print(f"START {stage} judge={a.judge_name} kind={a.kind} items={len(items)} pending={len(pending)} "
          f"done_rows={store.n_rows} prompt={pv}", flush=True)
    if not pending:
        print("COMPLETE (nothing pending)", flush=True)
        return store
    llm.load()   # a load failure must raise, not become per-item error rows
    t0, n, tally, failed_run = time.monotonic(), 0, Counter(), 0
    for s in range(0, len(pending), a.flush_every):
        block = pending[s:s + a.flush_every]
        try:
            rows = fn(llm, block, pv)
            lines = sc.encode_rows(rows)   # serialize inside try: a NaN/inf must become error rows, not a crash mid-block
        except Exception as e:
            print(f"INFERENCE ERROR: {type(e).__name__}: {e}", flush=True)
            rows = [sc._row(stage, *_si(stage, it), a.judge_name, pv, status="inference_error",
                            error=f"{type(e).__name__}: {e}"[:500]) for it in block]
            lines = sc.encode_rows(rows)
        store.write(rows, lines)
        n += len(block)
        st = Counter(r["status"] for r in rows)
        tally.update(st)
        failed_run = failed_run + 1 if st["inference_error"] == len(rows) else 0
        print(json.dumps(dict(stage=stage, processed=n, pending=len(pending), elapsed_s=round(time.monotonic() - t0, 1),
                              status=dict(st))), flush=True)
        if a.max_failed_blocks and failed_run >= a.max_failed_blocks:
            print(f"ABORT: {failed_run} consecutive blocks failed", flush=True)
            break
    print(f"SUMMARY {stage} judge={a.judge_name} processed={n}/{len(pending)} status={dict(sorted(tally.items()))}",
          flush=True)
    if tally["nonfinite"]:
        print(f"WARNING: {tally['nonfinite']} row(s) with non-finite scores (graded as missing decisions)", flush=True)
    if tally["inference_error"] or n < len(pending):
        print(f"INCOMPLETE: {tally['inference_error']} inference_error row(s), {len(pending) - n} item(s) not run; "
              f"rerun the same command to retry", flush=True)
        sys.exit(2)
    print("COMPLETE", flush=True)
    return store


def main(argv=None):
    a = parse(argv)
    streams = sc.load_words(a.words, a.lang_filter, a.limit)
    if a.cmd == "stageA":
        params = dict(max_new_tokens=a.max_new_tokens, retries=a.retries, do_sample=False, num_beams=1)
        return run_model_stage(a, "A", streams, lambda llm, b, pv: sc.run_stage_a(
            llm, b, a.judge_name, a.max_new_tokens, a.batch_size, a.retries, pv), params)
    if a.cmd == "grade":
        judges = {"Korean": tuple(a.judges_ko.split(",")), "English": tuple(a.judges_en.split(","))}
        files = [("A", a.stageA), *[("B", p) for p in a.stageB], *[("C", p) for p in a.stageC], *[("T", p) for p in a.tiebreak]]
        rows = [(st, p, sc.read_jsonl(p)) for st, p in files]
        prov = sc.check_provenance(file_digest(a.words), rows)
        by = {st: [r for s_, _, rs in rows if s_ == st for r in rs] for st in "ABCT"}
        if a.recipe == "v0.3":
            th = json.load(open(a.thresholds)) if a.thresholds else None
            labels, stats = sc.build_labels_v3(streams, by["A"], by["B"], by["C"], judges=judges, turn_end=a.turn_end == "all",
                                               extra=tuple(x for x in a.extra_candidates.split(",") if x), thresholds=th,
                                               neg_rules=tuple(x for x in a.neg_rules.split(",") if x))
        else:
            labels, stats = sc.build_labels(streams, by["A"], by["B"], by["C"], by["T"], strict=a.strict, judges=judges,
                                            disfl_negatives=a.disfl_negatives, turn_end=a.turn_end == "all",
                                            c_mask_types=tuple(x for x in a.c_mask_types.split(",") if x))
        stats["code_prompt_version"], stats["prompt_version"] = stats["prompt_version"], prov["prompt_version"]
        if prov["prompt_version"] != stats["code_prompt_version"]:
            print(f"WARNING: inputs were made with prompt {prov['prompt_version']}, code is at "
                  f"{stats['code_prompt_version']}", flush=True)
        stats["provenance"] = prov["files"]
        stats["inputs"] = {str(p): file_digest(p) for p in [a.words, a.stageA, *a.stageB, *a.stageC, *a.tiebreak]}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix(a.out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in labels:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
        tmp.replace(a.out)
        sc.write_json(a.stats, stats)
        print(json.dumps(dict(labels=len(labels), by_lang=stats["by_lang"], streams=stats["streams"]),
                         ensure_ascii=False), flush=True)
        return labels, stats
    a_rows = sc.read_jsonl(a.stageA)
    extra = tuple(x for x in getattr(a, "extra_candidates", "").split(",") if x)
    if a.cmd == "stageB":
        return run_model_stage(a, "B", sc.stage_b_items(streams, a_rows, extra, a.only_extra), lambda llm, b, pv: sc.run_stage_b(
            llm, b, a.judge_name, a.batch_size, "B", pv), dict(labels=sc.B_LABELS, prefix=sc.B_PREFIX,
            **({"extra": list(extra), "only_extra": a.only_extra} if extra else {})))
    if a.cmd == "tiebreak":
        items = sc.tiebreak_items(streams, a_rows, _rows_of(a.stageB), a.judges_ko.split(","))
        return run_model_stage(a, "T", items, lambda llm, b, pv: sc.run_stage_b(
            llm, b, a.judge_name, a.batch_size, "T", pv),
            dict(labels=sc.B_LABELS, prefix=sc.B_PREFIX, judges_ko=a.judges_ko))
    if a.cmd == "stageC":
        return run_model_stage(a, "C", sc.stage_c_items(streams, a_rows, extra, a.only_extra), lambda llm, b, pv: sc.run_stage_c(
            llm, b, a.judge_name, a.batch_size, a.future_words, a.future_s, pv),
            dict(labels=sc.C_LABELS, types=sc.C_TYPES, prefix=sc.C_PREFIX, future_words=a.future_words,
                 future_s=a.future_s, stageA_sha256=file_digest(a.stageA), **({"extra": list(extra), "only_extra": a.only_extra} if extra else {})))


if __name__ == "__main__":
    main()
