#!/usr/bin/env python3
"""Row-level AST targets from the monotonic generation (experiments/ast_mono_generate.py run): for every Stage 1 ASR row and
direction, the translation of all its units in order, tokenized with the training tokenizer, each token carrying the
stable-prefix commit time of its word (row/stream time, s). This is the translation stream of the ast-v1 output format
(wiki decision-ast-monotonic-data 9): the dataset puts it next to the transcript with the same chunk rule.

A row is kept only when every unit is present, translated, judged and passes the gate (content same/minor, no omission or
addition, grammar not incorrect); otherwise it is dropped with a reason (the row stays an ASR-only item in training).

  python experiments/ast_build_targets.py --root DIR [--gen-dir gen-v3-bf16] [--tokenizer CKPT] [--procs 32]
Output: DIR/targets-<gen-dir>/<direction>/u-NNNNN.jsonl.gz rows {id, set, direction, src, tgt, n_units, text, tokens: [[tid, t]],
al, pre_end} + summary.json (per direction: rows, kept, reasons, tokens). Finished shards (same name) are skipped.
"""
import argparse
import bisect
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

LANG = {"en": "English", "de": "German", "ko": "Korean"}
CKPT = "/soundai/Model/VAPASR/semcommit-q06-s1m3-d10/final"
_tok = None


def _init(tok_path):
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(tok_path)


def token_times(tok, words, times):
    """words (translation, whitespace split) + per-word times -> [[token id, time]] for ' '.join(words): every token takes the time of
    the word holding its last character (Qwen byte-level tokens carry the leading space), times kept non-decreasing."""
    text, starts, pos = "", [], 0
    for i, w in enumerate(words):
        if i:
            text += " "; pos += 1
        starts.append(pos); text += w; pos += len(w)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    out, last = [], 0.0
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        j = max(0, bisect.bisect_right(starts, max(s, e - 1)) - 1)
        last = max(last, float(times[j]))
        out.append([int(tid), round(last, 3)])
    return text, out


def row_target(units):
    """units of one row (any order) -> (words, times, stats) or (None, reason)."""
    units = sorted(units, key=lambda u: u["k"])
    n = units[0]["n"]
    if len(units) != n or [u["k"] for u in units] != list(range(n)):
        return None, "missing_units"
    words, times, al, pre = [], [], [], []
    for u in units:
        if not u.get("translation") or not u.get("word_times"):
            return None, "no_translation"
        if u.get("judge") is None:
            return None, "no_judge"
        if not u["pass_gate"]:
            return None, "gate"
        w = u["translation"].split()
        if len(w) != len(u["word_times"]):
            return None, "word_times_mismatch"
        words += w; times += u["word_times"]
        if u.get("latency"):
            al.append(u["latency"]["al"]); pre.append(u["latency"]["pre_end"])
    return (words, times, dict(al=round(sum(al) / len(al), 3) if al else None,
                               pre_end=round(sum(pre) / len(pre), 3) if pre else None)), None


def process(job):
    src_path, out_path, direction = job
    s, t = direction.split("-")
    rows, reasons, n_tok = defaultdict(list), Counter(), 0
    for line in gzip.open(src_path, "rt", encoding="utf-8"):
        u = json.loads(line)
        rows[u["row"]].append(u)
    kept = []
    for rid, us in rows.items():
        r, why = row_target(us)
        if r is None:
            reasons[why] += 1
            continue
        words, times, st = r
        text, toks = token_times(_tok, words, times)
        n_tok += len(toks)
        kept.append(dict(id=rid, set=us[0]["set"], direction=direction, src=LANG[s], tgt=LANG[t], n_units=len(us),
                         text=text, tokens=toks, **st))
    tmp = out_path.with_name(out_path.name + f".partial-{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    return direction, len(rows), len(kept), reasons, n_tok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--gen-dir", default="gen-v3-bf16")
    ap.add_argument("--tokenizer", default=CKPT, help="training tokenizer (the starting checkpoint)")
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    root = Path(a.root)
    gen, out = root / a.gen_dir, root / f"targets-{a.gen_dir}"
    jobs = []
    for d in sorted(p.name for p in gen.iterdir() if p.is_dir()):
        (out / d).mkdir(parents=True, exist_ok=True)
        for f in sorted((gen / d).glob("u-*.jsonl.gz")):
            dst = out / d / f.name
            if not dst.exists():
                jobs.append((str(f), dst, d))
    print(f"{len(jobs)} shards to build -> {out}", flush=True)
    t0 = time.monotonic()
    if jobs:
        with Pool(a.procs, initializer=_init, initargs=(a.tokenizer,)) as pool:
            for n, _ in enumerate(pool.imap_unordered(process, jobs, chunksize=1)):
                if n % 100 == 0:
                    print(f"[{time.monotonic() - t0:.0f} s] {n + 1}/{len(jobs)}", flush=True)
    # summary over everything built so far (also earlier runs)
    summ = {}
    for d in sorted(p.name for p in out.iterdir() if p.is_dir()):
        s = summ.setdefault(d, Counter())
        for f in sorted((gen / d).glob("u-*.jsonl.gz")):
            if not (out / d / f.name).exists():
                continue
            s["shards"] += 1
            rows = defaultdict(list)
            for line in gzip.open(f, "rt", encoding="utf-8"):
                u = json.loads(line); rows[u["row"]].append(u)
            for us in rows.values():
                r, why = row_target(us)
                s["rows"] += 1
                s["kept" if r else f"drop_{why}"] += 1
            for line in gzip.open(out / d / f.name, "rt", encoding="utf-8"):
                s["tokens"] += len(json.loads(line)["tokens"])
    summ = {d: dict(s, kept_share=round(s["kept"] / max(1, s["rows"]), 4)) for d, s in summ.items()}
    (out / "summary.json").write_text(json.dumps(dict(gen_dir=a.gen_dir, tokenizer=a.tokenizer, built=time.strftime("%F %T"),
                                                      directions=summ), ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
