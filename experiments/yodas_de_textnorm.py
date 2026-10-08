#!/usr/bin/env python3
"""YODAS-Granary German words rows → asr-tn-de-v1 training targets (vapasr/data/textnorm_de.py), CPU.

The aligned rows (experiments/yodas_de_align.py) carry the raw Qwen3-ASR text with punctuation and case. English and Korean
training targets are lowercase without punctuation (asr-tn-v1), so the German rows are converted the same way before training:
lowercase, punctuation tokens dropped, hyphenated words split, rows with digits / abbreviations / symbols quarantined. The raw
Qwen text stays in display_text (source of a later written-form target) and the Granary text in segments[].raw_text.

  python experiments/yodas_de_textnorm.py --root <yodas-de-v1> --split train [--tag tn2] [--workers 32]

Input: <root>/lists/de-words-<split>-v1.list (non-empty results/<shard>/words-<split>.jsonl). Output: <root>/<tag>/<shard>.words-<split>.jsonl,
<root>/<tag>/summary-<split>.json, <root>/lists/de-words-<split>-<tag>.list. A new normalisation version gets a new --tag
(tn1 = asr-tn-de-v1.0.0, tn2 = v1.1.0). Each converted row is checked by re-splitting its tokens
with the training tokenizer (semcommit_words.split_words — the SemCommitDataset tokenizer gate) and for monotonic token times;
a row that fails is dropped as tn_resplit. Existing outputs are kept (a rerun only fills in missing shards); nothing is deleted.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TOK = None


def _tok(path):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(path)          # the training scripts load the tokenizer the same way
    return _TOK


def convert(job):
    src, dst, tok_path = job
    from vapasr.data.semcommit_words import split_words
    from vapasr.data.textnorm_de import normalize_de_row
    tok = _tok(tok_path)
    if os.path.exists(dst):
        return json.loads(Path(dst + ".stats.json").read_text())
    rows, ct, hours = [], Counter(), Counter()
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        out = normalize_de_row(r, tok)
        if isinstance(out, tuple):
            why = out[1]
        else:
            ids, times = [int(t[0]) for t in out["tokens"]], [float(t[1]) for t in out["tokens"]]
            ws = split_words(ids, times, tok, text=out["text"])
            ok = ws is not None and [(w["text"], w["a"], w["b"]) for w in ws] == [(w["text"], w["a"], w["b"]) for w in out["words"]]
            ok = ok and all(b >= a - 1e-6 for a, b in zip(times, times[1:]))
            why = None if ok else "tn_resplit"
        state = why or ("ok_changed" if out["text"] != r["text"] or out["tokens"] != r["tokens"] else "ok_same")
        ct[state] += 1
        hours[state] += float(r["duration_s"]) / 3600
        if why is None:
            rows.append(out)
    with open(dst + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = dict(src=src, rows=dict(ct), hours={k: round(v, 4) for k, v in hours.items()}, kept=len(rows))
    Path(dst + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False))
    os.replace(dst + ".tmp", dst)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", choices=("train", "dev"), required=True)
    ap.add_argument("--tokenizer", default="/soundai/Model/Qwen3-ASR-1.7B")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", default="tn2", help="output folder and list suffix; one per normalisation version")
    a = ap.parse_args()
    root = Path(a.root)
    srcs = [l.strip() for l in open(root / "lists" / f"de-words-{a.split}-v1.list") if l.strip()]
    out = root / a.tag
    out.mkdir(exist_ok=True)
    jobs = [(s, str(out / f"{Path(s).parent.name}.words-{a.split}.jsonl"), a.tokenizer) for s in srcs]
    t0 = time.monotonic()
    rows, hours, done = Counter(), Counter(), []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, st in enumerate(ex.map(convert, jobs, chunksize=4)):
            rows.update(st["rows"])
            hours.update(st["hours"])
            done.append(st)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(jobs)} files {time.monotonic() - t0:.0f} s {dict(rows)}", flush=True)
    from vapasr.data.textnorm_de import TEXTNORM_DE_VERSION
    kept = [d for (s, d, _), st in zip(jobs, done) if st["kept"]]
    lst = root / "lists" / f"de-words-{a.split}-{a.tag}.list"
    lst.with_suffix(".tmp").write_text("".join(p + "\n" for p in kept))
    os.replace(lst.with_suffix(".tmp"), lst)
    ok = ("ok_same", "ok_changed")
    summary = dict(split=a.split, tag=a.tag, textnorm=TEXTNORM_DE_VERSION, tokenizer=a.tokenizer, files=len(jobs), files_kept=len(kept),
                   rows=dict(rows.most_common()), hours={k: round(v, 2) for k, v in hours.most_common()},
                   kept_rows=sum(rows[k] for k in ok), kept_hours=round(sum(hours[k] for k in ok), 2),
                   elapsed_s=round(time.monotonic() - t0, 1))
    (out / f"summary-{a.split}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
