#!/usr/bin/env python3
"""English semcommit parts for the part-unit labeling worker (slurm/semcommit-part-label-worker.sh) — words.jsonl built ahead,
so the worker skips its speechlm candidate/approval step (words.jsonl.ok present) and runs the teacher stages directly.

Source = Phase 1 training manifests already on mxc: <manifests>/<set>/streams.jsonl + the forced-alignment cache
<align-root>/<set>/_items-online-v2.json.gz (MonoStreamDataset items = aligned-manifest rows: tokens [[qwen_id, end_s]], text,
segments), joined by id and turned into words with vapasr.data.semcommit_words.build_stream (same as semcommit_build_words.py).
Only main-pool streams (--min-s ≤ duration ≤ --max-s, default 8–30 s) — the short pool is reserved for approved speechlm rows.
Punctuation: sets whose segment raw_text carries punctuation (voxpopuli, yodas, mnsc) get punct_final/punct_comma tags and pnc_text
from it (apply_en_pnc, word match ≥ --pnc-min-ratio); LibriSpeech (no LibriSpeech-PC on mxc) and Switchboard have none.
Streams are ordered by sha1(seed:id) inside a set (a partial run samples the whole set) and cut into parts of ≈ --part-hours.
Output: <out>/parts/en-<set>-NNNNNN/{words.jsonl, words.jsonl.ok} and <out>/en-parts/<set>.tsv (part, source, rows, hours),
<out>/en-parts/<set>.stats.json. Never overwrites: an existing part directory with words.jsonl is refused.

  python experiments/semcommit_build_en_parts.py --manifests /soundai/users/tskim/VAPKT-data/data/manifests \\
      --sets voxpopuli-train,yodas-en129,librispeech-960,swbd-train,mnsc-1000 --tokenizer /soundai/Model/Qwen3-ASR-0.6B \\
      --out /soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.semcommit_words import PieceVocab, build_stream

PUNCT = re.compile(r"[.,?!;:]")
MARKUP = re.compile(r"<[^>]*>\s*:?")   # mnsc '<Speaker1>: ' and similar speaker/event markup


def iter_items(cache_path):
    """Items of a MonoStreamDataset alignment cache ({..., "items": [ {...}, ... ]}) one at a time without loading the file."""
    dec = json.JSONDecoder()
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        buf = ""
        while '"items"' not in buf:
            chunk = f.read(1 << 20)
            if not chunk:
                raise ValueError(f"{cache_path}: no items array")
            buf += chunk
        pos = buf.index("[", buf.index('"items"')) + 1
        while True:
            while True:                                  # skip separators; refill when the buffer runs dry
                while pos < len(buf) and buf[pos] in " \n\r\t,":
                    pos += 1
                if pos < len(buf):
                    break
                chunk = f.read(1 << 20)
                if not chunk:
                    return
                buf, pos = buf[pos:] + chunk, 0
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except json.JSONDecodeError:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        raise
                    buf, pos = buf[pos:] + chunk, 0
            yield obj
            pos = end
            if pos > (1 << 22):
                buf, pos = buf[pos:], 0


def pnc_of(streams_row):
    """{utt_id: punctuated text} from segment raw_text, only for segments that actually carry punctuation."""
    out = {}
    for s in streams_row.get("segments") or []:
        raw = MARKUP.sub(" ", s.get("raw_text") or "").strip()
        if raw and PUNCT.search(raw) and not raw.isupper():
            out[s.get("utt_id")] = " ".join(raw.split())
    return out or None


def key(seed, sid):
    return hashlib.sha1(f"{seed}:{sid}".encode()).hexdigest()


def build_set(a, tok, name):
    t0 = time.time()
    streams = {}
    with open(Path(a.manifests) / name / "streams.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("lang") == "English" and a.min_s <= float(r["duration_s"]) <= a.max_s:
                streams[r["id"]] = r
    counts, drops, rows = Counter(streams_in_range=len(streams)), Counter(), []
    for it in iter_items(Path(a.align_root) / name / "_items-online-v2.json.gz"):
        counts["aligned_items"] += 1
        sr = streams.get(it.get("id"))
        if sr is None:
            continue
        counts["candidates"] += 1
        info = {}
        out = build_stream(it, sr, tok, roots=None, pnc_lookup=pnc_of(sr), info=info, set_name=name, pnc_min_ratio=a.pnc_min_ratio)
        if isinstance(out, tuple):
            drops[out[1]] += 1
            continue
        if not a.min_s <= out["duration_s"] <= a.max_s:
            drops["duration_after_build"] += 1
            continue
        counts["tagged"] += any(w.get("tags") for w in out["words"])
        counts["pnc_text"] += "pnc_text" in out
        rows.append(out)
    rows.sort(key=lambda r: key(a.seed, r["id"]))
    parts, cur, cur_s = [], [], 0.0
    for r in rows:
        cur.append(r); cur_s += r["duration_s"]
        if cur_s >= a.part_hours * 3600:
            parts.append(cur); cur, cur_s = [], 0.0
    if cur:
        parts.append(cur)
    out_root = Path(a.out)
    table = []
    for k, prows in enumerate(parts):
        part = f"en-{name}-{k:06d}"
        d = out_root / "parts" / part
        if (d / "words.jsonl").exists():
            raise SystemExit(f"refusing to overwrite {d}/words.jsonl")
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "words.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for r in prows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(d / "words.jsonl")
        h = sum(r["duration_s"] for r in prows) / 3600
        sha = hashlib.sha256((d / "words.jsonl").read_bytes()).hexdigest()
        (d / "words.jsonl.ok").write_text(json.dumps(dict(schema="semcommit-en-part-words-v1", part=part, source=name, rows=len(prows),
                                                        hours=round(h, 4), words_sha256=sha, seed=a.seed)) + "\n")
        table.append((part, name, len(prows), h))
    (out_root / "en-parts").mkdir(parents=True, exist_ok=True)
    (out_root / "en-parts" / f"{name}.tsv").write_text("".join(f"{p}\t{s}\t{n}\t{h:.4f}\n" for p, s, n, h in table))
    stats = dict(set=name, counts=dict(counts), dropped=dict(drops), kept=len(rows), hours=round(sum(r["duration_s"] for r in rows) / 3600, 2),
                 parts=len(parts), part_hours=a.part_hours, min_s=a.min_s, max_s=a.max_s, seed=a.seed, seconds=round(time.time() - t0, 1))
    (out_root / "en-parts" / f"{name}.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", required=True, help="directory with <set>/streams.jsonl")
    p.add_argument("--align-root", default=None, help="forced-alignment cache root (default <manifests>/align-asr-tn-v1)")
    p.add_argument("--sets", required=True, help="comma list of English manifest sets")
    p.add_argument("--tokenizer", required=True, help="Qwen3-ASR directory (tokenizer.json / vocab.json)")
    p.add_argument("--out", required=True, help="labels root (parts/ and en-parts/ are written below it)")
    p.add_argument("--part-hours", type=float, default=1.5)
    p.add_argument("--min-s", type=float, default=8.0); p.add_argument("--max-s", type=float, default=30.0)
    p.add_argument("--pnc-min-ratio", type=float, default=0.9); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    a.align_root = a.align_root or os.path.join(a.manifests, "align-asr-tn-v1")
    tok = PieceVocab(a.tokenizer)
    return [build_set(a, tok, name) for name in a.sets.split(",") if name]


if __name__ == "__main__":
    main()
