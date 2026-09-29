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
--lang / --part-prefix (2026-09-29): other languages and a separate root for SEM-neutral ASR words (semcommit_train --asr-words), e.g. E2's Korean
corpora: --lang Korean --part-prefix asr --min-s 0.5 --max-s 60 --out .../semcommit-work/asr-words-v1 (defaults keep the English labeling layout).

--regroup (2026-09-28): the streams the default mode leaves out, turned into 8–30 s main-pool streams → parts en-rg-<set>-NNNNNN:
  split   streams longer than --max-s made of several utterances (LibriSpeech: 581 h, 2.3 utterances each) are cut at utterance
          boundaries (the start of an utterance's leading silence) into consecutive pieces of ≤ --max-s;
  concat  utterances shorter than --min-s from the same recording (speaker/chapter, natural id order) are joined in order —
          target length drawn per stream in [--concat-min, --concat-max] s — with each utterance's own leading/trailing silence
          kept as the gap. The first word of every joined utterance is re-encoded with its leading space (Qwen BPE) and takes that
          word's token times. Both work on built word rows (build_stream), and every result is re-split from its tokens
          (split_words) and dropped unless it reproduces its words; single utterances over --max-s are left out.

  python experiments/semcommit_build_en_parts.py --manifests /soundai/users/tskim/VAPKT-data/data/manifests \\
      --sets voxpopuli-train,yodas-en129,librispeech-960,swbd-train,mnsc-1000 --tokenizer /soundai/Model/Qwen3-ASR-0.6B \\
      --out /soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035
"""
import argparse
from collections import Counter, defaultdict
import random
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


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def split_row(row, j0, j1, end_s, new_id):
    """Segments [j0, j1) of a built word row as a new stream starting at segment j0's leading silence and ending at end_s."""
    segs = row["segments"]; start = segs[j0]["offset_s"] - segs[j0]["silence_before_s"]
    ws = [w for w in row["words"] if j0 <= w["seg"] < j1]
    if not ws:
        return None
    a0, b1 = ws[0]["a"], ws[-1]["b"]
    D = round(end_s - start, 3)
    toks = [[int(t), round(float(x) - start, 3)] for t, x in row["tokens"][a0:b1]]
    if not toks or toks[0][1] < 0 or toks[-1][1] > D + 1e-6:
        return None
    words = [dict(w, i=k, a=w["a"] - a0, b=w["b"] - a0, end_time=round(w["end_time"] - start, 3), seg=w["seg"] - j0) for k, w in enumerate(ws)]
    nsegs = [dict(g, offset_s=round(g["offset_s"] - start, 3)) for g in segs[j0:j1]]
    out = dict(row, id=new_id, K=int(round(D / 0.08)), duration_s=D, tokens=toks, words=words, segments=nsegs, text=" ".join(w["text"] for w in words))
    out.pop("pnc_text", None)
    return out


def join_rows(rows, new_id, hf_tok):
    """Built word rows of consecutive utterances → one stream; later utterances' first word re-encoded with a leading space."""
    toks, words, segs, shift, prev_end, nseg = [], [], [], 0.0, 0.0, 0
    for k, r in enumerate(rows):
        rt = [[int(t), float(x)] for t, x in r["tokens"]]; rw = [dict(w) for w in r["words"]]
        if k:
            w0 = rw[0]; ids = hf_tok(" " + w0["text"], add_special_tokens=False)["input_ids"]
            old = [x for _, x in rt[w0["a"]:w0["b"]]]; L = len(old); n = len(ids)
            times = [old[min(L - 1, ((j + 1) * L + n - 1) // n - 1)] for j in range(n)]
            d = n - (w0["b"] - w0["a"])
            rt = [[i, t] for i, t in zip(ids, times)] + rt[w0["b"]:]
            rw = [dict(w0, b=w0["a"] + n)] + [dict(w, a=w["a"] + d, b=w["b"] + d) for w in rw[1:]]
        a_off = len(toks)
        toks += [[t, round(x + shift, 3)] for t, x in rt]
        for w in rw:
            words.append(dict(w, i=len(words), a=w["a"] + a_off, b=w["b"] + a_off, end_time=round(w["end_time"] + shift, 3), seg=w["seg"] + nseg))
        for g in r["segments"]:
            off = round(g["offset_s"] + shift, 3); segs.append(dict(g, offset_s=off, silence_before_s=round(off - prev_end, 3))); prev_end = off + g["dur_s"]
        nseg += len(r["segments"]); shift += float(r["duration_s"])
    D = round(shift, 3)
    out = dict(rows[0], id=new_id, K=int(round(D / 0.08)), duration_s=D, tokens=toks, words=words, segments=segs, text=" ".join(r["text"] for r in rows))
    pnc = [r.get("pnc_text") for r in rows]; out.pop("pnc_text", None)
    if all(pnc):
        out["pnc_text"] = " ".join(pnc)
    return out


def consistent(row, tok):
    from vapasr.data.semcommit_words import split_words
    ws = split_words([t for t, _ in row["tokens"]], [x for _, x in row["tokens"]], tok)
    return ws is not None and [(w["text"], w["a"], w["b"]) for w in ws] == [(w["text"], w["a"], w["b"]) for w in row["words"]]


def key(seed, sid):
    return hashlib.sha1(f"{seed}:{sid}".encode()).hexdigest()


def write_parts(a, prefix, name, rows):
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
        part = f"{prefix}-{k:06d}"
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
    (out_root / f"{a.part_prefix}-parts").mkdir(parents=True, exist_ok=True)
    (out_root / f"{a.part_prefix}-parts" / f"{prefix[len(a.part_prefix) + 1:]}.tsv").write_text("".join(f"{p}\t{s}\t{n}\t{h:.4f}\n" for p, s, n, h in table))
    return table, parts


def build_set(a, tok, name):
    t0 = time.time()
    streams = {}
    with open(Path(a.manifests) / name / "streams.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (a.lang == "any" or r.get("lang") == a.lang) and a.min_s <= float(r["duration_s"]) <= a.max_s:
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
    table, parts = write_parts(a, f"{a.part_prefix}-{name}", name, rows)
    stats = dict(set=name, counts=dict(counts), dropped=dict(drops), kept=len(rows), hours=round(sum(r["duration_s"] for r in rows) / 3600, 2),
                 parts=len(parts), part_hours=a.part_hours, min_s=a.min_s, max_s=a.max_s, seed=a.seed, seconds=round(time.time() - t0, 1))
    (Path(a.out) / f"{a.part_prefix}-parts" / f"{name}.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def build_regroup(a, tok, hf_tok, name):
    t0 = time.time()
    streams = {}
    with open(Path(a.manifests) / name / "streams.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = float(r["duration_s"])
            if (a.lang == "any" or r.get("lang") == a.lang) and (d < a.min_s or (d > a.max_s and len(r["segments"]) > 1)):
                streams[r["id"]] = r
    counts, drops, built = Counter(streams_selected=len(streams)), Counter(), {}
    for it in iter_items(Path(a.align_root) / name / "_items-online-v2.json.gz"):
        sr = streams.get(it.get("id"))
        if sr is None:
            continue
        out = build_stream(it, sr, tok, roots=None, pnc_lookup=pnc_of(sr), info={}, set_name=f"{name}-rg", pnc_min_ratio=a.pnc_min_ratio)
        if isinstance(out, tuple):
            drops["build_" + out[1]] += 1; continue
        out["_group"] = (sr.get("speaker") or "", sr.get("chapter") or ""); built[out["id"]] = out
    rows = []
    for r in [r for r in built.values() if r["duration_s"] > a.max_s]:        # split long multi-utterance streams
        segs = r["segments"]; bounds = [g["offset_s"] - g["silence_before_s"] for g in segs] + [r["duration_s"]]
        j0 = 0; k = 0
        while j0 < len(segs):
            j1 = j0 + 1
            while j1 < len(segs) and bounds[j1 + 1] - bounds[j0] <= a.max_s:
                j1 += 1
            if bounds[j1] - bounds[j0] > a.max_s:
                drops["split_single_utt_over_max"] += 1; j0 = j1; continue
            piece = split_row(r, j0, j1, bounds[j1], f"{r['id']}-p{k}")
            if piece is None:
                drops["split_empty"] += 1
            elif piece["duration_s"] < a.min_s:
                drops["split_tail_short"] += 1
            else:
                rows.append(piece); counts["split_pieces"] += 1
            j0, k = j1, k + 1
    groups = defaultdict(list)
    for r in built.values():
        if r["duration_s"] < a.min_s:
            groups[r["_group"]].append(r)
    for g, rs in groups.items():                                               # concat short utterances of one recording
        rs.sort(key=lambda r: natural_key(r["id"]))
        rnd = random.Random(f"{a.seed}:{name}:{g}"); target = rnd.uniform(a.concat_min, a.concat_max); cur = []; cur_s = 0.0
        def flush():
            nonlocal cur, cur_s, target
            if cur and cur_s >= a.min_s:
                rows.append(join_rows(cur, f"cat-{cur[0]['id']}-n{len(cur)}", hf_tok)); counts["concat_streams"] += 1; counts["concat_utts"] += len(cur)
            elif cur:
                drops["concat_tail_short"] += 1
            cur, cur_s, target = [], 0.0, rnd.uniform(a.concat_min, a.concat_max)
        for r in rs:
            if cur and cur_s + r["duration_s"] > a.max_s:
                flush()
            cur.append(r); cur_s += r["duration_s"]
            if cur_s >= target:
                flush()
        flush()
    good = []
    for r in rows:
        r.pop("_group", None)
        if not a.min_s <= r["duration_s"] <= a.max_s:
            drops["out_of_range"] += 1
        elif not consistent(r, tok):
            drops["resplit_mismatch"] += 1
        else:
            good.append(r)
    table, parts = write_parts(a, f"{a.part_prefix}-rg-{name}", name, good)
    stats = dict(set=name, mode="regroup", counts=dict(counts), dropped=dict(drops), kept=len(good), hours=round(sum(r["duration_s"] for r in good) / 3600, 2),
                 parts=len(parts), part_hours=a.part_hours, min_s=a.min_s, max_s=a.max_s, concat=[a.concat_min, a.concat_max], seed=a.seed,
                 seconds=round(time.time() - t0, 1))
    (Path(a.out) / f"{a.part_prefix}-parts" / f"rg-{name}.stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifests", required=True, help="directory with <set>/streams.jsonl")
    p.add_argument("--align-root", default=None, help="forced-alignment cache root (default <manifests>/align-asr-tn-v1)")
    p.add_argument("--sets", required=True, help="comma list of English manifest sets")
    p.add_argument("--tokenizer", required=True, help="Qwen3-ASR directory (tokenizer.json / vocab.json)")
    p.add_argument("--out", required=True, help="labels root (parts/ and <prefix>-parts/ are written below it)")
    p.add_argument("--lang", default="English", help="manifest row language filter: English (default) | Korean | any")
    p.add_argument("--part-prefix", default="en", help="part name prefix <prefix>-<set>-NNNNNN and table dir <prefix>-parts (default en)")
    p.add_argument("--part-hours", type=float, default=1.5)
    p.add_argument("--min-s", type=float, default=8.0); p.add_argument("--max-s", type=float, default=30.0)
    p.add_argument("--pnc-min-ratio", type=float, default=0.9); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regroup", action="store_true", help="split long multi-utterance streams + concat short utterances (parts en-rg-*)")
    p.add_argument("--concat-min", type=float, default=12.0); p.add_argument("--concat-max", type=float, default=25.0)
    a = p.parse_args(argv)
    a.align_root = a.align_root or os.path.join(a.manifests, "align-asr-tn-v1")
    tok = PieceVocab(a.tokenizer)
    if a.regroup:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(a.tokenizer)
        return [build_regroup(a, tok, hf_tok, name) for name in a.sets.split(",") if name]
    return [build_set(a, tok, name) for name in a.sets.split(",") if name]


if __name__ == "__main__":
    main()
