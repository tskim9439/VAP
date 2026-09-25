#!/usr/bin/env python3
"""Build candidate-only semcommit words from completed speechlm ASR/alignment shards.

No row from this exporter is training approved. The source summary's hashes are
verified before a part is consumed, and train split / mono-channel / ASR-silver
selection are required. Short utterances are retained as *candidates* with an
explicit duration flag; they must not be silently promoted to 8–30 s streams.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.semcommit_words import PieceVocab, split_words


LEX = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
FINAL = set(".?!。？！")
COMMA = set(",;:，；：")
EXCLUDED_CORPUS = ("earnings", "turnbench")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def completed_part(part, verify=True):
    summary_path = part / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("complete") is not True:
        return None
    for name in ("aligned.jsonl", "records.jsonl"):
        p = part / name
        if not p.is_file() or name not in summary.get("outputs", {}):
            raise ValueError(f"incomplete declared output: {p}")
        if verify and sha256(p) != summary["outputs"][name]:
            raise ValueError(f"source checksum mismatch: {p}")
    return summary


EDGE_PUNCT = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.UNICODE | re.DOTALL)


def punct_tags(trailing):
    """Tags for punctuation that follows a word. An ellipsis ('...', '…') trails off — not a sentence end."""
    t = trailing.strip()
    final = any(c in t for c in FINAL) and not (t.endswith("...") or "…" in t)
    return (["punct_final"] if final else []) + (["punct_comma"] if any(c in t for c in COMMA) else [])


def lexical_words(canonical):
    """Tokenizer-split words keep punctuation inside the word ('해야지.'); semcommit words (gold v1, labels) are lexical with
    punctuation as tags. Strip edge punctuation into punct_final/punct_comma tags; a punctuation-only word is merged into the
    previous word (or the next one at stream start) so tokens[a:b] still cover every token. Returns (words, None) or
    (None, reason)."""
    out = []
    lead = None
    for w in canonical:
        m = EDGE_PUNCT.match(w["text"]); core = m.group(2)
        if not core:
            if out:
                prev = out[-1]; prev["b"] = w["b"]; prev["_trail"] += w["text"]
            elif lead is None:
                lead = w
            else:
                lead = dict(lead, b=w["b"])
            continue
        nw = dict(w, text=core, _trail=m.group(3))
        if lead is not None:
            nw["a"] = lead["a"]; lead = None
        out.append(nw)
    if not out:
        return None, "punct_only"
    for i, w in enumerate(out):
        w["i"] = i; w["tags"] = punct_tags(w.pop("_trail"))
    return out, None


def _token_to_word(tokens, spans, lexical_spans):
    """Map every BPE piece to one whitespace word, attaching punctuation.

    Qwen tokens may include *leading whitespace*; crossing the preceding
    lexical word (rather than just whitespace) is rejected.
    """
    groups = [[] for _ in spans]
    previous_end = -1
    for i, tok in enumerate(tokens):
        start, end = tok.get("char_start"), tok.get("char_end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or start < previous_end:
            return None
        previous_end = end
        owner = next((j for j, (a, b) in enumerate(spans) if a < end <= b), None)
        if owner is None:
            return None
        if owner and start < lexical_spans[owner - 1][1]:
            return None
        groups[owner].append(i)
    if not all(g and g == list(range(g[0], g[-1] + 1)) for g in groups):
        return None
    if [i for g in groups for i in g] != list(range(len(tokens))):
        return None
    return groups


def build_candidate(aligned, record, min_recipe_s=8.0, max_recipe_s=30.0, tokenizer=None):
    source = record.get("source_manifest") or {}
    if record.get("selection_state") != "KEEP_ASR_SILVER" or record.get("split") != "train":
        return None, "not_train_silver"
    if source.get("split") != "train" or source.get("num_channels") != 1:
        return None, "split_or_channels"
    if any(x in str(source.get("corpus", "")).lower() for x in EXCLUDED_CORPUS):
        return None, "eval_corpus"
    if aligned.get("alignment_ok") is not True or aligned.get("training_eligible") is not False:
        return None, "alignment_or_approval"
    if aligned.get("dedup_status") != "unverified" or aligned.get("key") != record.get("key"):
        return None, "dedup_or_join"
    duration = aligned.get("duration_s")
    if not isinstance(duration, (int, float)) or duration <= 0 or duration > 30:
        return None, "duration"
    target = aligned.get("target_text", "")
    matches = list(LEX.finditer(target)) if tokenizer is None else list(re.finditer(r"\S+", target))
    if not matches:
        return None, "no_words"
    spans = [(m.start(), m.end()) for m in matches]
    tokens = aligned.get("tokens") or []
    if any(float(tokens[i]["end_s"]) < float(tokens[i - 1]["end_s"]) - 1e-6
           for i in range(1, len(tokens))):
        return None, "token_time_reversal"
    canonical = None
    if tokenizer is not None:
        canonical = split_words([int(t["id"]) for t in tokens], [float(t["end_s"]) for t in tokens], tokenizer, text=target)
        if canonical is None or len(canonical) != len(matches):
            return None, "tokenizer_mismatch"
        groups = [list(range(w["a"], w["b"])) for w in canonical]
    else:
        # Test-only fallback for synthetic token IDs without a vocabulary.
        token_spans = [(a, spans[j + 1][0] if j + 1 < len(spans) else len(target))
                       for j, (a, _) in enumerate(spans)]
        groups = _token_to_word(tokens, token_spans, spans)
        if groups is None:
            return None, "token_mapping"
    aligned_words = aligned.get("words") or []
    if canonical is not None:
        lex, why = lexical_words(canonical)
        if lex is None:
            return None, why
        # word-level timing check against the aligner's own word spans (character offsets of each lexical word in target)
        words, pos = [], 0
        for w in lex:
            start = target.find(w["text"], pos)
            if start < 0:
                return None, "lexical_offset"
            end = start + len(w["text"]); pos = end
            overlaps = [x for x in aligned_words if x["char_start"] < end and x["char_end"] > start]
            if not overlaps:
                return None, "word_timing"
            end_time = w["end_time"]          # split_words: max token time of the word (a merged punctuation-only word keeps it)
            if end_time + 0.25 < max(float(x["end_s"]) for x in overlaps):
                return None, "token_word_time_disagree"
            if not 0 <= end_time <= duration + 0.25:
                return None, "word_timing"
            if words and end_time < words[-1]["end_time"] - 1e-3:
                return None, "time_reversal"
            words.append(dict(i=w["i"], text=w["text"], a=w["a"], b=w["b"], end_time=end_time, seg=0, tags=w["tags"]))
        matches = []
    else:
        words = []
    for j, m in enumerate(matches):
        overlaps = [w for w in aligned_words if w["char_start"] < m.end() and w["char_end"] > m.start()]
        if not overlaps:
            return None, "word_timing"
        end_time = float(tokens[groups[j][-1]]["end_s"])
        if end_time + 0.25 < max(float(w["end_s"]) for w in overlaps):
            return None, "token_word_time_disagree"
        if not 0 <= end_time <= duration + 0.25:
            return None, "word_timing"
        if words and end_time < words[-1]["end_time"] - 1e-3:
            return None, "time_reversal"
        tail = target[m.end():matches[j + 1].start() if j + 1 < len(matches) else len(target)]
        tags = punct_tags(tail)
        words.append(dict(i=j, text=canonical[j]["text"] if canonical is not None else m.group(),
                          a=groups[j][0], b=groups[j][-1] + 1,
                          end_time=end_time, seg=0, tags=tags))
    audio = aligned.get("audio") or {}
    if not isinstance(audio.get("path"), str) or "::" not in audio["path"]:
        return None, "audio_ref"
    stream_id = f"slm-{aligned.get('source')}-{aligned['key']}"
    row = dict(id=stream_id, set="speechlm-partial", lang=aligned["lang"], K=math.ceil(float(duration) / 0.08),
               duration_s=float(duration), text=" ".join(w["text"] for w in words),
               tokens=[[int(t["id"]), float(t["end_s"])] for t in tokens],
               segments=[dict(path=audio["path"], offset_s=0.0, silence_before_s=0.0,
                              dur_s=float(duration), utt_id=aligned["key"], raw_text=source.get("original_transcript"))],
               words=words, display_text=target,
               candidate_only=True, training_eligible=False, dedup_status="unverified",
               duration_within_recipe=min_recipe_s <= duration <= max_recipe_s,
               source_shard=aligned.get("source_shard"), source_key=aligned["key"],
               timing_quality=aligned.get("timing_quality"), source_corpus=source.get("corpus"))
    return row, None


def build_part(part, out_dir, verify=True, min_recipe_s=8.0, max_recipe_s=30.0, tokenizer=None):
    summary = completed_part(part, verify=verify)
    if summary is None:
        return None
    out = out_dir / f"{part.name}.words.jsonl"
    stats_path = out_dir / f"{part.name}.stats.json"
    if out.exists() or stats_path.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate snapshot: {out} / {stats_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = {r["key"]: r for r in iter_jsonl(part / "records.jsonl")}
    counts = Counter()
    with out.open("x", encoding="utf-8") as f:
        for aligned in iter_jsonl(part / "aligned.jsonl"):
            counts["aligned_rows"] += 1
            rec = records.get(aligned.get("key"))
            if rec is None:
                counts["missing_record"] += 1
                continue
            row, reason = build_candidate(aligned, rec, min_recipe_s, max_recipe_s, tokenizer)
            if reason:
                counts[f"drop_{reason}"] += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            counts["candidates"] += 1
            if row["duration_within_recipe"]:
                counts["recipe_duration"] += 1
            counts[f"lang_{row['lang']}"] += 1
    stats = dict(part=part.name, source_summary_sha256=sha256(part / "summary.json"),
                 source_fingerprint=summary["fingerprint"], words_sha256=sha256(out), counts=dict(counts),
                 candidate_only=True, training_eligible=False, duration_recipe_s=[min_recipe_s, max_recipe_s])
    with stats_path.open("x", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--part", action="append", help="process only named part(s), e.g. part-000001")
    p.add_argument("--no-verify-hash", action="store_true")
    p.add_argument("--min-recipe-s", type=float, default=8.0)
    p.add_argument("--max-recipe-s", type=float, default=30.0)
    p.add_argument("--tokenizer", type=Path, required=True, help="Qwen3-ASR tokenizer.json directory for strict token round-trip")
    a = p.parse_args()
    tokenizer = PieceVocab(str(a.tokenizer))
    parts = [a.results / name for name in a.part] if a.part else sorted(a.results.glob("part-*"))
    for part in parts:
        stats = build_part(part, a.out_dir, not a.no_verify_hash, a.min_recipe_s, a.max_recipe_s, tokenizer)
        if stats:
            print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
