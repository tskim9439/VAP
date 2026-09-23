"""Approved Qwen-verbatim supervision -> deterministic mono stream records.

The selection/alignment artifacts are immutable.  This module only builds a
versioned training *view* and deliberately does not grant speaker/turn labels.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


SCHEMA = "vapasr-approved-mono-v1"
TARGET_POLICY = "qwen-verbatim-v1"


def alignment_density_problem(row: dict, max_same: int = 8, max_rate: float = 25.0):
    """Match the runtime loader's pathological-alignment guard before packing."""
    counts = defaultdict(int)
    for token in row["tokens"]: counts[float(token["end_s"])] += 1
    if not counts: return "empty_tokens"
    if max(counts.values()) > max_same: return "same_end_time_token_burst"
    if len(row["tokens"]) / max(.1, float(row["duration_s"])) > max_rate:
        return "token_rate_gt_25_per_s"
    return None


def stable_silence(left_key: str, right_key: str, low: float = .3,
                   high: float = .8) -> float:
    value = int(hashlib.sha256(f"{left_key}\0{right_key}".encode()).hexdigest()[:8], 16)
    return round(low + (high - low) * value / 0xffffffff, 3)


def affinity(row: dict) -> tuple:
    """Keep real dialogue chronology local; never infer a global speaker id."""
    source = row["source"]
    if row.get("timeline") == "original_dialogue":
        return source, row.get("parent_id") or row.get("group_id"), row.get("speaker")
    group = row.get("group_id")
    speaker = row.get("speaker")
    if not group and source == "kspon-full":
        group = str(Path(row["audio"]["path"]).parent)
    return source, row.get("subset"), group, speaker


def order_key(row: dict) -> tuple:
    return (float(row.get("start_s") or 0), row.get("utt_id") or "", row["key"])


def join_supervision(selected: dict, aligned: dict, approval_fingerprint: str) -> dict:
    """Join one selected row and one approved alignment, failing closed."""
    if selected["key"] != aligned["key"]:
        raise ValueError("key mismatch")
    target = (selected.get("recommended_training_target") or {}).get("text")
    if not target or target != aligned.get("target_text"):
        raise ValueError(f"target mismatch: {selected['key']}")
    if not aligned.get("alignment_ok") or aligned.get("training_eligible") is not False:
        raise ValueError(f"unapproved alignment shape: {selected['key']}")
    if aligned.get("waveform_sha256") != (selected.get("auto_selection") or {}).get("expected_waveform_sha256"):
        raise ValueError(f"waveform mismatch: {selected['key']}")
    return dict(
        schema=SCHEMA, key=selected["key"], source=selected["source"],
        corpus=selected.get("corpus") or selected["source"], split=selected.get("split", "train"),
        subset=selected.get("subset") or "train", lang=selected["lang"],
        parent_id=selected.get("parent_id"), utt_id=selected.get("utt_id"),
        speaker=selected.get("speaker"), speaker_scope=selected.get("speaker_scope"),
        group_id=selected.get("group_id"), timeline=selected.get("timeline"),
        start_s=selected.get("start_s"), end_s=selected.get("end_s"),
        audio=selected["audio"], duration_s=float(aligned["duration_s"]),
        target_text=target, target_origin="qwen3_asr_pseudo_label", target_normalization="none",
        display_style_verified=False, words=aligned["words"], tokens=aligned["tokens"],
        waveform_sha256=aligned["waveform_sha256"], target_sha256=aligned["target_sha256"],
        timing_quality=aligned["timing_quality"], approval_fingerprint=approval_fingerprint,
        asr_training_eligible=True, speaker_training_eligible=False, turn_training_eligible=False,
    )


def _finish(rows: list, approval_fingerprint: str) -> tuple:
    seed = "\0".join(r["key"] for r in rows)
    stream_id = "approved-" + hashlib.sha256(seed.encode()).hexdigest()[:24]
    segments, utts, cursor = [], [], 0.0
    for i, row in enumerate(rows):
        silence = 0.0 if i == 0 else stable_silence(rows[i - 1]["key"], row["key"])
        cursor = round(cursor + silence, 3)
        audio = row["audio"]
        segment = dict(utt_id=row["utt_id"] or row["key"], path=audio["path"],
                       offset_s=cursor, dur_s=row["duration_s"], silence_before_s=silence,
                       lexical_text=row["target_text"], text=row["target_text"],
                       raw_text=row["target_text"], display_source="qwen_raw",
                       speaker=row.get("speaker"), source_key=row["key"],
                       waveform_sha256=row["waveform_sha256"])
        if audio.get("offset_s") is not None:
            segment["src_offset_s"] = float(audio["offset_s"])
        shifted = [dict(id=int(t["id"]), end_time=round(cursor + float(t["end_s"]), 6))
                   for t in row["tokens"]]
        utts.append(dict(speaker=0, start=cursor, end=round(cursor + row["duration_s"], 6),
                         text=row["target_text"], tokens=shifted, source_key=row["key"]))
        segments.append(segment)
        cursor = round(cursor + row["duration_s"], 6)
    first = rows[0]
    manifest = dict(id=stream_id, corpus=first["source"], split=first["split"],
                    subset=first["subset"], mode="stream", lang=first["lang"],
                    duration_s=cursor, n_utts=len(rows), segments=segments,
                    meta=dict(schema=SCHEMA, target_policy=TARGET_POLICY,
                              approval_fingerprint=approval_fingerprint,
                              source_keys=[r["key"] for r in rows],
                              speaker_supervision=False, turn_supervision=False))
    return manifest, dict(id=stream_id, utts=utts)


def pack_streams(rows: list, approval_fingerprint: str, target_s: float = 22.0,
                 max_s: float = 25.0) -> list:
    """Pack clips inside safe affinity groups. Long clips remain standalone."""
    groups = defaultdict(list)
    for row in rows:
        if not row.get("asr_training_eligible"):
            raise ValueError("ineligible row passed to packer")
        groups[affinity(row)].append(row)
    result = []
    for key in sorted(groups, key=lambda x: repr(x)):
        current = []
        for row in sorted(groups[key], key=order_key):
            gap = 0 if not current else stable_silence(current[-1]["key"], row["key"])
            duration = sum(r["duration_s"] for r in current) + sum(
                stable_silence(a["key"], b["key"]) for a, b in zip(current, current[1:]))
            if current and duration >= target_s:
                result.append(_finish(current, approval_fingerprint)); current = []
                gap = 0
            elif current and duration + gap + row["duration_s"] > max_s:
                result.append(_finish(current, approval_fingerprint)); current = []
            current.append(row)
        if current:
            result.append(_finish(current, approval_fingerprint))
    return result


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
