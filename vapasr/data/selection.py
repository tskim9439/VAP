"""Data-selection v0.1: immutable labels, fail-closed QC, no training promotion.

This module intentionally needs only the standard library. A teacher agreement
is a REVIEW CANDIDATE, never a gold label or a turn-boundary annotation.
"""
import bisect
import hashlib
import json
import math
from pathlib import Path

VERSION = "data-selection-v0.1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def agreement(ref, qwen, whisper):
    """Already-normalized units; KO characters, EN words. No rate clipping."""
    return {"qwen_ref": edit_distance(ref, qwen) / max(1, len(ref)),
            "whisper_ref": edit_distance(ref, whisper) / max(1, len(ref)),
            "qwen_whisper": edit_distance(qwen, whisper) / max(1, len(qwen), len(whisper)),
            "ref_units": len(ref), "qwen_units": len(qwen), "whisper_units": len(whisper)}


def adjudicate(metrics=None, *, hard=(), review=(), teachers_ok=True):
    """Provisional rules. Disagreement alone must NEVER mean corrupt audio."""
    if hard:
        return "QUARANTINE", sorted(set(hard))
    if not teachers_ok or metrics is None:
        return "PENDING", ["missing_teacher_evidence"]
    if review:
        return "REVIEW", sorted(set(review))
    q, w, pair = (metrics[k] for k in ("qwen_ref", "whisper_ref", "qwen_whisper"))
    if metrics["ref_units"] == 0:
        return "REVIEW", ["empty_reference"]
    if not metrics.get("qwen_units", 1) or not metrics.get("whisper_units", 1):
        return "REVIEW", ["empty_teacher_hypothesis"]
    threshold = 0.0 if metrics["ref_units"] <= 4 else 0.05
    if max(q, w, pair) <= threshold:
        return "AGREEMENT_CANDIDATE", ["dual_teacher_agrees_not_gold"]
    if pair <= 0.05 and min(q, w) >= 0.15:
        return "CORRECTION_REVIEW", ["teachers_agree_against_reference_no_relabel"]
    return "REVIEW", ["teacher_disagreement"]


def exact_piece(channel, utterance, missing):
    """Never substitute a neighbouring crop when the actual crop is missing."""
    uid = utterance["utt_id"]
    try:
        index = int(uid.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None, "unresolved_crop_index"
    if index in missing:
        return None, "missing_crop_declared"
    stem = uid.rsplit("_", 1)[0]
    expected = f"{stem}_{index}.wav"
    hits = [p for p, offset in channel.get("pieces", [])
            if Path(p).name == expected and abs(float(offset) - utterance["start"]) < 0.001]
    if len(hits) != 1:
        return None, "missing_or_ambiguous_exact_crop"
    return {"path": hits[0], "offset_s": None, "duration_s": None}, None


def overlap_index(utterances):
    by = {}
    for u in utterances:
        by.setdefault(u["speaker"], []).append((float(u["start"]), float(u["end"])))
    result = {}
    for spk, intervals in by.items():
        starts, ends, maximum = [], [], -math.inf
        for start, end in sorted(intervals):
            starts.append(start)
            maximum = max(maximum, end)
            ends.append(maximum)
        result[spk] = starts, ends
    return result


def has_overlap(u, index):
    for spk, (starts, ends) in index.items():
        if spk == u["speaker"]:
            continue
        i = bisect.bisect_left(starts, float(u["end"])) - 1
        if i >= 0 and ends[i] > float(u["start"]):
            return True
    return False


def base_row(source, parent, uid, lang, text, raw, duration):
    return dict(schema=VERSION, key=digest([source, parent, uid]), source=source,
                parent_id=parent, utt_id=uid, lang=lang, text=text, raw_text=raw,
                duration_s=duration, reasons=[], source_flags=[], overlap=None,
                training_eligible=False, text_quality="pending",
                timing_quality="unverified", speaker_quality="unverified",
                turn_quality="unverified")


def stream_rows(d, source):
    for seg in d["segments"]:
        r = base_row(source, d["id"], seg["utt_id"], d["lang"], seg["text"],
                     seg.get("raw_text", seg["text"]), float(seg["dur_s"]))
        r.update(corpus=d["corpus"], split=d["split"], subset=d.get("subset"),
                 speaker=d.get("speaker"), speaker_scope="source_or_unknown",
                 group_id=d.get("chapter"), timeline="synthetic_stream",
                 start_s=seg["offset_s"], end_s=seg["offset_s"] + seg["dur_s"],
                 audio=dict(path=seg["path"], offset_s=seg.get("src_offset_s"),
                            duration_s=seg["dur_s"] if seg.get("src_offset_s") is not None else None),
                 source_flags=seg.get("flags") or [])
        yield r


def dialogue_rows(d, source):
    oi = overlap_index(d["utterances"])
    missing = set((d.get("meta") or {}).get("missing", []))
    for u in d["utterances"]:
        r = base_row(source, d["conv_id"], u["utt_id"], d["lang"], u["text"],
                     u.get("raw", u["text"]), float(u["end"]) - float(u["start"]))
        channel = d["channels"].get(u["speaker"])
        audio, problem = None, None
        if channel is None:
            problem = "missing_speaker_channel"
        elif channel.get("pieces") is not None:
            audio, problem = exact_piece(channel, u, missing)
        elif channel.get("path"):
            audio = dict(path=channel["path"], offset_s=float(u["start"]),
                         duration_s=r["duration_s"])
        else:
            problem = "empty_channel_reference"
        if problem:
            r["reasons"].append(problem)
        if float(u["start"]) < 0 or float(u["end"]) > float(d["duration_s"]) + 0.1:
            r["reasons"].append("outside_parent_timeline")
        r.update(corpus=d["corpus"], split=d["split"], audio=audio,
                 speaker=u["speaker"], speaker_scope="conversation_local",
                 group_id=d["conv_id"], timeline="original_dialogue",
                 start_s=u["start"], end_s=u["end"],
                 source_flags=u.get("flags") or [], overlap=has_overlap(u, oi))
        yield r


def stratum(r):
    if r["duration_s"] < 2:
        return "short"
    if r["overlap"]:
        return "overlap"
    if r["duration_s"] > 15:
        return "long"
    return "regular"


def audio_identity(r):
    """Locator identity, NOT a waveform/content hash."""
    return digest(r["audio"])
