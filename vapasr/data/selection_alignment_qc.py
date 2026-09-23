"""Fail-closed QC policy for qwen-verbatim-align-v1 records."""
import hashlib
import math

from .selection_alignment import project_items, surface, text_sha


POLICY = dict(version="qwen-verbatim-align-qc-v1", auto_clamp_max_s=0.080,
              review_max_s=0.320, acoustic_spot_check_required=True)


def _items(record):
    return [dict(text=x["text"], start_time=float(x["start_time"]),
                 end_time=float(x["end_time"])) for x in record["aligner_items"]]


def validate_original(record):
    """Recheck stored success without trusting its alignment_ok flag."""
    text, duration = record["target_text"], float(record["duration_s"])
    if record.get("schema") != "qwen-verbatim-align-v1" or not record.get("alignment_ok"):
        raise ValueError("wrong_schema_or_status")
    if record.get("target_sha256") != text_sha(text) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("target_or_duration_invalid")
    items, words, tokens = _items(record), record.get("words", []), record.get("tokens", [])
    if not items or len(items) != len(words) or not tokens:
        raise ValueError("missing_alignment_structure")
    raw = surface(text)[0]
    if "".join(surface(x["text"])[0] for x in items) != raw:
        raise ValueError("aligner_surface_mismatch")
    previous_start = previous_end = -1.0
    for item, word in zip(items, words):
        start, end = item["start_time"], item["end_time"]
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start <= end <= duration and start >= previous_start and end >= previous_end):
            raise ValueError("invalid_word_timestamp")
        if (word["text"] != item["text"] or float(word["start_s"]) != start
                or float(word["end_s"]) != end
                or not 0 <= word["char_start"] < word["char_end"] <= len(text)):
            raise ValueError("word_projection_mismatch")
        previous_start, previous_end = start, end
    for token in tokens:
        if (not isinstance(token["id"], int)
                or not 0 <= token["char_start"] < token["char_end"] <= len(text)
                or not math.isfinite(float(token["end_s"]))
                or not 0 <= float(token["end_s"]) <= duration
                or token["timing_kind"] not in ("word_end_projection", "nonlexical_anchor")
                or bool(token["lexical_timing_mask"]) != (token["timing_kind"] == "word_end_projection")):
            raise ValueError("invalid_token_projection")
    if record.get("training_eligible") is not False or record.get("timing_scope") != "decoded_clip_relative":
        raise ValueError("unsafe_training_or_timing_scope")
    return dict(state="ACCEPT_ORIGINAL", reason="stored_alignment_invariants_pass",
                max_overshoot_s=0.0, corrected=None)


def classify_failed(record, encoding=None):
    """Classify the one observed failure mode; only <=80 ms is auto-corrected."""
    if record.get("alignment_ok") is not False:
        raise ValueError("not_a_failed_record")
    if record.get("error") != "ValueError: invalid_or_out_of_clip_timestamp":
        return dict(state="REJECT_INVALID", reason="unexpected_failure_type",
                    max_overshoot_s=None, corrected=None)
    duration = float(record["duration_s"])
    items = _items(record)
    if not items or not math.isfinite(duration) or duration <= 0:
        return dict(state="REJECT_INVALID", reason="missing_items_or_duration",
                    max_overshoot_s=None, corrected=None)
    previous_start = previous_end = -1.0
    overshoot = 0.0
    for item in items:
        start, end = item["start_time"], item["end_time"]
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end
                and start >= previous_start and end >= previous_end):
            return dict(state="REJECT_INVALID", reason="non_boundary_timestamp_error",
                        max_overshoot_s=None, corrected=None)
        overshoot = max(overshoot, start - duration, end - duration)
        previous_start, previous_end = start, end
    if overshoot <= 0:
        return dict(state="REJECT_INVALID", reason="failure_not_reproduced",
                    max_overshoot_s=overshoot, corrected=None)
    if overshoot > POLICY["review_max_s"] + 1e-9:
        return dict(state="REJECT_GT_320MS", reason="clip_overshoot_gt_320ms",
                    max_overshoot_s=overshoot, corrected=None)
    if overshoot > POLICY["auto_clamp_max_s"] + 1e-9:
        return dict(state="REVIEW_80_320MS", reason="clip_overshoot_requires_review",
                    max_overshoot_s=overshoot, corrected=None)
    if encoding is None:
        return dict(state="NEED_ENCODING", reason="encoding_required_for_auto_correction",
                    max_overshoot_s=overshoot, corrected=None)
    clamped = [dict(text=x["text"], start_time=min(x["start_time"], duration),
                    end_time=min(x["end_time"], duration)) for x in items]
    projected = project_items(record["target_text"], clamped, encoding, duration)
    corrected = dict(record)
    corrected.update(projected, aligner_items=clamped, alignment_ok=True,
                     original_alignment_ok=False,
                     source_alignment_error=corrected.pop("error"),
                     timing_quality="forced_aligned_boundary_clamped",
                     training_eligible=False,
                     qc=dict(policy=POLICY["version"], state="ACCEPT_CLAMPED_80MS",
                             max_overshoot_s=overshoot, clamp_s=overshoot,
                             acoustic_spot_check_required=True))
    return dict(state="ACCEPT_CLAMPED_80MS", reason="clip_overshoot_le_80ms_clamped",
                max_overshoot_s=overshoot, corrected=corrected)


def sample_rank(key, state):
    return int(hashlib.sha256(f"{POLICY['version']}\0{state}\0{key}".encode()).hexdigest(), 16)
