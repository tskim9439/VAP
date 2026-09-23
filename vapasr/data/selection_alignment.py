"""Verbatim teacher-text alignment utilities (no text normalization)."""
import hashlib
import math
import unicodedata


SCHEMA = "qwen-verbatim-align-v1"


def target_text(row):
    target = row["recommended_training_target"]
    text = target["text"]
    if (row["selection_state"] != "KEEP_ASR_SILVER"
            or target["normalization"] != "none"
            or not isinstance(text, str) or not text.strip()
            or text != row["transcripts"]["qwen_raw"]):
        raise ValueError("not_a_verbatim_selected_qwen_target")
    return text


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def surface(text):
    # Mirrors the installed aligner's clean_token *only for offset matching*.
    # The saved target and tokenizer input are never changed. No number expansion.
    positions = [i for i, ch in enumerate(text)
                 if ch == "'" or unicodedata.category(ch)[0] in "LN"]
    return "".join(text[i] for i in positions), positions


def project_items(text, items, encoding, duration_s):
    """Aligner words -> original character spans -> word-projected BPE timing.

    Nonlexical tokens get an explicit punctuation/space anchor, not an acoustic
    timestamp. Mismatched text fails closed; never use cursor/uniform fallbacks.
    """
    raw, positions = surface(text)
    parts = [surface(it["text"])[0] for it in items]
    if not raw or not parts or any(not part for part in parts) or "".join(parts) != raw:
        raise ValueError("aligner_surface_mismatch")
    words, char_to_word, cursor = [], {}, 0
    prev_start = prev_end = -1.0
    for i, (item, part) in enumerate(zip(items, parts)):
        start, end = float(item["start_time"]), float(item["end_time"])
        if not (math.isfinite(start) and math.isfinite(end)
                and 0 <= start <= end <= duration_s
                and start >= prev_start and end >= prev_end):
            raise ValueError("invalid_or_out_of_clip_timestamp")
        covered = positions[cursor:cursor + len(part)]
        char_to_word.update((p, i) for p in covered)
        words.append(dict(text=item["text"], char_start=covered[0],
                          char_end=covered[-1] + 1, start_s=start, end_s=end))
        cursor += len(part)
        prev_start, prev_end = start, end
    ids, offsets = encoding["input_ids"], encoding["offset_mapping"]
    if not ids or len(ids) != len(offsets):
        raise ValueError("invalid_tokenizer_offsets")
    tokens, seen = [], set()
    for tid, (left, right) in zip(ids, offsets):
        if not 0 <= left < right <= len(text):
            raise ValueError("invalid_tokenizer_span")
        matched = sorted({char_to_word[p] for p in range(left, right) if p in char_to_word})
        seen.update(p for p in range(left, right) if p in char_to_word)
        if matched:
            end = max(words[i]["end_s"] for i in matched)
            kind = "word_end_projection"
        else:
            previous = [w for w in words if w["char_end"] <= left]
            end = previous[-1]["end_s"] if previous else words[0]["start_s"]
            kind = "nonlexical_anchor"
        tokens.append(dict(id=int(tid), char_start=int(left), char_end=int(right),
                           end_s=end, timing_kind=kind, lexical_timing_mask=bool(matched)))
    if seen != set(positions):
        raise ValueError("tokenizer_dropped_lexical_characters")
    return dict(words=words, tokens=tokens,
                zero_duration_words=sum(w["start_s"] == w["end_s"] for w in words),
                timing_scope="decoded_clip_relative", timing_quality="forced_aligned_unverified",
                training_eligible=False, speaker_supervision=False, turn_supervision=False)
