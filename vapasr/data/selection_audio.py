"""Strict selection-time decoding. Does not silently clamp channels or crops."""
import hashlib
import io
from pathlib import Path


def decode(row):
    import numpy as np
    import soundfile as sf
    ref = row["audio"]
    if not ref or not ref.get("path"):
        raise ValueError("missing_audio_reference")
    path = ref["path"]
    ch, explicit_channel = 0, "#ch" in path
    if explicit_channel:
        path, suffix = path.rsplit("#ch", 1)
        ch = int(suffix)
    if ch < 0:
        raise ValueError("negative_channel")
    offset = ref.get("offset_s") or 0.0
    duration = ref.get("duration_s")
    odd_byte = False
    is_pcm = path.split("::")[-1].endswith(".pcm")
    if "::" in path:
        from .archive import read_member
        archive, member = path.split("::", 1)
        data = read_member(archive, member)
        source = io.BytesIO(data)
    else:
        source = path
    if is_pcm:
        if ch != 0:
            raise ValueError("raw_pcm_channel_not_zero")
        data = source.getvalue() if isinstance(source, io.BytesIO) else Path(source).read_bytes()
        odd_byte = len(data) % 2 != 0
        raw = np.frombuffer(data[:len(data) - len(data) % 2], dtype="<i2")
        sr, channels, total = 16000, 1, len(raw)
        start = round(offset * sr)
        count = total - start if duration is None else round(duration * sr)
        x = raw[max(0, start):max(0, start) + count].astype(np.float32) / 32768
    else:
        with sf.SoundFile(source) as f:
            sr, channels, total = f.samplerate, f.channels, f.frames
            if ch >= channels:
                raise ValueError(f"channel_out_of_range:{ch}/{channels}")
            start = round(offset * sr)
            count = total - start if duration is None else round(duration * sr)
            if start < 0 or start >= total:
                raise ValueError("offset_out_of_range")
            f.seek(start)
            x = f.read(count, dtype="float32", always_2d=True)[:, ch]
    if start < 0 or count <= 0 or len(x) == 0:
        raise ValueError("empty_or_invalid_crop")
    if len(x) < count - round(0.02 * sr):
        raise ValueError("truncated_audio_crop")
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_audio")
    if not np.any(x):
        raise ValueError("all_zero_audio_with_text")
    review = []
    if channels > 1 and not explicit_channel:
        review.append("multichannel_without_explicit_channel_mapping")
    actual = len(x) / sr
    if abs(actual - row["duration_s"]) > max(.1, .05 * row["duration_s"]):
        review.append("label_audio_duration_mismatch")
    clipping = float(np.mean(np.abs(x) >= .999))
    if clipping > .01:
        review.append("clipping_above_1pct")
    if sr != 16000:
        import soxr
        x = soxr.resample(x, sr, 16000)
    x = np.ascontiguousarray(x, dtype="<f4")
    return x, dict(sha256=hashlib.sha256(x.tobytes()).hexdigest(), sample_rate=16000,
                   original_sample_rate=sr, original_channels=channels,
                   source_duration_s=total / sr, duration_s=actual,
                   rms=float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
                   clipping_ratio=clipping, discarded_trailing_pcm_byte=odd_byte,
                   review=review)
