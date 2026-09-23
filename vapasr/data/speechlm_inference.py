"""Read-only SpeechLM audio and conservative verbatim pseudo-label selection."""
from collections import OrderedDict
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import threading

from .selection import digest

POLICY = dict(version="speechlm-verbatim-pair-v1", max_pair_error=.05,
              max_duration_s=30., min_rms=.005, tn_scope="comparison_only",
              reference_agreement_required=False, training_eligible=False,
              dedup_status="required_before_training")


def adapt(row):
    """No transcript rewrite; key identifies a source row, not a dedup claim."""
    lang = {"en": "English", "ko": "Korean", "English": "English",
            "Korean": "Korean"}.get(row["language"])
    reasons = []
    duration = row["duration_sec"]
    if lang is None:
        reasons.append("unsupported_language")
    if row["split"] != "train":
        reasons.append("held_out_split")
    if not isinstance(duration, (float, int)) or not math.isfinite(duration) or not 0 < duration <= 30:
        reasons.append("outside_0_30s_scope")
    if row["num_channels"] != 1:
        reasons.append("not_verified_mono")
    if not isinstance(row["original_transcript"], str) or not row["original_transcript"].strip():
        reasons.append("empty_source_transcript")
    return dict(key=digest([row["source_arrow"], row["source_row_index"]]),
                source=row["catalog_id"], lang=lang, duration_s=duration,
                text=row["original_transcript"], raw_text=row["original_transcript"],
                split=row["split"], source_manifest=row,
                audio=dict(path=row["audio_tar"] + "::" + row["audio_member"],
                           offset_s=None, duration_s=None),
                selection_state="HOLD_METADATA" if reasons else "PENDING_AUDIO_TEACHERS",
                reasons=reasons, training_eligible=False, dedup_status="unverified")


class TarReader:
    """Bounded thread-safe index cache; NEVER writes next to source archives."""
    def __init__(self, cache_root, max_indexes=32):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = OrderedDict()
        self.max_indexes = max_indexes
        self.lock = threading.Lock()

    def index(self, archive):
        path = Path(archive)
        if path.suffix != ".tar":
            raise ValueError("Only uncompressed .tar supports indexed reads")
        stat = path.stat()
        identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
        key = digest(identity)
        with self.lock:
            if key not in self.cache:
                dest = self.root / (key + ".json")
                with (self.root / (key + ".lock")).open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not dest.exists():
                        index = {}
                        with tarfile.open(path, "r:") as handle:
                            for item in handle:
                                if item.isfile():
                                    if item.name in index:
                                        raise ValueError("Duplicate tar member")
                                    index[item.name] = [item.offset_data, item.size]
                        tmp = dest.with_suffix(f".{os.getpid()}.tmp")
                        after = path.stat()
                        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                            raise ValueError("Tar changed during indexing")
                        tmp.write_text(json.dumps(dict(identity=identity, members=index)))
                        tmp.replace(dest)
                    content = json.loads(dest.read_text())
                    if content["identity"] != identity:
                        raise ValueError("Tar index identity mismatch")
                    self.cache[key] = content["members"]
                if len(self.cache) > self.max_indexes:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            return self.cache[key]

    def read(self, archive, member):
        offset, size = self.index(archive)[member]
        path = Path(archive)
        with path.open("rb") as handle:
            handle.seek(offset)
            value = handle.read(size)
        if len(value) != size:
            raise ValueError("Truncated tar member")
        return value


def decode(row, reader):
    import numpy as np
    import soundfile as sf
    import soxr
    source = row["source_manifest"]
    data = reader.read(source["audio_tar"], source["audio_member"])
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if x.shape[1] != 1 or sr != source["sample_rate"]:
        raise ValueError("Audio header/manifest channel or sample-rate mismatch")
    x = x[:, 0]
    if not len(x) or not np.isfinite(x).all() or not np.any(x):
        raise ValueError("Empty, nonfinite or all-zero audio")
    duration = len(x) / sr
    review = []
    if abs(duration - row["duration_s"]) > max(.1, .05 * row["duration_s"]):
        review.append("duration_mismatch")
    clipping = float(np.mean(np.abs(x) >= .999))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if clipping > .01:
        review.append("clipping_above_1pct")
    if rms < POLICY["min_rms"]:
        review.append("low_rms")
    if not 0 < duration <= POLICY["max_duration_s"]:
        review.append("outside_0_30s_scope")
    if sr != 16000:
        x = soxr.resample(x, sr, 16000)
    x = np.ascontiguousarray(x, dtype="<f4")
    info = dict(sha256=hashlib.sha256(x.tobytes()).hexdigest(), duration_s=duration,
                original_sample_rate=sr, original_channels=1, sample_rate=16000,
                rms=rms, clipping_ratio=clipping, review=review)
    return x, info


def decide(row, qwen, whisper, metrics):
    if row.get("reasons"):
        return "HOLD_METADATA"
    if any(t.get("error") for t in (qwen, whisper)):
        return "HOLD_TEACHER"
    if row["audio_info"]["review"]:
        return "HOLD_AUDIO"
    if any(t.get("warning") for t in (qwen, whisper)):
        return "HOLD_GENERATION"
    if not metrics or min(metrics["qwen_units"], metrics["whisper_units"]) <= 0:
        return "HOLD_TEXT"
    rate = metrics["qwen_whisper"]
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Invalid pair distance")
    return "KEEP_ASR_SILVER" if rate <= POLICY["max_pair_error"] else "HOLD_DISAGREEMENT"
