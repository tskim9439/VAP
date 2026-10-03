"""Audio helpers: reading and integer-ratio resampling (polyphase), 16-bit PCM output."""
from fractions import Fraction
from pathlib import Path

import numpy as np


def read_mono(path) -> "tuple[np.ndarray, int]":
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim != 1:
        raise ValueError(f"{path}: expected mono audio, got shape {x.shape}")
    return x, sr


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    f = Fraction(sr_out, sr_in)
    return np.clip(resample_poly(x, f.numerator, f.denominator), -1.0, 1.0).astype(np.float32)


def write_pcm16(path, x: np.ndarray, sr: int) -> None:
    """Atomic write (tmp + rename) of 16-bit PCM WAV."""
    import soundfile as sf
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp" + p.suffix)
    sf.write(str(tmp), x, sr, subtype="PCM_16")
    tmp.replace(p)
