"""Phase 2 mono 혼합 — 화자별 채널(또는 조각)을 대화 시간축에 놓고 더한다 (정본 §8 "모델 입력은 mono 혼합").

  load_channel : ChannelRef → (T,) float32 @16 kHz. path 는 streams.load_utt_audio 규약(#chN·리샘플), pieces 는 조각을 offset 에 배치(나머지 무음 0).
  mix_dialogue : 화자별 발화 구간 RMS 를 target_dbfs 로 맞추고(±jitter_db 무작위) 합산, 피크 제한. 미래 통계를 쓰지 않도록 게인은 대화 전체에서 하나(입력 정규화이지 모델 단서가 아님).
"""
import math, random
from typing import Dict, List, Tuple, Optional
import numpy as np
from .dialogue import Dialogue, ChannelRef, Utterance
from .streams import load_utt_audio, SR

def load_channel(ref: ChannelRef, duration_s: float, sr: int = SR) -> np.ndarray:
    T = int(round(duration_s * sr)); out = np.zeros(T, dtype=np.float32)
    if ref.pieces is not None:
        for pc in ref.pieces:                                   # (path, offset) = 조각 파일 전체, (path, offset, src_offset, dur) = 원본의 구간(stitching)
            p, off = pc[0], pc[1]; x = load_utt_audio(p, pc[2], pc[3]) if len(pc) >= 4 else load_utt_audio(p)
            a = int(round(off * sr)); b = min(T, a + len(x))
            if a < T and b > a: out[a:b] = x[: b - a]
        return out
    x = load_utt_audio(ref.path); n = min(T, len(x)); out[:n] = x[:n]; return out

def _rms_over(x: np.ndarray, spans: List[Tuple[float, float]], sr: int) -> float:
    if not spans: return float(np.sqrt(np.mean(x ** 2) + 1e-12))
    acc, n = 0.0, 0
    for s, e in spans:
        a, b = int(s * sr), min(len(x), int(e * sr))
        if b > a: seg = x[a:b]; acc += float(np.sum(seg ** 2)); n += b - a
    return float(math.sqrt(acc / n)) if n else 1e-6

def mix_dialogue(dlg: Dialogue, sr: int = SR, target_dbfs: float = -23.0, jitter_db: float = 0.0, seed: int = 0, peak: float = 0.98) -> Tuple[np.ndarray, Dict]:
    """→ (mono (T,), meta{gains, peak_scale}). 채널이 없는 화자는 건너뛴다(meta.missing_channels)."""
    rng = random.Random(f"{dlg.conv_id}:{seed}"); T = int(round(dlg.duration_s * sr)); mono = np.zeros(T, dtype=np.float32); gains = {}; missing = []
    spans: Dict[str, List[Tuple[float, float]]] = {}
    for u in dlg.utterances: spans.setdefault(u.speaker, []).append((u.start, u.end))
    for spk in dlg.speakers:
        ref = dlg.channels.get(spk)
        if ref is None or (ref.pieces is None and not ref.path) or (ref.pieces is not None and not ref.pieces): missing.append(spk); continue
        x = load_channel(ref, dlg.duration_s, sr); rms = _rms_over(x, spans.get(spk, []), sr)
        g = (10 ** (target_dbfs / 20)) / max(rms, 1e-6) * (10 ** (rng.uniform(-jitter_db, jitter_db) / 20) if jitter_db else 1.0)
        gains[spk] = float(g); mono += x * g
    pk = float(np.max(np.abs(mono))) if T else 0.0; scale = peak / pk if pk > peak else 1.0
    return mono * scale, dict(gains=gains, peak_scale=scale, missing_channels=missing, sr=sr)

def crop_audio(mono: np.ndarray, t0: float, t1: float, sr: int = SR) -> np.ndarray:
    return mono[int(round(t0 * sr)): int(round(t1 * sr))]
