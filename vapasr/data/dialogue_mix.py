"""Phase 2 mono 혼합 — 화자별 채널(또는 조각)을 대화 시간축에 놓고 더한다 (정본 §8 "모델 입력은 mono 혼합").

  load_channel : ChannelRef → (T,) float32 @16 kHz. path 는 streams.load_utt_audio 규약(#chN·리샘플), pieces 는 조각을 offset 에 배치(나머지 무음 0).
  mix_dialogue : 화자별 발화 구간 RMS 를 target_dbfs 로 맞추고(±jitter_db 무작위) 합산, 피크 제한. 미래 통계를 쓰지 않도록 게인은 대화 전체에서 하나(입력 정규화이지 모델 단서가 아님).
                 원거리 버전(meta.mono_src, 2026-10-26)이면 far_mono 로 장치 녹음 한 채널을 쓴다. 혼합 조각(meta.mixed_pieces — B 등급)이면 게인 없이 잇는다.
  mix_stems    : mix_dialogue 와 같은 게인의 화자별 신호 (S, T) — 학습 중 화자마다 다른 RIR 을 곱해 다시 섞는 원거리 증강(far_aug)의 재료.
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

def far_mono(dlg: Dialogue, sr: int = SR, target_dbfs: float = -23.0, peak: float = 0.98) -> Tuple[np.ndarray, Dict]:
    """원거리 버전(meta.mono_src = {path, channel, offset_s}): 장치 녹음 한 채널을 대화 시간축에 놓는다 — 대화 시각 t 의 표본 = 장치 시각 t + offset_s
    (장치가 늦게 시작했으면 앞은 0). 이미 섞인 녹음이라 화자별 게인은 없고, 발화 구간 전체의 RMS 를 target_dbfs 로 맞춰 피크 제한한다."""
    src = dlg.meta["mono_src"]; T = int(round(dlg.duration_s * sr)); ch = int(src.get("channel", 0))
    x = load_utt_audio(src["path"] + (f"#ch{ch}" if ch else "")); off = int(round(float(src.get("offset_s", 0.0)) * sr)); y = np.zeros(T, dtype=np.float32)
    a, b = max(0, -off), min(T, len(x) - off)                                                   # y[a:b] = x[a + off : b + off]
    if b > a: y[a:b] = x[a + off: b + off]
    g = (10 ** (target_dbfs / 20)) / max(_rms_over(y, [(u.start, u.end) for u in dlg.utterances], sr), 1e-6); y *= g
    pk = float(np.max(np.abs(y))) if T else 0.0; scale = peak / pk if pk > peak else 1.0
    return y * scale, dict(gain=float(g), peak_scale=scale, src=dict(src), sr=sr)

def _speaker_gains(dlg: Dialogue, sr: int, target_dbfs: float, jitter_db: float, seed: int):
    rng = random.Random(f"{dlg.conv_id}:{seed}"); spans: Dict[str, List[Tuple[float, float]]] = {}
    for u in dlg.utterances: spans.setdefault(u.speaker, []).append((u.start, u.end))
    for spk in dlg.speakers:
        ref = dlg.channels.get(spk)
        if ref is None or (ref.pieces is None and not ref.path) or (ref.pieces is not None and not ref.pieces): yield spk, None, None; continue
        x = load_channel(ref, dlg.duration_s, sr); rms = _rms_over(x, spans.get(spk, []), sr)
        yield spk, x, (10 ** (target_dbfs / 20)) / max(rms, 1e-6) * (10 ** (rng.uniform(-jitter_db, jitter_db) / 20) if jitter_db else 1.0)

def mix_dialogue(dlg: Dialogue, sr: int = SR, target_dbfs: float = -23.0, jitter_db: float = 0.0, seed: int = 0, peak: float = 0.98) -> Tuple[np.ndarray, Dict]:
    """→ (mono (T,), meta{gains, peak_scale}). 채널이 없는 화자는 건너뛴다(meta.missing_channels). meta.mono_src 가 있으면 far_mono."""
    if dlg.meta and dlg.meta.get("mono_src"): return far_mono(dlg, sr, target_dbfs, peak)
    T = int(round(dlg.duration_s * sr)); mono = np.zeros(T, dtype=np.float32); gains = {}; missing = []
    if dlg.meta and dlg.meta.get("mixed_pieces"):                                            # B 등급(한 혼합 녹음의 조각, NIA23 회의): 게인 없이 이어 붙이고 전체를 맞춘다
        for spk in dlg.speakers:
            ref = dlg.channels.get(spk)
            if ref is None or not (ref.pieces or ref.path): missing.append(spk); continue
            x = load_channel(ref, dlg.duration_s, sr); n = min(T, len(x)); mono[:n] += x[:n]
        g = (10 ** (target_dbfs / 20)) / max(_rms_over(mono, [(u.start, u.end) for u in dlg.utterances], sr), 1e-6); mono *= g
        pk = float(np.max(np.abs(mono))) if T else 0.0; scale = peak / pk if pk > peak else 1.0
        return mono * scale, dict(gain=float(g), peak_scale=scale, missing_channels=missing, sr=sr, mixed_pieces=True)
    for spk, x, g in _speaker_gains(dlg, sr, target_dbfs, jitter_db, seed):
        if x is None: missing.append(spk); continue
        gains[spk] = float(g); mono += x * g
    pk = float(np.max(np.abs(mono))) if T else 0.0; scale = peak / pk if pk > peak else 1.0
    return mono * scale, dict(gains=gains, peak_scale=scale, missing_channels=missing, sr=sr)

def mix_stems(dlg: Dialogue, sr: int = SR, target_dbfs: float = -23.0, peak: float = 0.98) -> Tuple[np.ndarray, Dict]:
    """→ (stems (S, T), meta{speakers, gains, peak_scale}): mix_dialogue(jitter 0)와 같은 게인·피크 배율의 화자별 신호 — 합이 mono 캐시와 같다.
    채널이 없는 화자·원거리 버전(meta.mono_src)은 만들 수 없다(ValueError) — 화자별로 다시 섞을 재료가 없다."""
    if dlg.meta and (dlg.meta.get("mono_src") or dlg.meta.get("mixed_pieces")): raise ValueError(f"원거리 버전·혼합 조각(B 등급)은 화자별 신호가 없다: {dlg.conv_id}")
    T = int(round(dlg.duration_s * sr)); rows, gains, spk_ok = [], {}, []
    for spk, x, g in _speaker_gains(dlg, sr, target_dbfs, 0.0, 0):
        if x is None: raise ValueError(f"채널 없는 화자 {spk}: {dlg.conv_id}")
        rows.append(x * g); gains[spk] = float(g); spk_ok.append(spk)
    st = np.stack(rows) if rows else np.zeros((0, T), dtype=np.float32); mono = st.sum(0)
    pk = float(np.max(np.abs(mono))) if T and len(rows) else 0.0; scale = peak / pk if pk > peak else 1.0
    return (st * scale).astype(np.float32), dict(speakers=spk_ok, gains=gains, peak_scale=scale, sr=sr)

def crop_audio(mono: np.ndarray, t0: float, t1: float, sr: int = SR) -> np.ndarray:
    return mono[int(round(t0 * sr)): int(round(t1 * sr))]
