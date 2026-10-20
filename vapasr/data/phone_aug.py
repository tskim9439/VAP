"""전화 대역 증강(학습 전용, 2026-10-20) — TAXI(실제 독↔영 전화)에서 VibeVoice 대비 크게 뒤진 원인 후보(전화 음성이 학습에 거의 없다)를 겨눈다.

항목마다 확률 p 로(데이터로더 워커의 random 을 쓴다):
  1. 8 kHz 로 내린다(resample_poly 1/2 — 선형 위상 FIR 이라 지연이 없다).
  2. 전화 대역 통과: 하한 U(low) Hz · 상한 U(high) Hz 4 차 Butterworth 를 앞뒤로 건다(sosfiltfilt — 영위상, 단어 시각이 밀리지 않는다).
  3. 확률 mulaw_p 로 G.711 μ-law 8 비트 양자화(부호화 → 복호화).
  4. 16 kHz 로 올리고(resample_poly 2/1) 길이와 RMS 를 원래와 맞춘다. [-1, 1] 을 넘으면 전체를 줄인다.
시퀀스·K·라벨은 바꾸지 않는다(오디오만). AugChain 은 잡음·잔향(noise_aug.NoiseAugment) 뒤에 이어 거는 묶음이다(전화기는 방 소리 다음에 걸린다)."""
import math
from typing import Sequence

import numpy as np

MU = 255.0


def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """G.711 μ-law 8 비트 부호화·복호화(입력은 [-1, 1] 로 자른다)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / math.log1p(MU)
    q = np.round((y + 1.0) * 127.5) / 127.5 - 1.0
    return (np.sign(q) * np.expm1(np.abs(q) * math.log1p(MU)) / MU).astype(np.float32)


class PhoneAugment:
    def __init__(self, p: float = 0.3, low: Sequence[float] = (200.0, 400.0), high: Sequence[float] = (3000.0, 3600.0), mulaw_p: float = 0.5):
        if not 0.0 <= p <= 1.0 or not 0.0 <= mulaw_p <= 1.0: raise ValueError(f"확률은 [0, 1] — p {p}, mulaw_p {mulaw_p}")
        if not (0 < low[0] <= low[1] < high[0] <= high[1] < 4000): raise ValueError(f"대역(Hz)은 0 < low ≤ … < high ≤ … < 4000 — {low} {high}")
        self.p, self.low, self.high, self.mulaw_p = float(p), tuple(map(float, low)), tuple(map(float, high)), float(mulaw_p)

    def settings(self) -> dict:
        return dict(kind="phone", p=self.p, low=list(self.low), high=list(self.high), mulaw_p=self.mulaw_p)

    def __call__(self, wav: np.ndarray, rng) -> np.ndarray:
        x = np.asarray(wav, dtype=np.float32)
        if len(x) < 256 or rng.random() >= self.p: return x
        from scipy.signal import butter, resample_poly, sosfiltfilt
        y = resample_poly(x, 1, 2)
        sos = butter(4, [rng.uniform(*self.low), rng.uniform(*self.high)], btype="bandpass", fs=8000, output="sos")
        if len(y) > 3 * (2 * len(sos) + 1): y = sosfiltfilt(sos, y)
        if rng.random() < self.mulaw_p: y = mulaw_roundtrip(y)
        z = resample_poly(y, 2, 1)[: len(x)]
        if len(z) < len(x): z = np.pad(z, (0, len(x) - len(z)))
        rx, rz = math.sqrt(float(np.mean(x.astype(np.float64) ** 2))), math.sqrt(float(np.mean(z.astype(np.float64) ** 2)))
        if rz > 1e-9: z = z * (rx / rz)
        pk = float(np.max(np.abs(z))) if len(z) else 0.0
        if pk > 1.0: z = z / pk
        return np.ascontiguousarray(z, dtype=np.float32)


class AugChain:
    """증강 여럿을 차례로(None 은 건너뛴다). settings 는 각 증강의 설정 목록."""
    def __init__(self, *augs):
        self.augs = [a for a in augs if a is not None]

    def settings(self) -> dict:
        return dict(chain=[a.settings() for a in self.augs])

    def __call__(self, wav: np.ndarray, rng) -> np.ndarray:
        for a in self.augs: wav = a(wav, rng)
        return wav
