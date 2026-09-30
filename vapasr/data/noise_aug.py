"""잡음·잔향 증강(학습 전용) — experiments/build_noise_bank.py 가 만든 뱅크(<bank>/noise.jsonl + audio/, rir.npy + rir.jsonl)를 쓴다.

항목마다(데이터로더 워커의 random 을 쓴다):
  1. 잔향: 확률 p_rir 로 RIR 하나를 골라 직접음 최대점을 t=0 에 맞춰(앞부분을 잘라) 합성곱 → 길이는 그대로, 단어 시각·청크 라벨이 밀리지 않는다.
     합성 뒤 RMS 를 원래와 같게 맞춘다.
  2. 가산 잡음: 확률 p_noise 로 잡음(확률 music_frac 이면 음악) 파일 하나를 골라 무작위 위치에서 발화 길이만큼(짧으면 이어 붙임) 읽고,
     SNR ~ U(snr) dB(음악은 U(music_snr))로 더한다. 신호 전력 = 20 ms 프레임 전력의 상위 절반 평균(긴 무음이 SNR 을 왜곡하지 않게).
     합이 [-1, 1] 을 넘으면 전체를 줄인다.
RIR 뱅크는 np.load(mmap) 로 워커마다 처음 쓸 때 연다(페이지 캐시 공유). 잡음은 soundfile 로 필요한 구간만 읽는다.
시퀀스·K·라벨은 바꾸지 않는다(오디오만)."""
import json, math, os
from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np

SR = 16000


def speech_power(x: np.ndarray, frame: int = 320) -> float:
    n = len(x) // frame
    if n < 2: return float(np.mean(x.astype(np.float64) ** 2))
    p = np.mean(x[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1)
    return float(np.mean(np.sort(p)[n // 2:]))


class NoiseAugment:
    def __init__(self, bank: str, p_noise: float = 0.4, snr: Sequence[float] = (5.0, 30.0), music_snr: Sequence[float] = (10.0, 30.0),
                 music_frac: float = 0.25, p_rir: float = 0.2):
        self.bank = Path(bank); self.p_noise, self.p_rir, self.music_frac = float(p_noise), float(p_rir), float(music_frac)
        self.snr, self.music_snr = tuple(map(float, snr)), tuple(map(float, music_snr))
        rows = [json.loads(l) for l in open(self.bank / "noise.jsonl")] if self.p_noise > 0 else []
        self.noise = [r for r in rows if r["category"] == "noise"]; self.music = [r for r in rows if r["category"] == "music"]
        if self.p_noise > 0 and not self.noise: raise ValueError(f"잡음 뱅크가 비었다: {self.bank}/noise.jsonl")
        self.rir_len = [int(json.loads(l)["n"]) for l in open(self.bank / "rir.jsonl")] if self.p_rir > 0 else []
        if self.p_rir > 0 and not self.rir_len: raise ValueError(f"RIR 뱅크가 비었다: {self.bank}/rir.jsonl")
        self._rir = None

    def settings(self) -> dict:
        """학습 지문에 넣을 설정(뱅크 요약 sha 포함)."""
        import hashlib
        sm = self.bank / "summary.json"
        return dict(bank=str(self.bank), bank_summary_sha256=hashlib.sha256(sm.read_bytes()).hexdigest() if sm.exists() else None,
                    p_noise=self.p_noise, snr=list(self.snr), music_snr=list(self.music_snr), music_frac=self.music_frac, p_rir=self.p_rir,
                    n_noise=len(self.noise), n_music=len(self.music), n_rir=len(self.rir_len))

    def __call__(self, wav: np.ndarray, rng) -> np.ndarray:
        x = np.asarray(wav, dtype=np.float32)
        if len(x) == 0: return x
        if self.p_rir > 0 and rng.random() < self.p_rir: x = self.reverb(x, rng)
        if self.p_noise > 0 and rng.random() < self.p_noise: x = self.add_noise(x, rng)
        return np.ascontiguousarray(x, dtype=np.float32)

    def _rirs(self):
        if self._rir is None: self._rir = np.load(self.bank / "rir.npy", mmap_mode="r")
        return self._rir

    def reverb(self, x: np.ndarray, rng, i: Optional[int] = None) -> np.ndarray:
        from scipy.signal import fftconvolve
        i = rng.randrange(len(self.rir_len)) if i is None else i
        h = np.asarray(self._rirs()[i, : self.rir_len[i]], dtype=np.float32)
        k = int(np.argmax(np.abs(h))); h = h[k:]                                                  # 직접음을 t=0 에 → 지연 없음
        if len(h) == 0 or not np.any(h): return x
        y = fftconvolve(x, h)[: len(x)].astype(np.float32)
        rx, ry = math.sqrt(float(np.mean(x.astype(np.float64) ** 2))), math.sqrt(float(np.mean(y.astype(np.float64) ** 2)))
        return y * (rx / ry) if ry > 1e-9 else x

    def _segment(self, r: dict, n: int, rng) -> np.ndarray:
        import soundfile as sf
        N = int(r["n"])
        if N >= n:
            s = rng.randrange(N - n + 1); y, _ = sf.read(r["path"], start=s, frames=n, dtype="float32", always_2d=False)
        else:
            y, _ = sf.read(r["path"], dtype="float32", always_2d=False); s = rng.randrange(len(y)); y = np.roll(y, -s)
            y = np.tile(y, n // len(y) + 1)[:n]
        return np.asarray(y, dtype=np.float32).reshape(-1)[:n]

    def add_noise(self, x: np.ndarray, rng, r: Optional[dict] = None, snr_db: Optional[float] = None) -> np.ndarray:
        use_music = r is None and self.music and rng.random() < self.music_frac
        pool, lo_hi = (self.music, self.music_snr) if use_music else (self.noise, self.snr)
        r = rng.choice(pool) if r is None else r
        snr_db = rng.uniform(*lo_hi) if snr_db is None else snr_db
        nz = self._segment(r, len(x), rng); pn = float(np.mean(nz.astype(np.float64) ** 2)); px = speech_power(x)
        if pn < 1e-12 or px < 1e-12: return x
        y = x + nz * math.sqrt(px / (pn * 10 ** (snr_db / 10)))
        m = float(np.abs(y).max())
        return (y / m * 0.99 if m > 1.0 else y).astype(np.float32)


def from_args(bank: Optional[str], p_noise: float, snr: str, p_rir: float, music_snr: str = "10,30", music_frac: float = 0.25) -> Optional[NoiseAugment]:
    """학습 스크립트 인자 → NoiseAugment 또는 None(bank 없음)."""
    if not bank: return None
    sn = tuple(float(v) for v in snr.split(",")); ms = tuple(float(v) for v in music_snr.split(","))
    if len(sn) != 2 or len(ms) != 2 or sn[0] > sn[1] or ms[0] > ms[1]: raise ValueError(f"SNR 범위는 'lo,hi': {snr!r} {music_snr!r}")
    return NoiseAugment(bank, p_noise=p_noise, snr=sn, music_snr=ms, music_frac=music_frac, p_rir=p_rir)
