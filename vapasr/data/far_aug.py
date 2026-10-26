"""원거리 증강(학습 전용, 2026-10-26 — 데이터 확장 D-1, 사용자 "모두 진행하는 데, 모두 진행하고 나서 E5 E6 학습 진행하자").

근접 마이크 대화를 원거리 마이크 하나로 들은 것처럼 만든다. 창마다(확률 p):
  1. 화자별 신호(stems — dialogue_mix.mix_stems 캐시 (S, T), 합이 mono 캐시)가 있으면 화자마다 다른 RIR 을 곱해 섞는다.
     RIR 은 한 방에서 고른다 — 모의 RIR(RIRS_NOISES simulated_rirs/<크기>/RoomNNN)은 같은 RoomNNN 안에서 화자마다 다른 위치,
     확률 p_real 이면 실제 RIR 풀에서 화자마다 다른 것. 직접음 최대점을 t=0 에 맞춰(앞을 잘라) 지연이 없다 — 단어 시각·청크 라벨이 밀리지 않는다.
     화자마다 이득 ±gain_db(마이크까지 거리 차). 섞은 뒤 원래 mono 창과 같은 음성 전력으로 맞춘다.
  2. stems 가 없으면(원거리 버전·합성 셋 등) mono 창에 RIR 하나.
  3. 확률 p_noise 로 가산 잡음(NoiseAugment.add_noise, SNR U(snr)).
RIR·잡음 뱅크는 experiments/build_noise_bank.py 산출(<bank>/rir.npy + rir.jsonl{source, name, n}, noise.jsonl)을 그대로 쓴다. 시퀀스·K·라벨은 바꾸지 않는다."""
import math, re
from typing import Optional, Sequence

import numpy as np

from .noise_aug import NoiseAugment, speech_power


class FarFieldAugment:
    def __init__(self, bank: str, p: float = 0.5, p_noise: float = 0.8, snr: Sequence[float] = (5.0, 25.0), gain_db: float = 3.0, p_real: float = 0.3):
        import json
        from pathlib import Path
        self.p, self.p_real, self.gain_db = float(p), float(p_real), float(gain_db)
        self.na = NoiseAugment(bank, p_noise=p_noise, snr=snr, music_snr=snr, music_frac=0.0, p_rir=1.0)              # 잡음·RIR 읽기만 빌린다(음악 없음)
        rows = [json.loads(l) for l in open(Path(bank) / "rir.jsonl")]
        self.real = [i for i, r in enumerate(rows) if r.get("source") == "real"]; rooms = {}
        for i, r in enumerate(rows):
            m = re.search(r"simulated_rirs/([^/]+)/(Room\d+)/", r.get("name", ""))
            if m: rooms.setdefault((m.group(1), m.group(2)), []).append(i)
        self.rooms = [v for v in rooms.values() if len(v) >= 2]
        if not self.rooms and not self.real: raise ValueError(f"방별 RIR 이 없다: {bank}/rir.jsonl")

    def settings(self) -> dict:
        return dict(p=self.p, p_real=self.p_real, gain_db=self.gain_db, rooms=len(self.rooms), real=len(self.real), noise=self.na.settings())

    def pick_rirs(self, n: int, rng) -> list:
        """한 방에서 화자 n 명의 RIR 번호(가능하면 서로 다르게)."""
        if self.real and (not self.rooms or rng.random() < self.p_real): pool = self.real
        else: pool = rng.choice(self.rooms)
        return rng.sample(pool, n) if len(pool) >= n else [rng.choice(pool) for _ in range(n)]

    def render(self, stems: np.ndarray, mono: np.ndarray, rng) -> np.ndarray:
        """stems (S, n) — 같은 창의 화자별 신호, mono (n,) — 원래 mono 창(세기 기준)."""
        st = np.asarray(stems, dtype=np.float32); x = np.asarray(mono, dtype=np.float32)
        live = [s for s in range(st.shape[0]) if np.any(st[s])]
        if not live: return self.mono(x, rng)
        y = np.zeros(st.shape[1], dtype=np.float32)
        for s, i in zip(live, self.pick_rirs(len(live), rng)):
            y += self.na.reverb(st[s], rng, i) * float(10 ** (rng.uniform(-self.gain_db, self.gain_db) / 20))
        return self._finish(y, x, rng)

    def mono(self, wav: np.ndarray, rng) -> np.ndarray:
        x = np.asarray(wav, dtype=np.float32)
        return self._finish(self.na.reverb(x, rng, self.pick_rirs(1, rng)[0]), x, rng)

    def _finish(self, y: np.ndarray, ref: np.ndarray, rng) -> np.ndarray:
        py, pr = speech_power(y), speech_power(ref)
        if py > 1e-12 and pr > 1e-12: y = y * math.sqrt(pr / py)
        if self.na.p_noise > 0 and rng.random() < self.na.p_noise: y = self.na.add_noise(y, rng)
        m = float(np.abs(y).max()) if len(y) else 0.0
        return np.ascontiguousarray(y / m * 0.99 if m > 1.0 else y, dtype=np.float32)


def from_args(bank: Optional[str], p: float, p_noise: float = 0.8, snr: str = "5,25", gain_db: float = 3.0, p_real: float = 0.3) -> Optional[FarFieldAugment]:
    if not bank or p <= 0: return None
    sn = tuple(float(v) for v in snr.split(","))
    if len(sn) != 2 or sn[0] > sn[1]: raise ValueError(f"SNR 범위는 'lo,hi': {snr!r}")
    return FarFieldAugment(bank, p=p, p_noise=p_noise, snr=sn, gain_db=gain_db, p_real=p_real)
