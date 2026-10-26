"""E5 슬롯 활동 입력 흔들기(학습 전용, 2026-10-26 — output-diarization-integration-20261025 6 절).

정답 슬롯 활동 (K, S)(slot_activity — 80 ms 청크, 슬롯 = 창에서 먼저 말한 순서)을 E1 화자 분리 가지의 오류 모양으로 망가뜨려 LLM 입력에 준다
(AMI SDM eval: miss 13 % 의 46 % 가 겹친 구간, 말이 적은 화자(창 말 몫 < 5 %)의 68 % 를 통째로 놓침, fa 4–5 %, 구간 끝 흔들림).
항목마다(데이터로더 워커의 random) 확률 clean 이면 그대로. 아니면 차례로:
  1. 화자 통째 누락: 슬롯마다 창 안 활동 몫이 short_frac 미만이면 p_short, 아니면 p_drop.
  2. 구간 끝 흔들림: 켜진 구간마다 시작·끝을 ±jitter 청크 안에서 따로 옮긴다(구간이 사라지지는 않는다).
  3. 겹침 누락: 둘 이상 켜진 연속 구간마다 확률 p_ovl 로 그중 한 슬롯을 그 구간에서 끈다.
  4. 거짓 구간: 슬롯·청크마다 확률 p_fa 로 1–4 청크짜리 켜짐을 더한다.
시퀀스·라벨은 바꾸지 않는다(입력만)."""
from typing import Dict, Optional

import numpy as np

DEFAULT = dict(clean=0.3, drop=0.05, short=0.4, short_frac=0.05, jitter=3, ovl=0.3, fa=0.002)


class ActNoise:
    def __init__(self, clean: float = 0.3, drop: float = 0.05, short: float = 0.4, short_frac: float = 0.05, jitter: int = 3, ovl: float = 0.3, fa: float = 0.002):
        self.clean, self.drop, self.short, self.short_frac, self.jitter, self.ovl, self.fa = float(clean), float(drop), float(short), float(short_frac), int(jitter), float(ovl), float(fa)
        for v in (self.clean, self.drop, self.short, self.ovl, self.fa):
            if not 0.0 <= v <= 1.0: raise ValueError(f"확률은 [0, 1]: {self.settings()}")

    def settings(self) -> Dict[str, float]:
        return dict(clean=self.clean, drop=self.drop, short=self.short, short_frac=self.short_frac, jitter=self.jitter, ovl=self.ovl, fa=self.fa)

    def __call__(self, act, rng) -> np.ndarray:
        a = np.array(act, dtype=np.float32, copy=True)
        if a.ndim != 2 or rng.random() < self.clean: return a
        K, S = a.shape
        for s in range(S):                                                                     # 1. 화자 통째 누락
            share = float(a[:, s].mean()) if K else 0.0
            if share > 0 and rng.random() < (self.short if share < self.short_frac else self.drop): a[:, s] = 0.0
        if self.jitter > 0:                                                                     # 2. 구간 끝 흔들림
            for s in range(S):
                out = np.zeros(K, np.float32)
                for i, j in _runs(a[:, s]):
                    i2 = min(max(0, i + rng.randint(-self.jitter, self.jitter)), K - 1); j2 = min(max(i2 + 1, j + rng.randint(-self.jitter, self.jitter)), K)
                    out[i2:j2] = 1.0
                a[:, s] = out
        if self.ovl > 0:                                                                        # 3. 겹침 누락
            for i, j in _runs((a.sum(1) >= 2).astype(np.float32)):
                if rng.random() < self.ovl:
                    on = [s for s in range(S) if a[i:j, s].any()]; a[i:j, rng.choice(on)] = 0.0
        if self.fa > 0:                                                                         # 4. 거짓 구간
            for s in range(S):
                for k in range(K):
                    if rng.random() < self.fa: a[k: k + rng.randint(1, 4), s] = 1.0
        return a


def _runs(x):
    """0/1 열 → 켜진 연속 구간 [(시작, 끝)]."""
    d = np.diff(np.concatenate([[0], (np.asarray(x) > 0).astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def from_spec(spec: Optional[str]) -> ActNoise:
    """'clean=0.3,drop=0.05,…'(빈 문자열 = 기본값) → ActNoise."""
    kw = dict(DEFAULT)
    for x in (spec or "").split(","):
        if not x.strip(): continue
        k, _, v = x.partition("=")
        if k.strip() not in DEFAULT: raise ValueError(f"--act-noise 키: {sorted(DEFAULT)} — {k!r}")
        kw[k.strip()] = float(v)
    return ActNoise(**kw)
