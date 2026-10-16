"""NeMo AudioToMelSpectrogramPreprocessor(eval) 의 스트리밍 log-mel (numpy fp32, encoder-spec §1).

pre-emphasis 0.97 → STFT(center=True, 0 패딩 256, 대칭 hann 400 을 512 의 56 위치에, hop 160, rfft 512) → |X|²
→ slaney mel 128 (체크포인트 fb) → ln(mel + 2^-24). normalize "NA" 이므로 정규화 없음, dither 없음.
유효 프레임 수 T = N // 160. 프레임 t 는 n ≥ 160t + 256 이면 확정(보수 규칙) — 확정 프레임은 이후 바뀌지 않는다.
"""
from __future__ import annotations

import numpy as np

SR, HOP, WIN, NFFT, NMEL = 16000, 160, 400, 512, 128
PAD = NFFT // 2                      # center=True 의 좌우 0 패딩 (256)
PREEMPH = np.float32(0.97)
LOG_GUARD = np.float32(2.0 ** -24)


def check_preprocessor_cfg(pc: dict) -> None:
    """nemotron_encoder.json 의 preprocessor 가 이 구현의 고정값과 같은지 확인."""
    want = dict(sample_rate=SR, n_fft=NFFT, features=NMEL, window_size=0.025, window_stride=0.01, window="hann",
                normalize="NA", log=True, frame_splicing=1, pad_to=0, pad_value=0.0)
    bad = {k: (pc.get(k), v) for k, v in want.items() if k in pc and pc.get(k) != v}
    if bad:
        raise ValueError(f"지원하지 않는 preprocessor 설정: {bad}")


def final_frames(n: int) -> int:
    """n 샘플을 받은 뒤 확정된 프레임 수 (보수 규칙, end 패딩 전)."""
    return 0 if n < PAD else min(n // HOP, (n - PAD) // HOP + 1)


def frames_needed(t: int) -> int:
    """프레임 0..t-1 이 모두 확정되는 최소 샘플 수 (t ≥ 1)."""
    return HOP * (t - 1) + PAD


class LogMel:
    """창/필터뱅크를 들고 있는 프레임 계산기. window400 (400,), fb (1,128,257) 또는 (128,257) — 체크포인트 버퍼."""

    def __init__(self, window400, fb):
        w = np.zeros(NFFT, np.float32)
        off = (NFFT - WIN) // 2                                   # torch.stft: 짧은 창을 n_fft 가운데(56)에 둔다
        w[off: off + WIN] = np.asarray(window400, np.float32).reshape(-1)
        self.w512 = w
        self.fbT = np.ascontiguousarray(np.asarray(fb, np.float32).reshape(NMEL, NFFT // 2 + 1).T)   # (257,128)

    def frames(self, ypad: np.ndarray, t0: int, t1: int, base: int = 0) -> np.ndarray:
        """패딩된 pre-emphasis 신호(ypad[0] = 패딩 인덱스 base)에서 프레임 t0..t1-1 → (t1-t0, 128) fp32."""
        if t1 <= t0:
            return np.zeros((0, NMEL), np.float32)
        idx = (np.arange(t0, t1)[:, None] * HOP - base) + np.arange(NFFT)[None]
        spec = np.fft.rfft(ypad[idx] * self.w512, n=NFFT, axis=-1)                  # complex64
        power = spec.real * spec.real + spec.imag * spec.imag                         # |X|² (mag_power 2)
        return np.log(power @ self.fbT + LOG_GUARD).astype(np.float32, copy=False)


def logmel_full(x: np.ndarray, window400, fb) -> np.ndarray:
    """오프라인 기준: 발화 전체 → (N//160, 128)."""
    x = np.asarray(x, np.float32).reshape(-1)
    y = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]]) if x.size else x
    ypad = np.pad(y, (PAD, PAD))
    return LogMel(window400, fb).frames(ypad, 0, x.size // HOP)


class MelStream:
    """증분 log-mel. push() 는 샘플만 쌓고(계산 없음), take(t1) 가 확정 프레임 [t, t1) 을 계산해 돌려준다.
    finish() 는 끝 0 패딩 256 을 붙여 나머지(0~1개) 프레임까지 확정한다. 한 세그먼트에 하나 (재사용 금지)."""

    def __init__(self, window400=None, fb=None, logmel: LogMel | None = None):
        self.lm = logmel if logmel is not None else LogMel(window400, fb)
        self._buf = [np.zeros(PAD, np.float32)]   # 패딩 인덱스 base 부터의 pre-emphasis 신호 조각들
        self._base = 0                            # self._buf 첫 샘플의 패딩 인덱스
        self._last = None                         # 직전 원 샘플 (pre-emphasis 상태)
        self.n = 0                                # 받은 샘플 수
        self.t = 0                                # 계산해 내보낸 프레임 수
        self.done = False

    def push(self, pcm) -> None:
        assert not self.done, "finish() 뒤에는 push 할 수 없다"
        x = np.asarray(pcm, np.float32).reshape(-1)
        if not x.size:
            return
        y = np.empty_like(x)
        y[1:] = x[1:] - PREEMPH * x[:-1]
        y[0] = x[0] if self._last is None else x[0] - PREEMPH * self._last   # 첫 샘플은 그대로 (NeMo)
        self._last = np.float32(x[-1])
        self._buf.append(y)
        self.n += x.size

    def finish(self) -> None:
        if not self.done:
            self._buf.append(np.zeros(PAD, np.float32))   # center=True 의 끝 0 패딩
            self.done = True

    def ready(self) -> int:
        """지금 계산할 수 있는 프레임 수 (finish 뒤에는 N//160)."""
        return self.n // HOP if self.done else final_frames(self.n)

    def take(self, t1: int) -> np.ndarray:
        """프레임 [t, t1) 계산 (t1 ≤ ready()). 필요 없어진 앞부분 샘플은 버린다."""
        assert t1 <= self.ready(), (t1, self.ready())
        if len(self._buf) > 1:
            self._buf = [np.concatenate(self._buf)]
        out = self.lm.frames(self._buf[0], self.t, t1, self._base)
        self.t = max(self.t, t1)
        cut = self.t * HOP - self._base
        if cut > 0:
            self._buf[0] = self._buf[0][cut:]
            self._base += cut
        return out

    def push_frames(self, pcm) -> np.ndarray:
        """push + 확정된 프레임 전부 계산 (임의 크기 입력용 편의 함수)."""
        self.push(pcm)
        return self.take(self.ready())
