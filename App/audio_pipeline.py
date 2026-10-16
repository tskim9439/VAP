"""입력 오디오 → 세그먼트 명령 (audio-ui-spec §1.3–1.5, §3.2). 순수 numpy + soxr, asyncio 스레드에서 돈다.

흐름: native-rate float32 → soxr HQ 스트리밍 리샘플(16 kHz 면 우회, 비트 동일) → 512 샘플(32 ms) 프레임
→ VAD 특징(1차 80 Hz 고역통과 뒤 프레임 에너지, VAD 경로 전용) → 잡음 바닥 추적 → 세그먼트 상태 기계 → 세그먼트별 고정 이득.
모든 결정은 16 kHz 오디오 시계(시작 후 샘플 수) 위에서 이뤄지므로 1×·4×·최대 속도에서 경계가 같다.

불변식: 받은 샘플은 많아야 한 세그먼트에 속한다(겹침·누락 없음). received = in_segments + skipped + held, stop 뒤 held = 0.
합성 샘플(lead-in 배경, zero tail)은 받은 샘플이 아니며 따로 센다. 강제 절단은 아직 내보내지 않은 프레임에서만 고른다(cut ≥ released).

실제 방 소음 대응 (spec §1.3–1.5 에서 바꾼 점, F06/F07/F08/F31/F33):
- 특징: rfft 근사 대신 spec 의 1차 IIR 을 프레임 사이 상태를 이어 벡터로 계산 (rfft 는 50 Hz 험이 새어 휴지를 막았다).
- 디지털 무음(4 ms 이상 연속 |x| ≤ 1e-6)이 든 프레임은 바닥·문턱을 갱신하지 않는다 (0 한 프레임이 바닥을 −85 에 묶었다).
- 바닥 둘: 끝·onset 판정용 N = 최근 3 s 의 10 % 분위수(T_off = N + 6, T_on = min(중앙값 + 6, N + 10)),
  유성 수준(L_spk·이득·voiced) 판정용 N_lvl = 기존 최소 추적(+0.5 dB/s).
- SPEECH 의 유성 판정은 최근 5 프레임 중 3 개 이상이고, 마지막 유성 프레임은 4 ms 소블록 에너지 중앙값(D)도 문턱 이상이어야
  한다 (키보드 클릭·누름-뗌 쌍이 휴지를 되돌리지 않게). L_spk 를 알면 soft max 전에도 문턱을 L_spk − 25 dB 이상으로.
  onset 과 '유성'(L_spk·voiced) 도 D 를 함께 봐서 클릭만으로는 세그먼트가 열리거나 유성으로 세지지 않는다.
- 이득: 양(+)의 이득은 말소리 근거(유성 이력과 바닥 대비 15 dB)가 있을 때만, 지금 onset 이 이미 크면 올리지 않는다
  (번갈아 말하는 두 화자). 이력 없는 첫 세그먼트는 onset 이 아주 작을 때만 올림 후보가 되고, 그때는 시작을 최대 1 s
  미뤄(유성 16 프레임) 그 구간으로 정한다 — 숨소리·페이드인 3–4 프레임으로 +20 dB 를 주던 문제.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

SR = 16000
FRAME = 512                      # 32 ms
FRAME_S = FRAME / SR
INIT_FRAMES = 16                 # 잡음 바닥 초기화(0.5 s) — 인과적: 이 프레임들은 바닥이 정해질 때까지 보류
HIST_FRAMES = 96                 # lead-in·피크용 최근 3 s(+여유) 원음 보관
NONE_MAX_S = 30.0                # segmentation "none" 최대 입력 길이
TAIL_FLOOR = 3200                # zero tail 최소 200 ms [critic]
LEAD_IN = 5120                   # lead-in 길이 320 ms
# VAD 상수 (§1.3)
ON_DB, OFF_DB, ABS_ON, ABS_OFF = 10.0, 6.0, -60.0, -64.0
FLOOR_LO, FLOOR_HI, RISE = -85.0, -35.0, 0.5 * FRAME_S       # 수준 바닥 상승 0.5 dB/s
FLOOR_WIN, FLOOR_Q = 94, 10.0    # 끝 판정 바닥 = 최근 3 s(디지털 무음 제외) E 의 10 % 분위수
ONSET_Q, ONSET_DB = 50.0, 6.0    # onset 문턱 = min(같은 창의 중앙값 + 6, N + 10) dB
ZERO_EPS, ZERO_RUN = 1e-6, 64    # 디지털 무음: |x| ≤ 1e-6 이 64 샘플(4 ms) 이상 연속
REL_DB = 15.0                    # soft max 이후 휴지 = L_spk − 15 dB 미만
REL_PRE_DB = 25.0                # soft max 이전에도 L_spk − 25 dB 미만은 휴지 (L_spk 를 안 뒤)
VOICE_K, VOICE_N = 3, 5          # SPEECH 유성 판정: 최근 5 프레임 중 3 개 이상이 문턱 이상
SUBS = 8                         # 충격음 판별: 프레임을 4 ms 소블록 8 개로 나눠 에너지 중앙값(D)을 본다
MIN_VO = 30                      # L_spk 를 믿기 위한 최소 유성 프레임 수 (≈ 1 s)
VO_KEEP = 150                    # L_spk = 최근 150 개 유성 프레임의 중앙값
LIMIT = 0.95                     # 안전 소프트 리미터 시작점
PEAK_CAP_DB = -1.0
GAIN_DEADBAND, GAIN_MIN, GAIN_MAX, GAIN_FIRST_MAX = 6.0, -10.0, 30.0, 20.0
SPEECH_SNR_DB = 15.0             # 양의 이득 근거: 말소리 수준이 조용한 때의 바닥보다 15 dB 이상
FIRST_QUIET_DB = -45.0           # 첫 세그먼트: 가장 큰 onset 프레임도 이보다 작을 때만 올림 후보
DEFER_VOICED, DEFER_FR = 16, 31  # 올림 후보인 첫 세그먼트는 유성 16 프레임 또는 onset 뒤 1 s 까지 시작을 미뤄 그 구간으로 이득을 정한다
HP_A = math.exp(-2 * math.pi * 80 / SR)                       # 1차 80 Hz 고역통과 계수 (seg_proto.hp)
_CONT = ("soft_max", "max_len", "kv_guard")                    # 말이 이어지는 중의 절단 (조용한 때가 아님)


class ParamError(ValueError):
    """잘못된 세그먼트 설정 (서버는 error bad_config 로 바꾼다)."""


class InputTooLong(ValueError):
    """segmentation "none" 에 30 s 를 넘는 입력."""


def min_tail_s(delay: int) -> float:
    """δ 에서 허용하는 최소 tail_s = max(0.5, 0.08·(δ+2)) [critic]."""
    return max(0.5, 0.08 * (delay + 2))


@dataclass
class SegParams:  # 기본값 = audio-ui-spec §1.4/§1.5 ([critic] 반영)
    segmentation: str = "vad"        # "vad" | "none"
    pre_roll_ms: int = 480
    end_pause_ms: int = 800
    soft_max_s: float | None = None  # None ⇒ max(hard_max − 7, 0.6·hard_max) [critic] (기본 18)
    end_pause_soft_ms: int = 256
    hold_from_s: float | None = None # None ⇒ hard_max − hold − 2 프레임 (기본 24.296): 절단 창 640 ms 에 필요한 만큼만 늦춘다
    hold_ms: int = 640
    hard_max_s: float = 25.0
    tail_s: float = 1.0
    lead_in: bool = False            # [critic] 기본 꺼짐 (eval 은 앞 무음 없이 쟀다)
    gain: str = "auto"               # "auto" | "off"
    target_db: float = -26.0

    def resolved(self) -> "SegParams":
        """파생값을 채운 복사본."""
        h = float(self.hard_max_s)
        soft = self.soft_max_s if self.soft_max_s is not None else max(h - 7.0, 0.6 * h)
        # hold 는 강제 절단 창(아직 안 보낸 hold_ms)을 만들 때만 필요 — 그보다 일찍 잡으면 지연만 늘어난다 (F33)
        hold = self.hold_from_s if self.hold_from_s is not None else round(h - self.hold_ms / 1000.0 - 2 * FRAME_S, 6)
        return replace(self, soft_max_s=float(soft), hold_from_s=float(hold))

    def check(self, delay: int | None = None) -> "SegParams":
        """범위 검사(한국어 메시지의 ParamError). delay 를 주면 tail_s ≥ 0.08·(δ+2) 도 본다. 해석된 복사본을 돌려준다."""
        p = self.resolved()
        def need(ok, msg):
            if not ok: raise ParamError(msg)
        need(p.segmentation in ("vad", "none"), f"분할 방식은 vad 또는 none 이어야 합니다: {p.segmentation!r}")
        need(p.gain in ("auto", "off"), f"음량 맞춤은 auto 또는 off 여야 합니다: {p.gain!r}")
        need(160 <= p.pre_roll_ms <= 960, f"pre_roll_ms 는 160–960 ms 여야 합니다: {p.pre_roll_ms}")
        need(300 <= p.end_pause_ms <= 2000, f"끝 멈춤(end_pause_ms)은 300–2000 ms 여야 합니다: {p.end_pause_ms}")
        need(64 <= p.end_pause_soft_ms <= 2000, f"end_pause_soft_ms 는 64–2000 ms 여야 합니다: {p.end_pause_soft_ms}")
        need(10.0 <= p.hard_max_s <= 30.0, f"최대 길이(hard_max_s)는 10–30 s 여야 합니다: {p.hard_max_s}")
        need(0.0 < p.soft_max_s <= p.hard_max_s, f"soft_max_s 는 0 초과 hard_max_s 이하여야 합니다: {p.soft_max_s}")
        need(96 <= p.hold_ms <= 2000, f"hold_ms 는 96–2000 ms 여야 합니다: {p.hold_ms}")
        need(0.0 < p.hold_from_s <= p.hard_max_s - p.hold_ms / 1000.0 + 1e-9,
             f"hold_from_s 는 0 초과 hard_max_s − hold_ms 이하여야 합니다(절단 창 확보): {p.hold_from_s}")
        lo = 0.5 if delay is None else min_tail_s(int(delay))
        need(lo - 1e-9 <= p.tail_s <= 1.5, f"tail_s 는 {lo:.2f}–1.5 s 여야 합니다(δ 에 따라 최소 0.08·(δ+2) s): {p.tail_s}")
        need(-40.0 <= p.target_db <= -12.0, f"target_db 는 −40…−12 dBFS 여야 합니다: {p.target_db}")
        return p


@dataclass
class Cmd:
    kind: str                        # "seg_start" | "audio" | "seg_end"
    seg: int
    pcm: "np.ndarray | None" = None  # float32 16 kHz, 이득 적용됨 (seg_start 는 pre-roll/lead-in 묶음을 실을 수 있다)
    info: dict = field(default_factory=dict)
    # seg_start.info: t_start, reason, pre_roll_ms, lead_in_ms, gain_db (+ s_start, lead_in_samples, onset_t)
    # audio.info:     s0, s1 (세션 시계 샘플 구간), hold (hold-back 중 내보낸 오디오면 True)
    # seg_end.info:   t_end, reason, zero_tail_samples, voiced_s (+ zero_tail_ms, audio_s, s_end)


_K = np.arange(FRAME, dtype=np.float64)
_HP_UP = HP_A ** (_K + 1.0)                                    # a^(n+1)
_HP_DN = HP_A ** (-_K)                                         # a^(−j) (최대 ≈ 9.4e6, float64 에서 오차 ~1e-14 dB)


class HighPassDb:
    """VAD 특징: 1차 80 Hz 고역통과 y[n] = a·(y[n−1] + x[n] − x[n−1]) (spec §1.3, seg_proto.hp) → 프레임 dBFS.
    프레임 사이 필터 상태를 이어 가므로 연속된 프레임을 순서대로 넣어야 한다 (파이프라인마다 하나).
    프레임 안은 누적합으로 벡터화: y[n] = a^(n+1)·(Σ_{j≤n} d[j]·a^(−j) + y_prev), d = x 의 1차 차분.
    pair() 는 프레임 에너지 E 와 함께 4 ms 소블록 8 개 에너지의 중앙값 D 도 준다 — 키보드 클릭(≈ 6 ms) 같은 충격음은
    E 만 올리고 D 는 배경 수준이라, D 로 '말소리다운' 프레임을 가린다."""

    def __init__(self):
        self.px = 0.0; self.py = 0.0                           # 직전 입력·출력 샘플

    def pair(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(frames, np.float64).reshape(-1, FRAME)
        n = len(X)
        if not n: return np.zeros(0), np.zeros(0)
        flat = X.reshape(-1)
        D = X - np.concatenate(([self.px], flat[:-1])).reshape(n, FRAME)
        Z = _HP_UP * np.cumsum(D * _HP_DN, axis=1)               # 영상태 응답
        out = np.empty(n); sub = np.empty(n); py = self.py
        for i in range(n):                                       # 이전 출력의 감쇠만 프레임 순서로 더한다
            y = Z[i] + _HP_UP * py; py = float(y[-1]); p = y * y
            out[i] = float(p.sum()) / FRAME; sub[i] = float(np.median(p.reshape(SUBS, -1).mean(1)))
        self.px = float(flat[-1]); self.py = py
        return 10.0 * np.log10(out + 1e-10), 10.0 * np.log10(sub + 1e-10)

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return self.pair(frames)[0]


def frame_db(frames: np.ndarray) -> np.ndarray:
    """(n, 512) 연속 프레임 → VAD 에너지 dBFS (상태 0 에서 시작하는 HighPassDb 한 번)."""
    return HighPassDb()(frames)


def digital_silence(frames: np.ndarray) -> np.ndarray:
    """(n, 512) → bool (n,): |x| ≤ 1e-6 이 64 샘플 이상 연속인 프레임 (뮤트·0 패딩·soxr 잔향 1e-9 포함)."""
    z = np.abs(np.asarray(frames).reshape(-1, FRAME)) <= ZERO_EPS
    c = np.concatenate([np.zeros((len(z), 1), np.int32), np.cumsum(z, axis=1, dtype=np.int32)], axis=1)
    return ((c[:, ZERO_RUN:] - c[:, :-ZERO_RUN]) >= ZERO_RUN).any(axis=1)


def _quantile(s: list, q: float) -> float:
    """정렬된 목록의 분위수 — np.percentile 기본(linear)과 비트 동일 (numpy _lerp 의 두 갈래 그대로)."""
    pos = q / 100.0 * (len(s) - 1); i = int(pos); f = pos - i
    if i + 1 >= len(s): return s[i]
    d = s[i + 1] - s[i]
    return s[i + 1] - d * (1.0 - f) if f >= 0.5 else s[i] + d * f


class Resampler:
    """soxr HQ 스트리밍 리샘플러. 16 kHz 입력은 우회(비트 동일). flush() 는 한 번만(last=True 뒤 객체는 소모됨)."""

    def __init__(self, in_rate: int):
        self.in_rate = int(in_rate)
        if not 8000 <= self.in_rate <= 192000: raise ParamError(f"입력 표본율은 8000–192000 Hz 여야 합니다: {in_rate}")
        self.bypass = self.in_rate == SR
        self.name = "bypass" if self.bypass else "soxr-HQ"
        self.n_in = 0; self.n_out = 0; self.done = False
        self._rs = None
        if not self.bypass:
            import soxr
            self._rs = soxr.ResampleStream(self.in_rate, SR, 1, dtype="float32", quality="HQ")

    def push(self, x: np.ndarray) -> np.ndarray:
        if self.done: raise RuntimeError("resampler 는 이미 flush 되었습니다")
        self.n_in += len(x)
        y = x if self.bypass else self._rs.resample_chunk(x, last=False)
        self.n_out += len(y)
        return y

    def flush(self) -> np.ndarray:
        if self.done: return np.zeros(0, np.float32)
        self.done = True
        if self.bypass: return np.zeros(0, np.float32)
        y = self._rs.resample_chunk(np.zeros(0, np.float32), last=True)
        self.n_out += len(y)
        return np.asarray(y, np.float32)


def _as_mono_f32(pcm) -> np.ndarray:
    x = np.asarray(pcm)
    if x.ndim == 2: x = x.mean(axis=1)
    if x.dtype == np.int16: x = x.astype(np.float32) / 32768.0
    x = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
    if not np.isfinite(x).all(): x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


class AudioPipeline:
    """세션 하나의 리샘플·VAD·분할·이득. push/stop/force_end 는 Cmd 목록을 돌려준다(작업자 큐에 순서대로 넣는다)."""

    def __init__(self, in_rate: int, params: SegParams | None = None,
                 feature_fn: Callable[[np.ndarray], np.ndarray] | None = None):
        self._raw = params or SegParams()
        self.params = self._raw.check()
        self.none = self.params.segmentation == "none"
        self.res = Resampler(in_rate)
        self.in_rate = self.res.in_rate
        self.resampler = self.res.name
        self._feat = feature_fn or HighPassDb()  # 상태 있는 IIR (테스트에서 다른 특징을 넣을 수 있다)
        self._pending: dict = {}                 # update_params: 다음 세그먼트부터
        self._set_frames()
        # 원음 버퍼 (16 kHz): buf[0] = 세션 샘플 b0
        self._buf = np.zeros(SR * 8, np.float32); self._b0 = 0; self._bn = 0
        # 프레임 이력: E·D(소블록 중앙값)·유성(on)·디지털 무음 플래그, 프레임 h0 부터
        self._eh: list[float] = []; self._dh: list[float] = []; self._onh: list[bool] = []; self._zh: list[bool] = []; self._h0 = 0
        self._te = 0                             # E 를 계산한 프레임 수
        self._t = 0                              # 상태 기계를 거친 프레임 수
        self._partial_done = False
        # 바닥: N(끝·onset, 분위수) / N_lvl(유성 수준, 최소 추적) / N_sil(조용한 때의 N, 이득 근거용)
        self._N: float | None = None; self._nlvl: float | None = None; self._n_sil: float | None = None
        self._win: deque = deque(); self._ws: list[float] = []  # 바닥 창 (도착 순, 정렬)
        self._ton = ABS_ON; self._toff = ABS_OFF; self._ton_lvl = ABS_ON; self._lvl = -100.0
        self._vo: list[float] = []; self._lspk: float | None = None
        # 세그먼트 상태
        self._open = False; self._seg = 0; self._start = 0; self._last_v = -1
        self._prev_end = 0; self._prev_reason = ""; self._voiced = 0
        self._gain_db = 0.0; self._g = 1.0; self._want = 0; self._hold = False
        self._released = 0; self._defer = None         # _defer: 시작을 미룬 첫 세그먼트 (연 프레임, onset, 이유)
        # 계수
        self._n_seg = 0; self._n_skip = 0; self._n_lead = 0; self._n_zero = 0; self._limited = 0
        self._ignored = 0; self.stopped = False
        self._out: list[Cmd] = []

    # ---------- 설정 ----------
    def _set_frames(self):
        p = self.params
        fr = lambda ms: max(1, int(round(ms / (FRAME_S * 1000))))
        self._pre_fr = fr(p.pre_roll_ms); self._ep = fr(p.end_pause_ms); self._ep_soft = fr(p.end_pause_soft_ms)
        self._hold_fr = fr(p.hold_ms); self._hold_n = self._hold_fr * FRAME
        self._hold_from_fr = int(p.hold_from_s * SR / FRAME)          # 원형과 같은 내림
        self._tail_n = int(round(p.tail_s * SR))

    def update_params(self, **kw) -> None:
        """end_pause_ms, gain 등 — 다음 세그먼트 시작부터 적용(열린 세그먼트에는 적용 안 함, §4.2 update)."""
        new = replace(self._raw, **{**self._pending, **kw}).check()
        if new.segmentation != self.params.segmentation: raise ParamError("분할 방식은 세션 중에 바꿀 수 없습니다")
        self._pending = {**self._pending, **kw}

    def _apply_pending(self):
        if self._pending:
            self._raw = replace(self._raw, **self._pending); self.params = self._raw.check(); self._pending = {}; self._set_frames()

    # ---------- 버퍼 ----------
    @property
    def n_rx(self) -> int:
        return self._b0 + self._bn

    def _append(self, y: np.ndarray):
        if not len(y): return
        if self._bn + len(y) > len(self._buf):
            keep = min(self._released, self._t * FRAME - HIST_FRAMES * FRAME, self._te * FRAME)
            drop = max(0, keep - self._b0)
            if drop:
                self._buf[:self._bn - drop] = self._buf[drop:self._bn]; self._b0 += drop; self._bn -= drop
            if self._bn + len(y) > len(self._buf):
                nb = np.zeros(max(2 * len(self._buf), self._bn + len(y)), np.float32); nb[:self._bn] = self._buf[:self._bn]; self._buf = nb
        self._buf[self._bn:self._bn + len(y)] = y; self._bn += len(y)

    def _get(self, a: int, b: int) -> np.ndarray:
        assert a >= self._b0, (a, self._b0)
        return self._buf[a - self._b0:b - self._b0]

    def _fe(self, t: int) -> int:
        """프레임 t 의 끝 샘플 (stop 때 패딩된 마지막 부분 프레임은 n_rx)."""
        return min((t + 1) * FRAME, self.n_rx)

    def _E(self, t: int) -> float:
        return self._eh[t - self._h0]

    def _D(self, t: int) -> float:
        return self._dh[t - self._h0]

    def _energies(self, fr: np.ndarray) -> tuple:
        if hasattr(self._feat, "pair"): return self._feat.pair(fr)
        e = self._feat(fr); return e, e                   # 테스트용 특징 함수: D = E

    def _trim_hist(self):
        k = (self._t - 200) - self._h0                    # 처리 지점 기준 200 프레임(6.4 s)만 남긴다
        if k > 0: del self._eh[:k]; del self._dh[:k]; del self._onh[:k]; del self._zh[:k]; self._h0 += k

    # ---------- 이득 ----------
    def _gain(self, x: np.ndarray) -> np.ndarray:
        if self._g == 1.0: return np.array(x, np.float32, copy=True)
        y = (x * np.float32(self._g)).astype(np.float32)
        a = np.abs(y); m = a > LIMIT
        if m.any():
            self._limited += int(m.sum())
            y[m] = np.sign(y[m]) * (LIMIT + (1 - LIMIT) * np.tanh((a[m] - LIMIT) / (1 - LIMIT)))
        return y

    def speech_evidence(self) -> bool:
        """말소리 근거: 유성 이력 ≥ 30 프레임이고 L_spk 가 조용한 때의 바닥보다 15 dB 이상 (양의 이득·'최대 이득' 경고용)."""
        return self._lspk is not None and self._n_sil is not None and self._lspk - self._n_sil >= SPEECH_SNR_DB

    def _seg_gain(self, t: int, onset: int, settle: bool = False) -> float:
        """세그먼트 이득(dB), 세그먼트 동안 고정 — §1.5 + 소음·첫 세그먼트·번갈아 말하기 대책 (F31).
        settle: 미뤘던 첫 세그먼트를 onset 부터 프레임 t 까지의 유성 프레임으로 정한다."""
        p = self.params
        if self.none or p.gain != "auto": return 0.0
        vals = [self._E(k) for k in range(max(onset, self._h0), t + 1) if self._onh[k - self._h0]]
        l_on = max(vals) if vals else None               # onset(또는 미룬 구간)의 가장 큰 유성 프레임
        first = settle or self._lspk is None
        if not first:                                     # 이력: 최근 유성 프레임 중앙값
            g = p.target_db - self._lspk
            if g > 0 and not self.speech_evidence(): g = 0.0          # 소음만 있던 이력은 올리지 않는다
        else:                                             # 이력 없음: onset 창은 말의 약한 앞부분(숨·페이드인)일 수 있다
            if not vals: return 0.0
            med = float(np.median(vals)); g = p.target_db - med
            if g > 0 and (self._n_sil is None or med - self._n_sil < SPEECH_SNR_DB): g = 0.0
            if g > 0 and not settle and l_on > FIRST_QUIET_DB: g = 0.0
            if g > 0 and settle and len(vals) < DEFER_VOICED // 2: g = 0.0          # 유성 프레임이 몇 개 없으면 말소리 근거가 아니다
        if abs(g) <= GAIN_DEADBAND: g = 0.0
        g = min(max(g, GAIN_MIN), GAIN_MAX)
        if first: g = min(g, GAIN_FIRST_MAX)
        if g > 0 and l_on is not None:                    # 지금 화자의 onset 이 이미 크면 앞 화자 이력으로 올리지 않는다
            g = max(0.0, min(g, p.target_db + GAIN_DEADBAND - l_on))
        if g > 0:                                         # 최근 3 s 피크 × 이득 ≤ −1 dBFS
            e = self._fe(t); seg = self._get(max(self._b0, e - 3 * SR), e)
            pk = float(np.abs(seg).max()) if len(seg) else 0.0
            if pk > 0: g = max(0.0, min(g, PEAK_CAP_DB - 20 * math.log10(pk)))
        return math.trunc(g * 10) / 10                    # 0.1 dB 단위, 0 쪽으로 (보고값 = 적용값, cap 유지)

    def _lead_in(self, t: int) -> np.ndarray:
        """최근 3 s 의 가장 조용한 3 프레임 타일 × 0.7 (streams.silence_like 방식), 320 ms. 조건이 안 되면 빈 배열."""
        e = self._fe(t)
        if e < 3 * SR: return np.zeros(0, np.float32)    # 세션 이력 3 s 미만
        ks = [k for k in range(max(self._h0, t + 1 - int(3 * SR / FRAME)), t + 1) if k * FRAME >= self._b0]
        if len(ks) < 3: return np.zeros(0, np.float32)
        q = sorted(ks, key=self._E)[:3]
        if any(self._E(k) >= self._ton for k in q): return np.zeros(0, np.float32)
        edge = np.concatenate([self._get(k * FRAME, (k + 1) * FRAME) for k in sorted(q)])
        rng = np.random.RandomState(self._seg)
        reps = LEAD_IN // len(edge) + 2
        tiles = [edge if rng.rand() < 0.5 else edge[::-1] for _ in range(reps)]
        tiles = [x * (1 if rng.rand() < 0.5 else -1) for x in tiles]
        return (np.concatenate(tiles)[:LEAD_IN] * 0.7).astype(np.float32)

    # ---------- 방출 ----------
    def _emit_audio(self, to: int):
        """released → to 를 열린 세그먼트로 내보내거나(닫혀 있으면) 건너뜀으로 센다."""
        to = min(to, self.n_rx)
        if to <= self._released: return
        a = self._released
        if self._open:
            self._out.append(Cmd("audio", self._seg, self._gain(self._get(a, to)), dict(s0=a, s1=to, hold=self._hold)))
            self._n_seg += to - a
        else:
            self._n_skip += to - a
        self._released = to

    def _open_seg(self, t: int, start: int, onset: int, reason: str):
        assert start >= self._released, (start, self._released)
        self._emit_audio(start)                           # 그 앞은 건너뜀 (닫힌 상태)
        self._apply_pending()
        self._seg += 1; self._open = True; self._start = start; self._last_v = t; self._hold = False
        sf = start // FRAME
        self._voiced = sum(1 for k in range(max(sf, self._h0), t + 1) if self._onh[k - self._h0]) if not self.none else 0
        self._gain_db = self._seg_gain(t, onset)
        if self._gain_db > 0 and self._lspk is None:      # 이력 없이 올리려면: 시작 명령·오디오를 잡아 두고 근거를 더 모은다
            self._defer = (t, onset, reason); self._gain_db = 0.0; self._g = 1.0; self._want = start
            return
        self._start_cmd(t, t, onset, reason)

    def _settle(self, t: int):
        """미룬 첫 세그먼트를 시작한다: onset 부터 프레임 t 까지의 유성 프레임으로 이득을 정하고 잡아 둔 오디오를 한 묶음으로 낸다."""
        t0, onset, reason = self._defer; self._defer = None
        self._gain_db = self._seg_gain(t, onset, settle=True)
        self._start_cmd(t, t0, onset, reason)

    def _start_cmd(self, t: int, t0: int, onset: int, reason: str):
        """seg_start: 세그먼트 시작 → 프레임 t 끝까지 한 묶음 (이득 적용). t0 = 세그먼트를 연 프레임 (lead-in 판단용)."""
        start = self._start
        self._g = float(np.float32(10 ** (self._gain_db / 20)))
        onset_s = max(onset * FRAME, start)
        pre_ms = (onset_s - start) * 1000.0 / SR
        lead = np.zeros(0, np.float32)
        if self.params.lead_in and not self.none and pre_ms < 160: lead = self._lead_in(t0)
        end = self._fe(t) if not self.none else self.n_rx
        burst = self._get(start, end)
        pcm = self._gain(np.concatenate([lead, burst]) if len(lead) else burst)
        self._n_lead += len(lead); self._n_seg += end - start
        self._released = end; self._want = max(self._want, end)
        self._out.append(Cmd("seg_start", self._seg, pcm, dict(
            t_start=round(start / SR, 3), reason=reason, pre_roll_ms=int(round(pre_ms)), lead_in_ms=int(round(len(lead) * 1000 / SR)),
            gain_db=self._gain_db, s_start=start, lead_in_samples=len(lead), onset_t=round(onset_s / SR, 3),
            forced_start=reason in ("forced", "kv"))))

    def _end_seg(self, end: int, reason: str, trailing: int, t: int | None = None):
        if self._defer is not None: self._settle(self._t - 1 if t is None else t)   # 미룬 시작이 있으면 먼저 낸다
        assert end >= self._released, (end, self._released)
        self._emit_audio(end)
        if self.none: z = SR                              # eval 프로토콜: 정확히 16000 개의 0
        elif reason in ("max_len", "kv_guard"): z = self._tail_n
        else: z = max(TAIL_FLOOR, self._tail_n - max(0, trailing))
        self._n_zero += z
        self._out.append(Cmd("seg_end", self._seg, None, dict(
            t_end=round(end / SR, 3), reason=reason, zero_tail_samples=int(z), zero_tail_ms=int(round(z * 1000 / SR)),
            voiced_s=round(self._voiced * FRAME_S, 3), audio_s=round((end - self._start) / SR, 3), s_end=end)))
        self._open = False; self._prev_end = end; self._prev_reason = reason; self._want = end; self._hold = False

    # ---------- 프레임 처리 ----------
    def _features(self, final: bool = False):
        """새로 완성된 프레임의 E·디지털 무음 플래그를 계산하고, 바닥이 정해졌으면 상태 기계에 순서대로 넣는다."""
        n_full = self.n_rx // FRAME
        if n_full > self._te:
            fr = self._get(self._te * FRAME, n_full * FRAME).reshape(-1, FRAME)
            e, d = self._energies(fr)
            self._eh.extend(float(v) for v in e); self._dh.extend(float(v) for v in d); self._onh.extend([False] * len(fr))
            self._zh.extend(bool(v) for v in digital_silence(fr)); self._te = n_full
        if final and not self._partial_done and self.n_rx > self._te * FRAME:   # 마지막 부분 프레임: VAD 만 0 패딩
            x = np.zeros((1, FRAME), np.float32); r = self._get(self._te * FRAME, self.n_rx); x[0, :len(r)] = r
            e, d = self._energies(x); self._eh.append(float(e[0])); self._dh.append(float(d[0])); self._onh.append(False)
            self._zh.append(True)                         # 0 패딩이라 바닥 정보로 쓰지 않는다
            self._te += 1; self._partial_done = True
        if self._N is None and not self._calibrate(final):
            return
        self._run(self._te)

    def _run(self, upto: int):
        while self._t < upto:
            self._step(self._t); self._t += 1
            if self._t - self._h0 > 300: self._trim_hist()

    def _calibrate(self, final: bool) -> bool:
        """바닥 초기화(인과적). 앞쪽 디지털 무음 프레임은 바닥 정보가 없으니 ABS 문턱으로 먼저 흘려보내고(아무것도 열 수 없다),
        첫 비무음 프레임부터 16 개가 모일 때까지 보류한다 (무음·비무음이 섞여도 보류는 3 s 이하)."""
        while self._t < self._te and self._zh[self._t - self._h0]:
            self._run(self._t + 1)
        ks = [k for k in range(self._t, min(self._te, self._t + FLOOR_WIN)) if not self._zh[k - self._h0]]   # 패킷 크기와 무관
        if len(ks) < INIT_FRAMES and not final and self._te - self._t < FLOOR_WIN:
            return False
        vals = sorted(self._E(k) for k in ks[:INIT_FRAMES])
        if vals:
            self._N = float(np.clip(_quantile(vals, FLOOR_Q), FLOOR_LO, FLOOR_HI))
            self._nlvl = float(np.clip(vals[0], FLOOR_LO, FLOOR_HI))
            self._ton = self._onset_thr(_quantile(vals, ONSET_Q))
        else:                                             # 디지털 무음뿐: 정보 없음 → ABS 문턱
            self._N = self._nlvl = FLOOR_LO; self._ton = ABS_ON
        self._toff = max(self._N + OFF_DB, ABS_OFF); self._ton_lvl = max(self._nlvl + ON_DB, ABS_ON)
        return True

    def _onset_thr(self, med: float) -> float:
        """T_on = min(중앙값 + 6, N + 10): 요동 소음에선 중앙값 기준(민감), 말이 끝난 직후엔 창 중앙값이 아직 말소리라
        10 % 분위수 기준으로 (0.3 s 면 소음으로 돌아온다 — 큰 화자 뒤 작은 화자를 놓치지 않게)."""
        return max(min(med + ONSET_DB, self._N + ON_DB, FLOOR_HI + ONSET_DB), ABS_ON)

    def _floor(self, e: float):
        """디지털 무음이 아닌 프레임으로 바닥·문턱을 갱신한다."""
        self._nlvl = float(np.clip(min(e, self._nlvl + RISE), FLOOR_LO, FLOOR_HI))
        self._win.append(e); bisect.insort(self._ws, e)
        if len(self._win) > FLOOR_WIN:
            del self._ws[bisect.bisect_left(self._ws, self._win.popleft())]
        if len(self._win) >= INIT_FRAMES:                 # 그 전에는 초기화 값 유지 (같은 프레임들이라 이어진다)
            self._N = float(np.clip(_quantile(self._ws, FLOOR_Q), FLOOR_LO, FLOOR_HI))
            self._ton = self._onset_thr(_quantile(self._ws, ONSET_Q))
        self._toff = max(self._N + OFF_DB, ABS_OFF); self._ton_lvl = max(self._nlvl + ON_DB, ABS_ON)
        if not self._open and (self._prev_reason not in _CONT or self._t * FRAME - self._prev_end >= self._ep * FRAME):
            self._n_sil = self._N                         # 조용한 때의 바닥 (말 중간 절단 직후의 높아진 N 은 쓰지 않는다)

    def _step(self, t: int):
        e = self._E(t); self._lvl = e
        if self._N is not None and not self._zh[t - self._h0]: self._floor(e)   # 디지털 무음 프레임: 바닥·문턱 고정
        ton, toff = self._ton, self._toff
        on = min(e, self._D(t)) >= self._ton_lvl; self._onh[t - self._h0] = on   # 유성 = 평균·소블록 중앙값 모두 (클릭 제외)
        if self.none:
            if on and self._open: self._voiced += 1
            return
        p = self.params
        if not self._open:                                # SILENCE
            w = [self._E(k) for k in range(max(0, t - 3), t + 1)]         # 4 프레임 중 3 개 ≥ T_on, 그중 하나는 소블록 중앙값도 (클릭 제외)
            if len(w) >= 3 and sum(v >= ton for v in w) >= 3 and any(self._D(k) >= ton for k in range(max(0, t - 3), t + 1)):
                onset = t - len(w) + 1 + next(i for i, v in enumerate(w) if v >= ton)
                start = max((onset - self._pre_fr) * FRAME, self._prev_end, 0, self._released)
                reason = "soft" if (self._prev_reason == "soft_max" and start == self._prev_end) else "vad"
                self._open_seg(t, start, onset, reason)
            else:                                         # 앞으로 어떤 onset 도 닿지 못할 프레임은 건너뜀
                self._emit_audio(max(0, t - 2 - self._pre_fr) * FRAME)
            return
        # SPEECH
        fe = self._fe(t); L = (fe - self._start) / SR
        if on:
            self._vo.append(e); self._voiced += 1
            if len(self._vo) > VO_KEEP: del self._vo[0]
            self._lspk = float(np.median(self._vo)) if len(self._vo) >= MIN_VO else None
        soft = L >= p.soft_max_s
        if self._defer is not None and (self._voiced >= DEFER_VOICED or t - self._defer[1] >= DEFER_FR): self._settle(t)
        if soft: thr = max(toff, (float(np.median(self._vo)) if self._vo else self._ton_lvl) - REL_DB)
        elif self._lspk is not None: thr = max(toff, self._lspk - REL_PRE_DB)
        else: thr = toff
        hits = [k for k in range(max(self._h0, t - VOICE_N + 1), t + 1) if self._E(k) >= thr]
        if len(hits) >= VOICE_K:                          # 최근 5 프레임 중 3 개 이상 (키 누름·뗌 쌍은 못 넘는다)
            good = [k for k in hits if self._D(k) >= thr] # 그중 소블록 중앙값도 문턱 이상인 프레임만 '마지막 유성'이 된다 (클릭 제외)
            if good: self._last_v = max(self._last_v, good[-1])
        # soft max 의 짧은 휴지(256 ms)는 말이 다시 시작되는 프레임(E·D 모두 문턱 이상)이 최근 2 프레임에 있으면 끝내지 않는다
        # — 5 중 3 확인을 기다리는 동안 단어 첫머리를 자르지 않게 (클릭은 D 가 낮아 미루지 못한다)
        fresh = soft and any(self._E(k) >= thr and self._D(k) >= thr for k in range(max(self._h0, t - 1), t + 1))
        if t - self._last_v >= (self._ep_soft if soft else self._ep) and not fresh:
            self._end_seg(fe, "soft_max" if soft else "pause", fe - self._fe(self._last_v), t); return
        if L >= p.hard_max_s:                             # 강제 절단: 아직 내보내지 않은 프레임 중 가장 조용한 곳
            sf = self._start // FRAME
            lo = max(sf + self._hold_from_fr, t + 1 - self._hold_fr, -(-self._released // FRAME), self._h0)
            w = np.array([self._E(k) for k in range(lo, t + 1)])
            m = np.convolve(w, np.ones(3) / 3, "same") if len(w) >= 3 else w
            cut = lo + int(np.argmin(m)); cs = cut * FRAME
            assert cs >= self._released, (cs, self._released)
            moved = sum(1 for k in range(cut, t + 1) if self._onh[k - self._h0])
            self._voiced -= moved
            self._end_seg(cs, "max_len", 0)
            self._open_seg(t, cs, cut, "forced")
            return
        if L >= p.hold_from_s:
            self._hold = True; self._want = max(self._want, fe - self._hold_n)
        else:
            self._want = max(self._want, fe)

    # ---------- 공개 API ----------
    def push(self, pcm) -> list[Cmd]:
        """native-rate float32 mono (int16·2 채널도 받는다) → Cmd 목록."""
        if self.stopped:
            self._ignored += len(np.asarray(pcm).reshape(-1)); return []
        y = self.res.push(_as_mono_f32(pcm))
        self._out = []
        if self.none and self.n_rx + len(y) > NONE_MAX_S * SR:
            raise InputTooLong(f"분할 없음(평가 방식)은 {NONE_MAX_S:.0f} s 이하 입력만 받습니다")
        self._append(y)
        if self.none: self._none_release()
        self._features()
        if self._open and not self.none and self._defer is None: self._emit_audio(self._want)
        return self._out

    def _none_release(self):
        if not self._open:
            if self.n_rx > self._released: self._open_seg(max(self._t - 1, 0), self._released, self._released // FRAME, "none")
        else:
            self._emit_audio(self.n_rx)

    def stop(self, reason: str = "stop") -> list[Cmd]:
        """soxr 를 last=True 로 비우고, 남은 프레임을 처리하고, 열린 세그먼트를 끝낸다. 이후 held = 0."""
        if self.stopped: return []
        self._out = []
        y = self.res.flush()
        if self.none and self.n_rx + len(y) > NONE_MAX_S * SR:
            raise InputTooLong(f"분할 없음(평가 방식)은 {NONE_MAX_S:.0f} s 이하 입력만 받습니다")
        self._append(y)
        if self.none: self._none_release()
        self._features(final=True)
        if self._open:
            trailing = 0 if self.none else self.n_rx - self._fe(self._last_v)
            self._end_seg(self.n_rx, reason, trailing)
        else:
            self._emit_audio(self.n_rx)
        self.stopped = True
        assert self._released == self.n_rx
        return self._out

    def force_end(self, reason: str = "kv_guard", seg: int | None = None, restart: bool = True) -> list[Cmd]:
        """작업자 요청(예: kv_guard)으로 열린 세그먼트를 지금 끝낸다. seg 가 현재 세그먼트가 아니면 무시.
        restart: 같은 지점에서 다음 세그먼트를 바로 연다(reason "kv") — 말이 이어지는 중이어도 샘플을 잃지 않는다."""
        if self.stopped or not self._open or (seg is not None and seg != self._seg): return []
        self._out = []
        if self.none:
            self._end_seg(self.n_rx, reason, 0)            # 다음 push 에서 새 "none" 세그먼트가 열린다
            return self._out
        end = self._t * FRAME                              # 처리한 마지막 프레임 끝 (≥ released)
        self._end_seg(end, reason, 0)
        if restart: self._open_seg(self._t - 1, end, self._t, "kv")
        return self._out

    def vad_snapshot(self) -> dict:
        if self._open and not self.none: st = "speech" if self._last_v == self._t - 1 else "hangover"
        elif self.none: st = "speech" if self._lvl >= self._toff else "silence"
        else: st = "silence"
        r = lambda v: None if v is None else round(float(v), 1)
        return dict(t=round(self._t * FRAME / SR, 3), level_db=r(self._lvl), floor_db=r(self._N), on_db=r(self._ton),
                    off_db=r(self._toff), state=st, gain_db=r(self._gain_db), seg=self._seg if self._open else None,
                    limited=self._limited, speech_db=r(np.median(self._vo)) if self._vo else None, hold=self._hold,
                    calibrating=self._N is None, speech_evidence=self.speech_evidence())

    def counters(self) -> dict:
        held = self.n_rx - self._released
        return dict(audio_rx_s=self.n_rx / SR, audio_seg_s=self._n_seg / SR, audio_skipped_s=self._n_skip / SR,
                    audio_held_s=held / SR, lead_in_s=self._n_lead / SR, zero_tail_s=self._n_zero / SR,
                    limited_samples=self._limited, audio_in_s=self.res.n_in / self.in_rate,
                    samples_rx=self.n_rx, samples_seg=self._n_seg, samples_skipped=self._n_skip, samples_held=held,
                    samples_ignored=self._ignored)
