"""턴 목록 → 세션 연속 음성(타임라인). CST-Bench Natural/L0 조건(겹침 없음)의 기본 구성.

원본에 턴 간 시각이 없는 코퍼스(TAXI)는 턴 사이 간격을 분포에서 뽑아 이어 붙인다. 간격 분포는 이름으로 고른다:
  natural   일반 대화의 턴 교대 간격 — 정규(평균 0.2 s, 표준편차 0.25 s)를 [0.05, 1.0] s 로 자름(Heldner & Edlund 2010 의 0 근처 분포를
            겹침 없이 근사; 이 조건은 겹침을 만들지 않는다)
  mediated  통역을 거치는 대화 — 듣는 사람이 번역을 읽고 답하는 대기를 가정한 균등 [1.0, 3.0] s
같은 화자 턴이 연달아 오면(사이의 garbage 턴을 뺀 경우) 화자 내 쉼으로 보고 균등 [0.3, 0.8] s.
출력: 단일 채널 혼합(mono), 화자별 2 채널(stereo, speakers 순서대로 왼쪽·오른쪽), 턴별 시작·끝 시각(초).
겹침 조건(L1·L2, Early Turn·Interruption)은 이 타임라인 위에 사건 생성기가 시작 시각을 옮겨 만든다(별도 모듈)."""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import random

import numpy as np

GAPS: Dict[str, Callable[[random.Random], float]] = {
    "natural": lambda r: min(max(r.gauss(0.2, 0.25), 0.05), 1.0),
    "mediated": lambda r: r.uniform(1.0, 3.0),
}
SAME_SPEAKER_PAUSE = (0.3, 0.8)


@dataclass
class Placed:
    turn_id: str
    speaker: str
    start_s: float
    end_s: float
    gap_before_s: float


def place_turns(turns: Sequence, gap: str = "natural", seed: int = 0, lead_s: float = 0.5) -> List[Placed]:
    """turns(.turn_id·.speaker·.duration_s 를 가진 객체, 대화 순서) → 겹침 없는 배치. seed 로 재현."""
    r = random.Random(seed); out: List[Placed] = []; t = lead_s; prev = None
    for tu in turns:
        g = 0.0 if prev is None else (r.uniform(*SAME_SPEAKER_PAUSE) if tu.speaker == prev.speaker else GAPS[gap](r))
        start = t + g; end = start + float(tu.duration_s)
        out.append(Placed(tu.turn_id, tu.speaker, round(start, 4), round(end, 4), round(g, 4))); t = end; prev = tu
    return out


def render(placed: List[Placed], audio: Dict[str, np.ndarray], speakers: List[str], sr: int, tail_s: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """배치 + 턴별 음성(float32 mono, sr) → (mono 혼합 (N,), 2 채널 (N, len(speakers))). 겹침이 있으면 혼합에서 더해진다."""
    n = int(round((max(p.end_s for p in placed) + tail_s) * sr)) if placed else 0
    st = np.zeros((n, len(speakers)), dtype=np.float32)
    for p in placed:
        x = audio[p.turn_id]; a = int(round(p.start_s * sr)); b = min(a + len(x), n)
        st[a:b, speakers.index(p.speaker)] += x[: b - a]
    mono = np.clip(st.sum(1), -1.0, 1.0)
    return mono, np.clip(st, -1.0, 1.0)


def trim_silence(x: np.ndarray, sr: int, frame_s: float = 0.02, rel: float = 0.1, margin_s: float = 0.1) -> Tuple[int, int]:
    """턴 파일 앞뒤 무음 경계(샘플 index). 20 ms 프레임 RMS 가 (95 백분위 × rel) 를 넘는 첫·마지막 프레임 ± margin.
    TAXI 처럼 버튼으로 턴을 자른 파일은 앞 0.9 s·뒤 0.4 s(중앙값) 무음이 있어, 그대로 이으면 말 사이 간격이 과장된다."""
    fr = max(int(frame_s * sr), 1); n = len(x) // fr
    if n == 0:
        return 0, len(x)
    e = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1)); thr = max(float(np.percentile(e, 95)) * rel, 1e-4)
    on = np.nonzero(e > thr)[0]
    if len(on) == 0:
        return 0, len(x)
    m = int(margin_s * sr)
    return max(on[0] * fr - m, 0), min((on[-1] + 1) * fr + m, len(x))


def overlap_ratio(placed: List[Placed]) -> float:
    """두 화자가 동시에 말하는 시간 / 전체 발화 시간(턴 구간 기준)."""
    ev = sorted([(p.start_s, 1, p.speaker) for p in placed] + [(p.end_s, -1, p.speaker) for p in placed])
    active: Dict[str, int] = {}; last = None; both = speech = 0.0
    for t, d, s in ev:
        if last is not None:
            k = sum(1 for v in active.values() if v > 0)
            speech += (t - last) if k >= 1 else 0.0; both += (t - last) if k >= 2 else 0.0
        active[s] = active.get(s, 0) + d; last = t
    return both / speech if speech > 0 else 0.0
