"""채널별 에너지 VAD. 분리 stereo(누설 −64 dB) 전제. 히스테리시스 + 최소 길이 규칙."""
from typing import List, Tuple
import numpy as np

def frame_db(x: np.ndarray, sr: int, hop_ms: float) -> np.ndarray:
    n = int(sr * hop_ms / 1000); T = len(x) // n
    fr = x[: T * n].reshape(T, n).astype(np.float32)
    return 10 * np.log10((fr ** 2).mean(1) + 1e-10)

def energy_vad(x: np.ndarray, sr: int, hop_ms: float = 20.0, on_db: float = -40.0, off_db: float = -45.0,
               min_speech_ms: float = 100.0, min_silence_ms: float = 200.0, adaptive: bool = True) -> np.ndarray:
    """(T,) bool @ 1000/hop_ms Hz. adaptive: 임계값을 채널 노이즈 바닥(10 백분위) 기준 상대값으로 잡는다."""
    db = frame_db(x, sr, hop_ms)
    if adaptive:
        floor = np.percentile(db, 10); on_db = max(on_db, floor + 12); off_db = max(off_db, floor + 8)
    act = np.zeros(len(db), dtype=bool); on = False
    for i, v in enumerate(db):
        on = v > on_db if not on else v > off_db
        act[i] = on
    # 최소 길이 규칙 (짧은 발화/짧은 침묵 제거)
    act = _remove_short(act, int(min_silence_ms / hop_ms), value=False)
    act = _remove_short(act, int(min_speech_ms / hop_ms), value=True)
    return act

def _remove_short(a: np.ndarray, min_len: int, value: bool) -> np.ndarray:
    a = a.copy(); i = 0; T = len(a)
    while i < T:
        if a[i] == value:
            j = i
            while j < T and a[j] == value: j += 1
            if j - i < min_len: a[i:j] = not value
            i = j
        else: i += 1
    return a

def frames_to_segments(act: np.ndarray, hop_s: float) -> List[Tuple[float, float]]:
    segs = []; T = len(act); i = 0
    while i < T:
        if act[i]:
            j = i
            while j < T and act[j]: j += 1
            segs.append((i * hop_s, j * hop_s)); i = j
        else: i += 1
    return segs

def segments_to_frames(segs: List[Tuple[float, float]], hop_s: float, n_frames: int) -> np.ndarray:
    act = np.zeros(n_frames, dtype=bool)
    for s, e in segs:
        a, b = int(round(s / hop_s)), int(round(e / hop_s)); act[max(0, a): min(n_frames, b)] = True
    return act
