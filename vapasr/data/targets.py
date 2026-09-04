"""VAD(초 단위 세그먼트) → 학습 target.

- vad_frames   : (T, 2) float @ frame_hz  (원 VAP 의 vad_list_to_onehot 과 동일 규칙)
- vap_labels   : (T,) long 256-class  — 원 VAP 저장소의 ObjectiveVAP.get_labels 를 그대로 사용 (bins [0.2,0.4,0.6,0.8] 누적 = [0,.2],[.2,.6],[.6,1.2],[1.2,2.0])
- pool_vad     : 50 Hz → 12.5 Hz (any-pool)
- time_to_next_onset : 화자별 다음 onset 까지의 시간(프레임) + censored 플래그 → hazard bin
- derive_events: VAD 만으로 EOT(SHIFT)/HOLD/INTERRUPT/BACKCHANNEL 유도 (휴리스틱 — otoSpeech 사람 라벨로 검증 필요)
"""
from typing import List, Tuple, Dict
import numpy as np, torch

BIN_TIMES = [0.2, 0.4, 0.6, 0.8]                    # 원 VAP 기본 (50 Hz: 10/20/30/40 프레임)
BIN_TIMES_12HZ = [0.16, 0.4, 0.64, 0.8]             # 12.5 Hz(80 ms): 2/5/8/10 프레임 = 누적 0.16/0.56/1.2/2.0 s — 정수 프레임 근사
HAZARD_EDGES_S = [0.16, 0.32, 0.64, 1.28, 2.56]     # τ bin 경계 (초). 마지막 이후 = censored
EVENT_TYPES = ["SHIFT", "HOLD", "INTERRUPT", "BACKCHANNEL"]   # SHIFT = EOT 로 이어진 교대

def vad_frames(vad: List[List[Tuple[float, float]]], frame_hz: float, duration: float) -> np.ndarray:
    T = int(round(duration * frame_hz)); out = np.zeros((T, 2), dtype=np.float32)
    for c in (0, 1):
        for s, e in vad[c]:
            a, b = int(round(s * frame_hz)), int(round(e * frame_hz)); out[max(0, a): min(T, b), c] = 1.0
    return out

def pool_vad(va50: np.ndarray, factor: int = 4, mode: str = "any") -> np.ndarray:
    T = (len(va50) // factor) * factor; v = va50[:T].reshape(-1, factor, va50.shape[1])
    return (v.max(1) if mode == "any" else (v.mean(1) >= 0.5).astype(np.float32))

_OBJ = {}
def vap_bin_times(frame_hz: float) -> List[float]:
    return BIN_TIMES_12HZ if abs(frame_hz - 12.5) < 1e-6 else BIN_TIMES

def vap_labels(va: np.ndarray, frame_hz: float) -> np.ndarray:
    """va (T,2) → (T,) int64. 마지막 horizon 프레임은 원 VAP 와 같이 유효하지 않음(패딩, -100) — 학습 시 마스킹.
    frame_hz 50 → 원 VAP bins, 12.5 → BIN_TIMES_12HZ."""
    from vap.objective import ObjectiveVAP
    if frame_hz not in _OBJ: _OBJ[frame_hz] = ObjectiveVAP(bin_times=vap_bin_times(frame_hz), frame_hz=frame_hz)
    obj = _OBJ[frame_hz]
    lab = obj.get_labels(torch.from_numpy(va)[None].float())      # (1, T-horizon)
    out = np.full(len(va), -100, dtype=np.int64); out[: lab.shape[1]] = lab[0].numpy(); return out

def _onsets(v: np.ndarray, min_gap_frames: int) -> np.ndarray:
    """v (T,) {0,1}: 최소 min_gap 침묵 뒤에 시작하는 프레임 index 들."""
    on = np.flatnonzero(np.diff(np.concatenate([[0], v])) == 1); keep = []
    for i in on:
        if i == 0 or v[max(0, i - min_gap_frames): i].sum() == 0: keep.append(i)
    return np.array(keep, dtype=int)

def time_to_next_onset(va: np.ndarray, frame_hz: float, min_gap_s: float = 0.2, horizon_s: float = HAZARD_EDGES_S[-1]) -> Dict[str, np.ndarray]:
    """각 프레임 t, 각 화자 s 에 대해 τ = (s 의 다음 onset 프레임) − t. s 가 t 에서 발화 중이면 현재 세그먼트는 제외(재개 아님).
    returns tau (T,2) float 프레임 단위 (없으면 inf), censored (T,2) bool (horizon 내 onset 없음), bin (T,2) int (0..len(edges), 마지막 = censored)"""
    T = len(va); H = int(round(horizon_s * frame_hz)); tau = np.full((T, 2), np.inf, dtype=np.float32)
    for s in (0, 1):
        on = _onsets(va[:, s], int(min_gap_s * frame_hz)); j = 0
        for t in range(T):
            while j < len(on) and on[j] <= t: j += 1
            if j < len(on): tau[t, s] = on[j] - t
    censored = ~np.isfinite(tau) | (tau > H)
    edges_f = np.array(HAZARD_EDGES_S) * frame_hz
    bins = np.digitize(np.where(np.isfinite(tau), tau, 1e9), edges_f)      # 0..len(edges)
    bins[censored] = len(HAZARD_EDGES_S)
    return dict(tau=tau, censored=censored, bin=bins.astype(np.int64))

def derive_events(va: np.ndarray, frame_hz: float, pause_s: float = 0.2, window_s: float = 3.0, bc_max_s: float = 1.0,
                  overlap_tol_s: float = 0.5, resume_s: float = 1.0) -> List[Tuple[float, int, str]]:
    """VAD 휴리스틱 이벤트. 반환 [(time_s, speaker, type)] — speaker 는 '행위자' (SHIFT/HOLD: 말을 마친 화자, INTERRUPT/BACKCHANNEL: 끼어든 화자).
    규칙:
      세그먼트 종료 t_e (화자 A, 이후 ≥ pause 침묵) → window(3 s = TurnBench 매칭창) 내 첫 발화가 B 이면 SHIFT@t_e, A 이면 HOLD@t_e, 아무도 없으면 HOLD
      B 가 A 발화 중 시작(overlap) →
         A 가 onset 후 overlap_tol 안에 끝나면 = terminal overlap (자연스러운 교대) → SHIFT@A_end (행위자 A), INT 아님
         그 밖에 B 세그먼트가 A 의 종료를 지나 이어지고 A 가 window 내 재개 안 하면 INTERRUPT@onset (B)
         B 세그먼트 ≤ bc_max 이고 A 가 계속 말하면 BACKCHANNEL@onset (B)
      종료 시점에 상대가 이미 말하는 중이면(방해당함/backchannel 종료) 종료 이벤트 없음 — terminal overlap 은 위에서 SHIFT 로 처리
    """
    T = len(va); P = int(pause_s * frame_hz); W = int(window_s * frame_hz); BC = int(bc_max_s * frame_hz); OT = int(overlap_tol_s * frame_hz); RS = int(resume_s * frame_hz); ev = []
    segs = {s: [] for s in (0, 1)}
    for s in (0, 1):
        v = va[:, s]; i = 0
        while i < T:
            if v[i]:
                j = i
                while j < T and v[j]: j += 1
                segs[s].append((i, j)); i = j
            else: i += 1
    for s in (0, 1):
        o = 1 - s
        for (i, j) in segs[s]:
            # 종료 이벤트: 다음 pause 이상 침묵인 종료 + 종료 시점에 상대가 말하고 있지 않을 것(= 이 화자가 floor 를 쥐고 있었음).
            # 상대가 이미 말하는 중이면 backchannel 종료이거나 방해당한 종료 → EOT 가 아니다.
            if j + P <= T and va[j: j + P, s].sum() == 0 and va[j - 1, o] == 0:
                nxt_o = np.flatnonzero(va[j: j + W, o]); nxt_s = np.flatnonzero(va[j: j + W, s])
                if len(nxt_o) and (not len(nxt_s) or nxt_o[0] < nxt_s[0]): ev.append((j / frame_hz, s, "SHIFT"))
                else: ev.append((j / frame_hz, s, "HOLD"))
            # 시작 이벤트: 상대가 말하는 중에 시작
            if i > 0 and va[i, o] == 1 and va[i - 1, s] == 0:
                o_end = i
                while o_end < T and va[o_end, o]: o_end += 1
                resumes = va[o_end: o_end + RS, o].sum() > 0 if o_end < T else False
                if (o_end - i) <= OT and j > o_end and not resumes:
                    ev.append((o_end / frame_hz, o, "SHIFT"))                 # terminal overlap: 상대의 EOT
                elif j > o_end and not resumes: ev.append((i / frame_hz, s, "INTERRUPT"))
                elif (j - i) <= BC: ev.append((i / frame_hz, s, "BACKCHANNEL"))
    return sorted(ev)
