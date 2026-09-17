"""Phase 2 평가 지표 v2 — 네 축을 분리한다([[output-phase2-eval-plan]] §1).
  A. 인식(화자 무관): utterance-assigned 오류율 — 가설 구간의 단어를 시간 겹침이 가장 큰 참조 발화에 붙여 채점(lane 무시). pooled(시간순 결합) 도 참고로.
  B. 화자 귀속: cp 오류율(lane→화자 최적 배정, 쌍별 편집거리 비용 행렬 + 최적 배정), lane-DER(누락·오경보·혼동), 화자별 lane 전환 수.
  C. 구간·턴: ONSET/EOT 매칭 P·R(허용오차 다중)과 시각 오차, 턴 교대 검출 F1·지연, turn 내 EOT 오방출률, 활동 F1.
  D. 지연: 토큰 지연은 lane_state.token_latency 를 쓴다.
입력 규약: ref_utts = [dict(speaker, start, end, text)], hyp_segs = [dict(lane, start, end, text)] — 시각은 같은 기준(창 상대 또는 절대), text 는 이미 정규화된 문자열.
단위: unit="cer" 면 공백 제거 글자열, "wer" 면 공백 단어열.
"""
import itertools
from typing import Dict, List, Optional, Tuple
import numpy as np

def units(text: str, unit: str) -> List[str]:
    return list((text or "").replace(" ", "")) if unit == "cer" else (text or "").split()

def edit_ops(hyp: List[str], ref: List[str]) -> Tuple[int, int, int]:
    """(치환, 누락, 삽입) — 표준 Levenshtein 역추적."""
    n, m = len(ref), len(hyp)
    if n == 0: return 0, 0, m
    if m == 0: return 0, n, 0
    D = np.zeros((n + 1, m + 1), dtype=np.int32); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1): D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ri != hyp[j - 1]))
    i, j = n, m; S = Dl = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]): S += int(ref[i - 1] != hyp[j - 1]); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1: Dl += 1; i -= 1
        else: I += 1; j -= 1
    return S, Dl, I

def edit_distance(hyp: List[str], ref: List[str]) -> int:
    s, d, i = edit_ops(hyp, ref); return s + d + i

def overlap(a0, a1, b0, b1) -> float: return max(0.0, min(a1, b1) - max(a0, b0))

# ── A. 인식(화자 무관)
def utterance_assigned(ref_utts: List[dict], hyp_segs: List[dict], unit: str) -> dict:
    """가설 구간 → 겹침 최대 참조 발화. 겹치는 참조가 없으면 가장 가까운 발화(시작 시각 거리 ≤ 1 s) 에 붙이고, 그것도 없으면 전부 삽입.
    참조 발화별로 붙은 가설을 시간순 결합해 편집거리. → dict(rate, S, D, I, n_ref, n_unassigned_units)."""
    buckets: Dict[int, List[Tuple[float, str]]] = {i: [] for i in range(len(ref_utts))}; loose = 0
    for h in hyp_segs:
        hs, he = h["start"], h["end"] if h["end"] is not None else h["start"] + 0.5
        best, bo = None, 0.0
        for i, r in enumerate(ref_utts):
            ov = overlap(hs, he, r["start"], r["end"])
            if ov > bo: best, bo = i, ov
        if best is None and ref_utts:
            d, best2 = min((min(abs(hs - r["start"]), abs(he - r["end"])), i) for i, r in enumerate(ref_utts))
            if d <= 1.0: best = best2
        if best is None: loose += len(units(h["text"], unit)); continue
        buckets[best].append((hs, h["text"]))
    S = Dl = I = 0; n = 0
    for i, r in enumerate(ref_utts):
        hyp = " ".join(t for _, t in sorted(buckets[i], key=lambda x: x[0])); ru = units(r["text"], unit); hu = units(hyp, unit)
        s, d, ins = edit_ops(hu, ru); S += s; Dl += d; I += ins; n += len(ru)
    I += loose
    return dict(rate=((S + Dl + I) / n if n else None), S=S, D=Dl, I=I, n_ref=n, unassigned_units=loose)

def pooled(ref_utts: List[dict], hyp_segs: List[dict], unit: str) -> dict:
    ru = units(" ".join(r["text"] for r in sorted(ref_utts, key=lambda r: r["start"])), unit); hu = units(" ".join(h["text"] for h in sorted(hyp_segs, key=lambda h: (h["start"] if h["start"] is not None else 0.0))), unit)
    s, d, i = edit_ops(hu, ru); return dict(rate=((s + d + i) / len(ru) if ru else None), S=s, D=d, I=i, n_ref=len(ru))

# ── B. 화자 귀속
def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    """최소 비용 1:1 배정(행 ≤ 열 가정 없음). 작으면 전수, 크면 scipy."""
    R, C = cost.shape
    if R == 0 or C == 0: return []
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost); return list(zip(ri.tolist(), ci.tolist()))
    except ImportError:
        k = min(R, C); best, bp = None, None
        rows, cols = list(range(R)), list(range(C))
        for rs in itertools.combinations(rows, k):
            for cs in itertools.permutations(cols, k):
                v = sum(cost[r, c] for r, c in zip(rs, cs))
                if best is None or v < best: best, bp = v, list(zip(rs, cs))
        return bp or []

def cp_error(ref_utts: List[dict], hyp_segs: List[dict], unit: str) -> dict:
    """cpWER/cpCER: 화자별 참조 결합 텍스트 vs lane 별 가설 결합 텍스트의 쌍별 편집거리 비용 행렬에 최적 배정. 배정 안 된 lane 은 전부 삽입, 배정 안 된 화자는 전부 누락.
    → dict(rate, errors, n_ref, mapping{lane: speaker}, unmatched_lanes, unmatched_speakers)."""
    spk = sorted({r["speaker"] for r in ref_utts}); lanes = sorted({h["lane"] for h in hyp_segs})
    rt = {s: units(" ".join(r["text"] for r in sorted(ref_utts, key=lambda r: r["start"]) if r["speaker"] == s), unit) for s in spk}
    ht = {l: units(" ".join(h["text"] for h in sorted(hyp_segs, key=lambda h: (h["start"] if h["start"] is not None else 0.0)) if h["lane"] == l), unit) for l in lanes}
    n_ref = sum(len(v) for v in rt.values())
    if not lanes: return dict(rate=(1.0 if n_ref else None), errors=n_ref, n_ref=n_ref, mapping={}, unmatched_lanes=[], unmatched_speakers=spk)
    cost = np.zeros((len(lanes), len(spk)), dtype=np.int64)
    for i, l in enumerate(lanes):
        for j, s in enumerate(spk): cost[i, j] = edit_distance(ht[l], rt[s])
    pairs = _assign(cost); mapping = {lanes[i]: spk[j] for i, j in pairs}
    errors = int(sum(cost[i, j] for i, j in pairs)) + sum(len(ht[l]) for l in lanes if l not in mapping) + sum(len(rt[s]) for s in spk if s not in mapping.values())
    return dict(rate=(errors / n_ref if n_ref else None), errors=errors, n_ref=n_ref, mapping=mapping, unmatched_lanes=[l for l in lanes if l not in mapping], unmatched_speakers=[s for s in spk if s not in mapping.values()])

def lane_der(ref_utts: List[dict], hyp_segs: List[dict], mapping: Dict[int, str], collar: float = 0.25, step: float = 0.01, horizon: Optional[float] = None) -> dict:
    """프레임(step) 단위 DER: 참조 화자 활동 vs 매핑된 lane 활동. collar 안(참조 경계 ±collar)은 채점 제외. → dict(der, miss, fa, conf, ref_time)."""
    T = horizon if horizon is not None else max([r["end"] for r in ref_utts] + [h["end"] or 0.0 for h in hyp_segs] + [0.0]); n = int(np.ceil(T / step)) + 1
    spk = sorted({r["speaker"] for r in ref_utts} | set(mapping.values())); idx = {s: i for i, s in enumerate(spk)}
    ref = np.zeros((len(spk), n), dtype=bool); hyp = np.zeros((len(spk), n), dtype=bool); score = np.ones(n, dtype=bool); other = np.zeros(n, dtype=bool)
    for r in ref_utts:
        a, b = int(r["start"] / step), int(np.ceil(r["end"] / step)); ref[idx[r["speaker"]], a:b] = True
        for e in (r["start"], r["end"]): score[max(0, int((e - collar) / step)): int(np.ceil((e + collar) / step))] = False
    for h in hyp_segs:
        a, b = int((h["start"] or 0.0) / step), int(np.ceil((h["end"] if h["end"] is not None else T) / step))
        if h["lane"] in mapping: hyp[idx[mapping[h["lane"]]], a:b] = True
        else: other[a:b] = True
    nref = ref.sum(0); nhyp = hyp.sum(0) + other; both = (ref & hyp).sum(0)
    miss = np.maximum(nref - nhyp, 0)[score].sum(); fa = np.maximum(nhyp - nref, 0)[score].sum(); conf = (np.minimum(nref, nhyp) - both)[score].sum(); tot = nref[score].sum()
    return dict(der=((miss + fa + conf) / tot if tot else None), miss=(miss / tot if tot else None), fa=(fa / tot if tot else None), conf=(conf / tot if tot else None), ref_time=round(tot * step, 2))

def lane_switches(ref_utts: List[dict], hyp_segs: List[dict]) -> dict:
    """화자별로 (겹침 최대 참조 발화의 화자 기준) 가설 lane 이 시간순으로 바뀐 횟수. → dict(switches, per_speaker{spk: [lanes...]}, n_segments)."""
    seq: Dict[str, List[int]] = {}
    for h in sorted(hyp_segs, key=lambda h: (h["start"] if h["start"] is not None else 0.0)):
        best, bo = None, 0.0
        for r in ref_utts:
            ov = overlap(h["start"] or 0.0, h["end"] if h["end"] is not None else (h["start"] or 0.0) + 0.5, r["start"], r["end"])
            if ov > bo: best, bo = r["speaker"], ov
        if best is not None: seq.setdefault(best, []).append(h["lane"])
    sw = sum(sum(1 for a, b in zip(v, v[1:]) if a != b) for v in seq.values())
    return dict(switches=sw, per_speaker=seq, n_segments=sum(len(v) for v in seq.values()))

# ── C. 구간·턴
def match_events(hyp_t: List[float], ref_t: List[float], tol: float) -> Tuple[int, int, int, List[float]]:
    """|Δ| ≤ tol 탐욕 1:1 매칭 → (일치, hyp 수, ref 수, 일치 쌍의 hyp−ref 오차 목록)."""
    used = [False] * len(ref_t); hit = 0; errs = []
    for h in sorted(hyp_t):
        best, bi = tol + 1, -1
        for i, r in enumerate(ref_t):
            if not used[i] and abs(h - r) < best: best, bi = abs(h - r), i
        if bi >= 0 and best <= tol: used[bi] = True; hit += 1; errs.append(h - ref_t[bi])
    return hit, len(hyp_t), len(ref_t), errs

def event_metrics(hyp_t: List[float], ref_t: List[float], tols=(0.2, 0.4, 0.8)) -> dict:
    out = {}
    for tol in tols:
        hit, nh, nr, errs = match_events(hyp_t, ref_t, tol)
        out[str(tol)] = dict(hit=hit, hyp=nh, ref=nr, p=(hit / nh if nh else None), r=(hit / nr if nr else None), err_median=(float(np.median(errs)) if errs else None), err_p90=(float(np.percentile(np.abs(errs), 90)) if errs else None))
    return out

def turn_changes(utts: List[dict], key: str) -> List[Tuple[float, str, str]]:
    """시간순 발화에서 화자(key)가 바뀌는 지점: (다음 발화 시작 시각, 앞 화자, 다음 화자). 같은 화자가 이어지면 교대 아님."""
    out = []; prev = None
    for u in sorted(utts, key=lambda u: (u["start"] if u["start"] is not None else 0.0)):
        cur = u[key]
        if prev is not None and cur != prev: out.append((u["start"] if u["start"] is not None else 0.0, prev, cur))
        prev = cur
    return out

def turn_change_metrics(ref_utts: List[dict], hyp_segs: List[dict], tol: float = 0.5) -> dict:
    """참조 화자 교대 시점 vs 가설 lane 교대 시점(±tol) → F1 과 검출 지연(hyp − ref) 중앙값."""
    r = [t for t, _, _ in turn_changes(ref_utts, "speaker")]; h = [t for t, _, _ in turn_changes([s for s in hyp_segs if s["start"] is not None], "lane")]
    hit, nh, nr, errs = match_events(h, r, tol); p = hit / nh if nh else None; rc = hit / nr if nr else None
    return dict(hit=hit, hyp=nh, ref=nr, p=p, r=rc, f1=((2 * p * rc / (p + rc)) if p and rc else (0.0 if nr else None)), latency_median=(float(np.median(errs)) if errs else None))

def eot_in_turn(ref_utts: List[dict], hyp_eot_t: List[float], mapping: Dict[int, str], hyp_eot_lanes: Optional[List[int]] = None, margin: float = 0.3) -> dict:
    """turn 내 EOT 오방출: 가설 EOT 시각이 (매핑된 화자의) 참조 발화 안쪽(끝 − margin 이전)에 떨어진 수 / 참조 발화 총 시간(분). lane 정보가 없으면 어느 화자든 발화 중이면 오방출로 센다."""
    dur = sum(r["end"] - r["start"] for r in ref_utts); bad = 0
    for i, t in enumerate(hyp_eot_t):
        spk = mapping.get(hyp_eot_lanes[i]) if hyp_eot_lanes is not None else None
        for r in ref_utts:
            if (spk is None or r["speaker"] == spk) and r["start"] + 0.1 <= t <= r["end"] - margin: bad += 1; break
    return dict(count=bad, per_min=(bad / (dur / 60) if dur else None), ref_speech_s=round(dur, 2))

def activity_f1(ref_act: np.ndarray, hyp_act: np.ndarray, lane_to_col: Dict[int, int]) -> dict:
    """ref_act (K, S) 화자 열, hyp_act (K, R) lane 열, lane_to_col: lane→화자 열 매핑. → 매핑된 열끼리 F1."""
    tp = fp = fn = 0
    for l, c in lane_to_col.items():
        h = hyp_act[:, l - 1] > 0.5; r = ref_act[:, c] > 0.5; tp += int((h & r).sum()); fp += int((h & ~r).sum()); fn += int((~h & r).sum())
    fn += int(sum(int((ref_act[:, c] > 0.5).sum()) for c in range(ref_act.shape[1]) if c not in lane_to_col.values()))
    return dict(f1=((2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None), tp=tp, fp=fp, fn=fn)

def bootstrap_ci(values: List[float], weights: Optional[List[float]] = None, n: int = 1000, seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """세션 단위 값의 (가중) 평균 95 % CI."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2: return None, None
    w = np.array([weights[i] for i, v in enumerate(values) if v is not None], dtype=float) if weights else np.ones(len(vals)); v = np.array(vals, dtype=float); rng = np.random.default_rng(seed); est = []
    for _ in range(n):
        idx = rng.integers(0, len(v), len(v)); est.append(float((v[idx] * w[idx]).sum() / w[idx].sum()))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))
