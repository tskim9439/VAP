"""EOT soft target — 구간 끝 뒤 horizon 안의 참조 결과로 p_end 를 정한다 (정본 §5.2, decision-eot-immediate-soft-label).

결과 분류(구간 끝 e, horizon H):
  shift      다른 화자가 e 시점에 겹치거나 (e, e+H] 안에 시작, 본인 재개 없음          → 1.0
  silence    아무도 말하지 않음                                                        → 0.8
  both       다른 화자도 말하고 본인도 H 안에 재개                                     → 0.5
  hold_long  본인만 1–3 s 뒤 재개                                                      → 0.3
  hold_short 본인만 1 s 안 재개                                                        → 0.0
  unobserved e+H 가 observed_until 을 넘음(EOF·결손)                                   → None (mask)
미래는 정답 산출에만 쓴다. 수치는 초기값이며 Q0 에서 동결한다.
"""
from typing import List, Optional, Dict
from .dialogue import Episode

HORIZON_S = 3.0
P_END: Dict[str, Optional[float]] = {"shift": 1.0, "silence": 0.8, "both": 0.5, "hold_long": 0.3, "hold_short": 0.0, "unobserved": None}

def classify(ep: Episode, episodes: List[Episode], observed_until: float, horizon: float = HORIZON_S) -> str:
    e = ep.end
    if e + horizon > observed_until + 1e-9: return "unobserved"
    nxt = min((o.start for o in episodes if o.speaker == ep.speaker and o.start > ep.start and o.start >= e), default=None)
    self_gap = (nxt - e) if nxt is not None and nxt - e <= horizon else None
    other = any((o.start <= e < o.end) or (e < o.start <= e + horizon) for o in episodes if o.speaker != ep.speaker)
    if other and self_gap is None: return "shift"
    if other: return "both"
    if self_gap is not None: return "hold_short" if self_gap < 1.0 else "hold_long"
    return "silence"

def assign_p_end(episodes: List[Episode], observed_until: float, horizon: float = HORIZON_S, table: Dict[str, Optional[float]] = P_END) -> Dict[str, int]:
    """모든 episode 에 outcome/p_end 를 채우고 분류 히스토그램을 돌려준다."""
    hist: Dict[str, int] = {}
    for ep in episodes:
        ep.outcome = classify(ep, episodes, observed_until, horizon); ep.p_end = table[ep.outcome]
        hist[ep.outcome] = hist.get(ep.outcome, 0) + 1
    return hist
