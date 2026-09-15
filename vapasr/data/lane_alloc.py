"""Phase 2 lane 배정 — reference allocator (정본 §3.1–3.2).

정책
  lazy_free  (정본):  1) 화자가 소유한 lane 이 있으면 그 lane  2) 없으면 FREE 중 첫째  3) FREE 없으면 HELD 중 lexical 이 가장 먼저 닫힌 lane
                     (단, 이전 소유자의 EOT 청크 < 새 ONSET 청크 여야 후보)  4) 없으면 lane=None (lane_capacity_exhausted)
  never_free (이전 정본 K 슬롯): 도착순으로 1..R, 초과 화자는 None.
N ≤ R 이면 두 정책의 결과가 같다(정본 §3.1 따름정리) — tests/test_lane_protocol.py 가 고정한다.

lane 상태(라벨 측): FREE → OPEN(에피소드 진행) → HELD(closed_at = 에피소드 끝, eot_chunk = EOT 후보 청크) → 재개/재배정.
시간 경과로 FREE 로 돌아가지 않는다.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
from .dialogue import Episode, chunk_of, CHUNK_S

EOT_GAP_S = 0.24     # EOT 후보 = max(마지막 lexical 청크, floor((end+0.24)/0.08)) (정본 §5.1)

def eot_chunk(ep: Episode, delay_text: int, chunk_s: float = CHUNK_S) -> int:
    k_last = max([chunk_of(t, delay_text, chunk_s) for _, t in ep.tokens], default=-1)
    return max(k_last, chunk_of(ep.end + EOT_GAP_S, 0, chunk_s))

@dataclass
class Lane:
    idx: int; owner: Optional[str] = None; generation: int = 0; state: str = "FREE"; closed_chunk: int = -1

@dataclass
class AllocStats:
    episodes: int = 0; reassigned: int = 0; exhausted: int = 0; max_lanes_used: int = 0

def allocate(episodes: List[Episode], R: int = 6, policy: str = "lazy_free", delay_text: int = 4, delay_onset: int = 0,
             chunk_s: float = CHUNK_S) -> AllocStats:
    """episodes(시작순)에 lane/generation 을 채운다(in place). 반환: 통계."""
    assert policy in ("lazy_free", "never_free"), policy
    lanes = [Lane(i + 1) for i in range(R)]; st = AllocStats(); order: Dict[str, int] = {}
    # 시간순으로 처리하되, 각 에피소드가 시작하는 시점에 이미 끝난 에피소드의 lane 을 HELD 로 갱신한다
    active: List[Episode] = []
    for ep in sorted(episodes, key=lambda e: (e.start, e.speaker)):
        st.episodes += 1
        still = []
        for a in active:
            if a.end <= ep.start and a.lane is not None:
                ln = lanes[a.lane - 1]
                if ln.owner == a.speaker and ln.generation == a.generation:
                    ln.state = "HELD"; ln.closed_chunk = eot_chunk(a, delay_text, chunk_s)
            else: still.append(a)
        active = still
        if policy == "never_free":
            if ep.speaker not in order:
                if len(order) < R: order[ep.speaker] = len(order) + 1
                else: order[ep.speaker] = 0
            lane = order[ep.speaker] or None
            if lane is None: st.exhausted += 1; ep.lane = None; continue
            ln = lanes[lane - 1]; ln.owner = ep.speaker; ln.state = "OPEN"; ep.lane = lane; ep.generation = ln.generation
            active.append(ep); st.max_lanes_used = max(st.max_lanes_used, len(order)); continue
        own = [ln for ln in lanes if ln.owner == ep.speaker]
        if own: ln = own[0]
        else:
            free = [ln for ln in lanes if ln.state == "FREE"]
            if free: ln = free[0]
            else:
                k_on = chunk_of(ep.start, delay_onset, chunk_s)
                held = [ln for ln in lanes if ln.state == "HELD" and ln.closed_chunk < k_on]
                if not held: st.exhausted += 1; ep.lane = None; continue
                ln = min(held, key=lambda l: (l.closed_chunk, l.idx)); ln.generation += 1; st.reassigned += 1
            ln.owner = ep.speaker
        ln.state = "OPEN"; ep.lane = ln.idx; ep.generation = ln.generation; active.append(ep)
        st.max_lanes_used = max(st.max_lanes_used, sum(1 for l in lanes if l.state != "FREE"))
    return st
