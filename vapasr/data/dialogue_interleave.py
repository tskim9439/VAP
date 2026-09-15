"""Phase 2 시각순 교차 직렬화 — lane 태그·ONSET·lexical·EOT(soft) (정본 §4·§5).

블록:  [AUDIO_k] payload* <NEXT_AUDIO>      payload = (<SPK_r>)? tok | <SPK_r><ONSET> | <SPK_r><EOT>
배치:  ONSET → floor(start/0.08)+δ_on,  lexical → floor(when/0.08)+δ_text (when = 화자 내 단조 end_time),
       EOT   → max(마지막 lexical 청크, floor((end+0.24)/0.08)),  같은 청크 안 정렬 = (reference_s, lane, kind ONSET<lexical<EOT, ordinal).
태그:  ONSET/EOT 앞에는 항상, lexical 앞에는 현재 selector 와 다를 때만.
soft:  EOT 를 예측하는 위치의 label 은 (EOT: p_end, EOT 를 건너뛴 다음 토큰: 1-p_end). p_end=None 이면 -100(mask). EOT 는 입력열에 항상 남는다.
꼬리:  δ 때문에 스트림 끝을 넘긴 payload 는 <EMPTY_AUDIO> 라운드로 flush 한다(기존 mono 경로와 같음).
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
from .dialogue import Episode, chunk_of, CHUNK_S
from .lane_alloc import eot_chunk

KIND_PRI = {"onset": 0, "text": 1, "eot": 2}

@dataclass
class LaneSpecials:
    next_audio: int; empty_audio: int; lanes: List[int]; onset: int; eot: int
    def lane_tag(self, lane: int) -> int: return self.lanes[lane - 1]

@dataclass
class Emit:
    tid: int; kind: str; lane: Optional[int] = None; ep: int = -1; p_end: Optional[float] = None

@dataclass
class SerStats:
    chunks: int = 0; text: int = 0; onset: int = 0; eot: int = 0; tags: int = 0; overflow: int = 0; max_backlog: int = 0
    skipped_episodes: int = 0; per_chunk_hist: Dict[int, int] = field(default_factory=dict)

def _items(episodes: List[Episode], delay_text: int, delay_onset: int, chunk_s: float, st: SerStats):
    items = []; ordinal = 0
    for ep in episodes:
        if ep.lane is None: st.skipped_episodes += 1; continue
        items.append((chunk_of(ep.start, delay_onset, chunk_s), ep.start, ep.lane, 0, ordinal, ep, None)); ordinal += 1
        when = ep.start
        for tid, t in ep.tokens:
            when = max(t, when)
            items.append((chunk_of(when, delay_text, chunk_s), when, ep.lane, 1, ordinal, ep, tid)); ordinal += 1
        ref = max(ep.end, when)
        items.append((eot_chunk(ep, delay_text, chunk_s), ref, ep.lane, 2, ordinal, ep, None)); ordinal += 1
    return items

def serialize(episodes: List[Episode], duration_s: float, sp: LaneSpecials, delay_text: int = 4, delay_onset: int = 0,
              chunk_s: float = CHUNK_S, max_per_chunk: int = 0) -> Tuple[List[Tuple[int, List[Emit]]], SerStats]:
    """→ ([(chunk_idx, [Emit...]) ...], stats). chunk_idx ≥ n_chunks 는 <EMPTY_AUDIO> flush 라운드."""
    st = SerStats(); n_chunks = int(math.ceil(duration_s / chunk_s)); st.chunks = n_chunks
    buckets: List[List] = [[] for _ in range(n_chunks + 1)]
    for it in _items(episodes, delay_text, delay_onset, chunk_s, st): buckets[min(n_chunks, it[0])].append(it)
    out = []; backlog: List = []; selector: Optional[int] = None
    def emit_round(k: int, pending: List, cap: int):
        nonlocal selector, backlog
        emit: List[Emit] = []; n_txt = 0; rest: List = []
        for it in pending:
            _, ref, lane, pri, _, ep, tid = it
            if cap and n_txt >= cap: rest.append(it); continue
            if pri == 1:
                if selector != lane: emit.append(Emit(sp.lane_tag(lane), "tag", lane, ep.ep_id)); selector = lane; st.tags += 1
                emit.append(Emit(tid, "text", lane, ep.ep_id)); n_txt += 1; st.text += 1
            else:
                emit.append(Emit(sp.lane_tag(lane), "tag", lane, ep.ep_id)); selector = lane; st.tags += 1
                if pri == 0: emit.append(Emit(sp.onset, "onset", lane, ep.ep_id)); st.onset += 1
                else: emit.append(Emit(sp.eot, "eot", lane, ep.ep_id, ep.p_end)); st.eot += 1
        backlog = rest; st.overflow += len(rest); st.max_backlog = max(st.max_backlog, len(rest))
        st.per_chunk_hist[n_txt] = st.per_chunk_hist.get(n_txt, 0) + 1
        return emit
    key = lambda it: (it[1], it[2], it[3], it[4])
    for k in range(n_chunks):
        emit = emit_round(k, sorted(backlog + buckets[k], key=key), max_per_chunk)
        emit.append(Emit(sp.next_audio, "next")); out.append((k, emit))
    pending = sorted(backlog + buckets[n_chunks], key=key); backlog = []; k = n_chunks
    while pending:                                   # flush: 무제한 라운드, 한 라운드에 cap 만큼
        emit = emit_round(k, pending, max_per_chunk); pending = backlog; backlog = []
        emit.append(Emit(sp.next_audio, "next")); out.append((k, emit)); k += 1
    return out, st

def flatten(chunks: List[Tuple[int, List[Emit]]], n_chunks: int, audio_pad: int, empty_audio: int, prefix_ids: List[int] = (), mask_chunks: Optional[set] = None) -> Dict:
    """→ dict(ids, is_audio, chunk_of, labels, soft_pos, soft_alt, soft_w, kinds, lanes).
    labels[i] = ids[i] (모델이 labels[:,1:] 로 shift), prefix·audio·EMPTY 위치는 -100. soft_*: EOT 위치 j 의 대안 label 과 가중치 p_end.
    mask_chunks: 이 청크들의 payload(NEXT 포함)는 label -100 — 전사 없는 음성 구간에 아무 학습 신호도 주지 않는다(정본 §8 조치 2). soft 도 제외."""
    ids, is_audio, cof, labels, kinds, lanes = list(prefix_ids), [False] * len(prefix_ids), [-1] * len(prefix_ids), [-100] * len(prefix_ids), ["prefix"] * len(prefix_ids), [0] * len(prefix_ids)
    eot_pos: List[Tuple[int, Optional[float]]] = []; mask_chunks = mask_chunks or set(); n_masked = 0
    for k, emits in chunks:
        if k < n_chunks: ids.append(audio_pad); is_audio.append(True); cof.append(k)
        else: ids.append(empty_audio); is_audio.append(False); cof.append(-1)
        labels.append(-100); kinds.append("audio" if k < n_chunks else "empty"); lanes.append(0); m = k in mask_chunks
        for e in emits:
            if e.kind == "eot" and not m: eot_pos.append((len(ids), e.p_end))
            ids.append(e.tid); is_audio.append(False); cof.append(k if k < n_chunks else -1); labels.append(-100 if m else e.tid); kinds.append(e.kind); lanes.append(e.lane or 0); n_masked += int(m)
    soft_pos, soft_alt, soft_w = [], [], []
    for j, p in eot_pos:
        if p is None: labels[j] = -100; continue
        soft_pos.append(j); soft_alt.append(ids[j + 1]); soft_w.append(float(p))
    return dict(ids=ids, is_audio=is_audio, chunk_of=cof, labels=labels, soft_pos=soft_pos, soft_alt=soft_alt, soft_w=soft_w, kinds=kinds, lanes=lanes, n_masked=n_masked, masked_chunks=sorted(mask_chunks))

def lane_activity(episodes: List[Episode], n_chunks: int, R: int, chunk_s: float = CHUNK_S) -> List[List[int]]:
    """(n_chunks, R) 0/1 — 청크 [k·0.08, (k+1)·0.08) 와 episode 가 겹치면 그 lane 활성. lane 없는 episode 는 제외."""
    act = [[0] * R for _ in range(n_chunks)]
    for ep in episodes:
        if ep.lane is None: continue
        for k in range(max(0, int(ep.start / chunk_s)), min(n_chunks, int(math.ceil(ep.end / chunk_s)))): act[k][ep.lane - 1] = 1
    return act

def render(chunks: List[Tuple[int, List[Emit]]], names: Dict[int, str], n_chunks: int) -> str:
    """사람이 읽는 블록 문자열(테스트·QC 용). names: token_id → 표시 문자열."""
    lines = []
    for k, emits in chunks:
        head = f"[AUDIO_{k}]" if k < n_chunks else f"[EMPTY_{k - n_chunks}]"
        lines.append(head + " " + " ".join(names.get(e.tid, str(e.tid)) for e in emits))
    return "\n".join(lines)
