"""IS-SLM interleaved target 시퀀스 생성 (U0).

입력: 화자별 토큰 스트림 [(token_id, end_time_s)], chunk 길이(80 ms), 지연 δ(프레임), chunk 당 최대 방출 M.
출력: chunk 단위 [(chunk_idx, [emissions...])], emissions = 특수/텍스트 토큰 id 리스트. 시퀀스 규약:
  chunk k:  <AUDIO_k>  [ <SPK_x> 토큰(화자 바뀔 때) tok tok ... ]  <NEXT_AUDIO>
  토큰은 (end_time + δ) 가 속한 chunk 에 배치. chunk 내 정렬은 종료 시각 순, 동시면 A 우선.
  M 초과분은 다음 chunk 로 이월(overflow) — 이월 수·추가 지연을 통계로 남긴다.
특수 토큰 id 는 호출자가 준다 (U1 에서 tokenizer 에 추가).
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math

@dataclass
class Specials:
    next_audio: int; empty_audio: int; spk: Tuple[int, int]; onset: int = -1; endpoint: int = -1

@dataclass
class InterleaveStats:
    chunks: int = 0; tokens: int = 0; overflow_tokens: int = 0; max_backlog: int = 0; extra_delay_frames: int = 0
    per_chunk_hist: Dict[int, int] = field(default_factory=dict)

def build_interleaved(streams: List[List[Tuple[int, float]]], duration_s: float, sp: Specials, chunk_s: float = 0.08,
                      delay_frames: int = 2, max_per_chunk: int = 4, add_spk_tags: bool = True):
    """streams[s] = [(token_id, end_time_s), ...] 시간순. → (chunk_emissions, stats)"""
    n_chunks = int(math.ceil(duration_s / chunk_s)); buckets: List[List[Tuple[float, int, int]]] = [[] for _ in range(n_chunks + 1)]
    for s, toks in enumerate(streams):
        for tid, t_end in toks:
            k = min(n_chunks, int(t_end / chunk_s) + delay_frames); buckets[k].append((t_end, s, tid))
    out = []; st = InterleaveStats(chunks=n_chunks); backlog: List[Tuple[float, int, int]] = []; last_spk = None
    for k in range(n_chunks):
        pending = sorted(backlog + buckets[k], key=lambda x: (x[0], x[1])); backlog = []
        emit, n_txt = [], 0
        for item in pending:
            if n_txt >= max_per_chunk: backlog.append(item); continue
            t_end, s, tid = item
            if add_spk_tags and s != last_spk: emit.append(sp.spk[s]); last_spk = s
            emit.append(tid); n_txt += 1; st.tokens += 1
        st.overflow_tokens += len(backlog); st.max_backlog = max(st.max_backlog, len(backlog)); st.extra_delay_frames += len(backlog)
        st.per_chunk_hist[n_txt] = st.per_chunk_hist.get(n_txt, 0) + 1
        emit.append(sp.next_audio); out.append((k, emit))
    if backlog:   # 스트림 종료: EMPTY_AUDIO 뒤 flush
        out.append((n_chunks, [tid for _, _, tid in sorted(backlog)] + [sp.empty_audio]))
    return out, st
