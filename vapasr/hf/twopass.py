"""<SEM_END> 트리거 2-pass 최종 재디코드의 순수 함수(모델·GPU 없음) — 계획 wiki/outputs/output-semcommit-two-pass-plan.md 의 최소판(SEM + EOS 트리거).

입력: semcommit_eval 스트림 jsonl 행(emitted, hyp.words/events/word_k, K, delta) + 원 스트림 길이 t_eos(words 행 duration_s).
트리거(방출 순):
  soft = 유효 commit: <SEM_END> 이벤트 중 mid_word 가 아니고 앞에 가설 단어가 있는 것(after ≥ 0), 트리거 시각 emit_time(k, K) ≤ t_eos 인 것.
         t_eos 뒤(패딩 꼬리·flush 라운드)에 나온 commit 은 버린다 — 그 시각에는 EOS(hard)가 이미 남은 오디오를 전부 확정했다.
  hard = EOS: 시각 t_eos 에 남은 오디오 [마지막 절단점, t_eos] 를 확정한다. 남은 1차 가설 단어가 없으면(마지막 commit 이 마지막 단어) 세그먼트를 만들지 않는다
         — 무음만 남은 꼬리를 2차 디코더에 넣으면 환각 위험이 있다. 1차 가설 단어가 하나도 없는 스트림은 EOS 세그먼트 하나(전체)로 둔다.
절단점(soft): commit 이 확정한 가설 단어 j(= 이벤트 after)의 마지막 토큰 방출 청크 k_last(word_k[j][1]) → cut = (k_last − δ + 1)·0.08.
  단어 끝 e 는 직렬화 계약(ref_chunk: k = int(e/0.08)+δ)상 [(k_last−δ)·0.08, (k_last−δ+1)·0.08) 에 있으므로 상한으로 자른다 — 확정 단어를 자르지 않고,
  다음 단어 앞머리가 최대 80 ms 섞일 수 있다. [직전 절단점, t_eos] 로 자른다. cut ≤ (k_sem−δ+1)·0.08 < emit_time(k_sem) 라 인과적이다.
보류(soft 만): 새 세그먼트가 min_seg_s 보다 짧거나 1차 가설 단어가 min_words 개 미만이면 자르지 않고 다음 트리거(soft·EOS)와 합친다. EOS 는 항상 확정.
최종 전사 = 세그먼트별 2차 텍스트를 순서대로 공백으로 이음(compose).
지연(algorithmic, 계산 시간 제외): 참조 단어 i(끝 e_i)는 e_i 가 든 세그먼트((start, end], 시간 기준 — 계획 §6.2)의 트리거 시각에 최종 확정 → t_final − e_i.
  1차 표시 지연(display_latencies)은 commit_metrics.display_metrics 와 같은 정렬(align_pairs·norm_word)로 짝지은 가설 단어 마지막 토큰의 emit_time − e_i.
"""
from typing import Dict, List, Optional, Sequence

from .commit_metrics import CHUNK_S, SEM_END_ID, align_pairs, emit_time, latency_stats, norm_word

_EPS = 1e-6


def valid_commits(events: Sequence[dict], sem_id: int = SEM_END_ID) -> List[dict]:
    """방출 순 <SEM_END> 이벤트 중 유효 commit(mid_word 아님, after ≥ 0)."""
    return [e for e in events if int(e["id"]) == int(sem_id) and not e.get("mid_word") and int(e["after"]) >= 0]


def build_segments(hyp: dict, delta: int, K: int, t_eos: float, *, sem_id: int = SEM_END_ID, min_seg_s: float = 0.8, min_words: int = 2) -> List[dict]:
    """1차 가설(hyp: words, events, word_k) → 세그먼트 목록. 각 세그먼트:
    dict(start, end (초), t_trig (최종 확정 시각), kind 'sem'|'eos', w0, w1 (1차 가설 단어 [w0, w1) — 이 세그먼트가 대신하는 1차 텍스트),
         k_sem (soft 만), held (이 세그먼트에 합쳐진 보류 트리거 수)).
    보류 트리거 수의 합·버린 t_eos 뒤 commit 수는 segment_stats 로 센다."""
    words, wk = hyp["words"], hyp.get("word_k")
    assert wk is not None and len(wk) == len(words), "hyp.word_k 가 필요하다(가설 단어별 [첫, 마지막] 토큰 청크 — semcommit_eval.backfill_word_k)"
    t_eos = float(t_eos); segs: List[dict] = []; start, w0, held = 0.0, 0, 0
    for ev in valid_commits(hyp["events"], sem_id):
        t_trig = emit_time(int(ev["k"]), K)
        if t_trig > t_eos + _EPS: continue                                                  # EOS 가 먼저 확정한다
        j = int(ev["after"])
        if j < w0: continue                                                                 # 이미 확정된 단어 뒤(방어 — sem-guard 로는 생기지 않는다)
        cut = min(max((int(wk[j][1]) - int(delta) + 1) * CHUNK_S, start), t_eos)
        if cut - start < min_seg_s - _EPS or (j + 1 - w0) < min_words or cut <= start + _EPS:
            held += 1; continue
        segs.append(dict(start=round(start, 6), end=round(cut, 6), t_trig=t_trig, kind="sem", w0=w0, w1=j + 1, k_sem=int(ev["k"]), held=held))
        start, w0, held = cut, j + 1, 0
    if w0 < len(words) or not words:                                                        # 남은 1차 단어가 있거나(또는 가설이 비었으면) EOS 가 확정
        if t_eos > start + _EPS:
            segs.append(dict(start=round(start, 6), end=round(t_eos, 6), t_trig=t_eos, kind="eos", w0=w0, w1=len(words), k_sem=None, held=held))
        elif segs:                                                                          # 오디오가 남지 않았다: 남은 단어는 마지막 세그먼트 몫
            segs[-1]["w1"] = len(words); segs[-1]["held"] += held
    return segs


def segment_stats(hyp: dict, segs: Sequence[dict], K: int, t_eos: float, sem_id: int = SEM_END_ID) -> dict:
    """세그먼트화 통계: 유효 commit 수, t_eos 뒤 commit(버림), 보류 수, soft/eos 세그먼트 수."""
    vc = valid_commits(hyp["events"], sem_id)
    late = sum(emit_time(int(e["k"]), K) > float(t_eos) + _EPS for e in vc)
    return dict(commits=len(vc), after_eos=late, held=sum(int(s["held"]) for s in segs), seg_sem=sum(s["kind"] == "sem" for s in segs),
                seg_eos=sum(s["kind"] == "eos" for s in segs), mid_word=sum(int(e["id"]) == int(sem_id) and bool(e.get("mid_word")) for e in hyp["events"]))


def compose(texts: Sequence[Optional[str]]) -> str:
    """세그먼트 2차 텍스트 → 최종 전사(순서대로, 빈 조각은 건너뛰고 공백 하나로 이음)."""
    return " ".join(t.strip() for t in texts if t and t.strip())


def final_latencies(ref_ends: Sequence[float], segs: Sequence[dict]) -> List[Optional[float]]:
    """참조 단어 끝 시각 → 최종 확정 지연(t_trig − e_i). e_i 가 든 세그먼트는 시간 기준 (start, end](첫 세그먼트는 [0, end]).
    세그먼트가 없거나 마지막 세그먼트 끝 뒤의 단어는 None(확정되지 않음 — EOS 세그먼트를 만들지 않은 꼬리)."""
    out: List[Optional[float]] = []
    for e in ref_ends:
        seg = next((s for s in segs if float(e) <= float(s["end"]) + _EPS), None)
        out.append(None if seg is None else round(float(seg["t_trig"]) - float(e), 6))
    return out


def unfinalized_tail(ref_ends: Sequence[float], segs: Sequence[dict]) -> List[float]:
    """마지막 세그먼트 끝 뒤에 끝나는 참조 단어(final_latencies 가 None) — 그 단어의 최종 시각은 1차 표시 시각으로 둔다(합성할 때)."""
    last = max((float(s["end"]) for s in segs), default=0.0)
    return [float(e) for e in ref_ends if float(e) > last + _EPS]


def display_latencies(ref_words: Sequence[str], ref_ends: Sequence[float], hyp_words: Sequence[str], word_k: Sequence[Sequence[int]], K: int) -> List[Optional[float]]:
    """1차 표시 지연: align_pairs(norm_word) 로 짝지은 가설 단어 마지막 토큰의 emit_time − e_i(삭제된 참조 단어는 None). commit_metrics.display_metrics 와 같은 짝."""
    R = [norm_word(w) for w in ref_words]; H = [norm_word(w) for w in hyp_words]; out: List[Optional[float]] = [None] * len(R)
    for i, j in align_pairs(R, H):
        if i is not None and j is not None: out[i] = round(emit_time(int(word_k[j][1]), K) - float(ref_ends[i]), 6)
    return out


def latency_summary(xs: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """None(확정 안 된·삭제된 단어)을 뺀 commit_metrics.latency_stats(n·mean·p50·p90·p95·p99·max) + 뺀 수(none)."""
    v = [float(x) for x in xs if x is not None]
    return dict(latency_stats(v), none=len(xs) - len(v))
