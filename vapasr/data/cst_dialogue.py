"""Stage 3 CST 대화 데이터셋 — 2 인 같은 언어 대화의 화자 귀속 + 스트리밍 ASR + 턴 끝 <EOT>
(wiki decision-cst-training-stages-and-speaker-slots, output-stage3-prep-plan-20261014 P3).

입력: Phase 2 정제 대화(<corpus>.refined.dialogues.jsonl, 정렬 토큰 내장)·mono 캐시·창 선택(DialogueWindowDataset 규칙 그대로:
      20–40 s, hop 10 s, 무음 시작, 전사 없는 음성은 마스크).
prefix(ast-v1 을 두 슬롯으로 넓힌 것):  <assistant 머리> language <SPK_A>{L} <SPK_B>{L}<asr_text><DELAY_d>
  같은 언어 두 슬롯은 창 안에서 먼저 말한 화자가 A(보조 규칙). 한 화자만 말하는 창도 두 슬롯을 선언한다.
출력(청크 k = ⌊t/80 ms⌋ + δ, Stage 1·2 와 같은 버킷 — semcommit_dataset.build_interleaved_events, 스트림 0 = A, 1 = B):
  전사 토큰 = 정렬 토큰 끝 시각. 화자 태그 <SPK_A>/<SPK_B> 는 첫 방출 앞과 화자가 바뀔 때만(Stage 2 의 구간 태그 규칙).
  <EOT> = episode(같은 화자 발화를 gap < 0.25 s 로 병합, dialogue.build_episodes) 끝마다 그 화자의 마지막 토큰 뒤에 두는 명시 청크 이벤트.
          k_min = ⌊(end + 0.24)/0.08⌋ → 실제 청크 = max(마지막 토큰 청크, k_min) (Phase 2 lane_alloc.eot_chunk 와 같은 위치).
          창 끝을 넘는 episode(말하는 중에 창이 끝남)와 창 안 토큰이 없는 episode 에는 <EOT> 를 넣지 않는다.
  flush(δ 때문에 창 끝을 넘긴 방출, <EMPTY_AUDIO> 라운드): build_interleaved_events 는 flush 에 화자 태그를 넣지 않으므로(단일 화자 시절 규약)
          여기서 화자가 바뀌는 곳에 태그를 다시 넣는다.
soft label(decision-eot-immediate-soft-label): 결과·p_end 는 대화 전체 시간축에서 정한다(eot_soft.assign_p_end — 창 밖 미래까지 보므로 Phase 2 의
  창 안 판정이 창 끝 근처 교대를 놓치던 것을 피한다). 결정 위치 = <EOT>, 단 그 <EOT> 를 내려고 화자 태그를 새로 연 경우는 그 태그 위치.
  결정 위치 label = 결정 토큰(가중 p_end) + <EOT> 바로 뒤 토큰(1 − p_end), 위치 가중 decision_weight(0 = 기본). <EOT> 는 입력에 항상 남는다.
  p_end None(미관측) → 결정 위치와 <EOT> 의 label −100.
마스크: 전사 없는 음성(quarantine·미정렬·결손)이 덮는 청크는 label −100 (DialogueWindowDataset.mask_chunks, Phase 2 규칙).
화자 수: max_speakers(기본 2) 보다 많은 화자가 말하는 창은 버린다. V2(decision-v2-run-scope-20261020)는 최대 4 — 슬롯 A·B·C·D(C·D = <SPK_3>·<SPK_4>),
  창 안에서 먼저 말한 순서. prefix 의 선언 슬롯 수는 max(2, 화자 수) — 학습은 declare_all_p 확률로 4 슬롯 모두(추론 때 화자 수를 모르면 최대를 선언),
  평가는 declare 로 고정할 수 있다. 번역 모드는 흐름 2N 개(슬롯 N = max(2, 화자 수) 전사 + 그 번역). 정체 회복 뒤집기는 2 슬롯 창만.
맞장구 선택(backchannel=(최대 길이 s, 최대 토큰 수), 기본 끔 = Phase 2 규칙): 그 이하의 episode 는 (1) 다른 화자의 결과 판정에서 '상대가 말함'으로
  세지 않고 (2) 자기 <EOT> 를 내지 않는다(전사는 그대로). 2026-10-14 측정: 끄면 결과의 24–50 % 가 both(0.5), 켜면(1 s·3 토큰) 11–32 %.
<SEM_END> 는 쓰지 않는다 → collate_cst 가 sem_only_mask 를 내 그 행의 <SEM_END> logit 만 softmax 에서 뺀다(중립). <EOT> 는 학습한다.
합성 대화(meta.mask_eot — sim_dialogue·dialogue_stitch): <EOT> 결정 label 을 모두 −100 으로 둔다(턴 타이밍이 인공적). 전사·화자 태그·화자 활동은 학습한다.
정체 회복 학습(recover, 기본 끔 — 2026-10-17 오류 분해: 교대 하나를 놓치거나 더 내면 뒤 턴들의 정체가 뒤집힌 채 회복하지 않는다):
  학습 창의 recover.p 비율에서 방출 원소 한 구간 [n0, n1) 의 문맥 슬롯을 뒤집어 태그를 다시 단다(retag) — 교대 지점에서 시작하면 '놓친 교대',
  run 안에서 시작하면 '헛 교대' 를 흉내 낸다. n0·n1 은 단어 첫 토큰(tokenizer 가 있으면 — 디코드가 공백으로 시작). 구간 안 출력 label 은 −100(틀린 문맥에서 이어 쓰는 법을 가르치지 않는다). 회복 지점 n1 만 정답을 둔다:
  n1 이 교대 지점이면 문맥 슬롯이 이미 그 화자라 태그 없이 이어 쓰고('boundary' — 목소리로 정체를 판단), run 안이면 바로잡는 태그를 낸다('inrun').
  평가·점검 경로(cst_sequence 기본)는 뒤집지 않는다 — __getitem__ 만 rng 를 넘긴다.
화자 슬롯 활동(spk_activity, 기본 끔 — 모델 add_spk_head 의 라벨): 청크마다 [슬롯 A 말함, 슬롯 B 말함] 0/1(slot_activity). 슬롯 순서는 직렬화와 같은
  window_speakers(먼저 말한 사람 = A), 말함 = episode(전사 없는 발화 포함, gap < 0.25 s 병합) 구간. collate_cst 가 activity (B,K,2)·activity_mask (B,K) 로 묶는다.
격리 발화 되살리기(revive, 기본 끔 — 2026-10-18 분석: 한국어 3 코퍼스는 화자 교대의 69–80 % 가 마스크 발화에 닿아 태그·<EOT> 감독이 빠진다):
  revive = {conv_id: {utt_id: [(tid, t), …]}}(load_revive — experiments/stage3_revive_tokens.py 가 정렬 결과에서 다시 뽑은, REVIVABLE 사유로만
  격리된 발화의 정렬 토큰, meta.missing 결손 조각은 오디오가 없어 뺀다). 창을 고른 뒤에 붙이므로 창 목록은 끈 것과 같다. 그 발화는 마스크 구간에서 빠지고(_untranscribed), 그 발화의 전사
  토큰 label 과 방출 구간(청크 ⌊start/0.08⌋+δ … ⌊end/0.08⌋+δ, 창 끝을 넘으면 flush)의 <NEXT_AUDIO> label 만 −100 이다 — 전사가 틀릴 수 있어
  입력 문맥으로만 둔다. 화자 태그·<EOT> 결정·겹친 상대 화자 전사는 학습한다. 평가·점검(cst_stage3_eval)은 끈 채로 쓴다.
교대·머무름 결정 가중(change_weight·hold_weight, 기본 0 = 끔 — V3-p1 C1, 2026-10-21 TAXI 분해: 오류 단어의 89 % 가 놓친 교대 뒤): decision_positions 가 고른
  교대 태그(다른 슬롯 태그 + 단어 첫 글자)와 <EOT> 뒤 머무름 결정(같은 슬롯 재개 태그 | 태그 없는 단어 첫 글자)의 pos_weight 를 max(기존, w) 로 올린다.
  켜면 시퀀스에 dec_kind(위치별 0 · 1 교대 · 2 머무름)·n_switch·n_hold 가 붙고(collate_cst → 모델 forward 통계만), 끄면 반환 키도 예전 그대로.
합성 창 EOT-문맥 dropout(eot_drop_synth, 기본 0 = 끔 — V3-p1 C5: 교차 언어 TAXI 디코드에서 V1 은 <EOT> 를 하나도 내지 않는데 합성 학습의 교대는 늘 <EOT> 뒤에 온다):
  meta.mask_eot 대화의 창을 그 확률로 골라 감독 없는 합성 <EOT>(label 원래 −100)를 문맥에서 뺀다(window_streams(4) drop_eot → ghost).
  그 <EOT> 가 나왔을 청크 c = max(k_min, 꼬리 토큰 청크) 가 창 안이면 c 를 닫는 <NEXT_AUDIO> label 을 −100(‘여기는 <EOT> 아님’을 가르치지 않게), flush 는 그대로.
  학습 rng 가 있을 때만 뽑는다(평가·점검은 drop_eot 로 직접 정한다).
"""
import json
import math
import random
import zlib
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from .dialogue import Dialogue, Episode, CHUNK_S, build_episodes, chunk_of, clamp_to_spans
from .dialogue_dataset import DialogueWindowDataset
from .dialogue_mix import crop_audio
from .eot_soft import assign_p_end, classify, P_END
from .interleave import Specials
from .lane_alloc import EOT_GAP_S
from .semcommit_dataset import build_interleaved_events
from ..uslm.mono_data import collate_streams

SLOTS = ("<SPK_A>", "<SPK_B>", "<SPK_3>", "<SPK_4>")                # 슬롯 A·B·C·D(V2 최대 4 화자 — C·D 는 Phase 2 예약 토큰, 모델 enable_slots). 2 화자 데이터셋은 앞 둘만 쓴다
SR = 16000
REVIVABLE = frozenset({"asr_disagree"})                         # 되살릴 격리 사유(시각·화자는 믿을 수 있고 전사만 의심) — implausible_rate·TN 사유는 그대로 마스크


def cst_head_text(langs: Sequence[str], tgts: Optional[Sequence[str]] = None) -> str:
    """슬롯 prefix 본문 'language <SPK_A>Korean <SPK_B>Korean[\\ntranslate <SPK_A>English <SPK_B>English]<asr_text>'
    (ast_dataset.ast_head_text 의 한 슬롯 형식을 슬롯 순서 A, B[, C, D] 로 넓힌 것 — 선언 슬롯 수 = len(langs))."""
    s = "language " + " ".join(f"{t}{l}" for t, l in zip(SLOTS, langs))
    if tgts: s += "\ntranslate " + " ".join(f"{t}{l}" for t, l in zip(SLOTS, tgts))
    return s + "<asr_text>"


def cst_head_text_xl(langs: Sequence[str], translate: bool = False) -> str:
    """교차 언어 창의 먼저 말한 순서 슬롯 prefix 본문(V3 R4 — xl_order 'arrival'): 'language English German[\ntranslate auto]<asr_text>'.
    두 언어를 이름순으로만 적고 슬롯에 묶지 않는다 — 어느 슬롯이 어느 언어인지(먼저 말한 화자 = <SPK_A>)와 번역 방향(각 슬롯을 상대 언어로)은 모델이 소리로 정한다."""
    return "language " + " ".join(sorted(set(langs))) + ("\ntranslate auto" if translate else "") + "<asr_text>"


def is_backchannel(ep: Episode, backchannel: Optional[Tuple[float, int]]) -> bool:
    return backchannel is not None and ep.end - ep.start <= backchannel[0] and len(ep.tokens) <= backchannel[1]


def dialogue_episodes(d: Dialogue, backchannel: Optional[Tuple[float, int]] = None) -> List[Episode]:
    """대화 전체의 episode 와 EOT 결과(p_end). 전사 없는 발화도 '말하고 있음'으로 결과 판정에 넣는다.
    backchannel 이 있으면 그 이하 episode 는 다른 화자의 판정에서 빼고, 자신은 outcome 'backchannel'(p_end None, <EOT> 없음)."""
    eps = build_episodes(d)
    if backchannel is None: assign_p_end(eps, d.observed())
    else:
        bc = [is_backchannel(e, backchannel) for e in eps]
        for e, b in zip(eps, bc):
            if b: e.outcome, e.p_end = "backchannel", None; continue
            e.outcome = classify(e, [o for o, ob in zip(eps, bc) if o.speaker == e.speaker or not ob], d.observed()); e.p_end = P_END[e.outcome]
    if d.meta.get("mask_eot"):                                  # 합성(stitch·sim): 턴 타이밍이 인공적 — EOT 감독 없음. <EOT> 는 문맥에 남고 결정 label 만 −100(맞장구 표시는 둔다)
        for e in eps:
            if e.outcome != "backchannel": e.outcome, e.p_end = "synthetic", None
    return eps


def speaker_firsts(eps: List[Episode], t0: float, L: float) -> List[Tuple[float, str]]:
    """창 [t0, t0+L) 에서 말하는 화자의 (창 안 첫 발화 시각, 화자) — 먼저 말한 순서(같으면 화자 ID)."""
    first: Dict[str, float] = {}
    for ep in eps:
        if ep.end > t0 and ep.start < t0 + L: first[ep.speaker] = min(first.get(ep.speaker, 1e18), max(ep.start, t0))
    return sorted(((v, k) for k, v in first.items()))


def window_speakers(eps: List[Episode], t0: float, L: float) -> List[str]:
    """창 [t0, t0+L) 에서 말하는 화자를 먼저 말한 순서로(창 안 첫 발화 시각, 같으면 화자 ID)."""
    return [s for _, s in speaker_firsts(eps, t0, L)]


def window_streams(eps: List[Episode], speakers: Sequence[str], t0: float, L: float, eot_id: int, drop_eot: bool = False, heads: Optional[set] = None):
    """episode → 슬롯별 이벤트열(창 기준 시각). 원소 = (tid, t) 전사 토큰 | (eot_id, t_end, k_min) 명시 청크 이벤트.
    → (streams, {(슬롯, 원소 index): p_end}, 통계). drop_eot(C5): 합성 episode(outcome 'synthetic' — meta.mask_eot 라 p_end 가 없다)의 <EOT> 원소를 넣지 않고
    ghost (슬롯, k_min, 꼬리 [(흐름, index)]) 로 남긴다 — 통계 eot 는 그대로 세고 eot_dropped 를 더한다 → (streams, eot_p, 통계, ghosts)(build_interleaved_events with_src 와 같은 관례).
    heads(set 을 주면 채운다 — C1): episode 마다 창 안 첫 전사 원소 (슬롯, index). 반환은 그대로다."""
    t1 = t0 + L; streams: List[list] = [[] for _ in speakers]; eot_p: Dict[Tuple[int, int], Optional[float]] = {}; st = dict(eot=0, eot_truncated=0, eot_no_tokens=0, eot_backchannel=0)
    slot = {s: i for i, s in enumerate(speakers)}; ghosts: List[tuple] = []
    if drop_eot: st["eot_dropped"] = 0
    for ep in eps:
        if ep.end <= t0 or ep.start >= t1 or ep.speaker not in slot: continue
        s = slot[ep.speaker]; toks = [(int(tid), float(t) - t0) for tid, t in ep.tokens if t0 <= t <= t1]
        if heads is not None and toks: heads.add((s, len(streams[s])))
        streams[s] += toks
        if ep.end > t1 + 1e-9: st["eot_truncated"] += 1; continue
        if not toks: st["eot_no_tokens"] += 1; continue
        if ep.outcome == "backchannel": st["eot_backchannel"] += 1; continue
        e = ep.end - t0
        if drop_eot and ep.outcome == "synthetic": ghosts.append((s, chunk_of(e + EOT_GAP_S), [(s, len(streams[s]) - 1)])); st["eot"] += 1; st["eot_dropped"] += 1; continue
        streams[s].append((int(eot_id), e, chunk_of(e + EOT_GAP_S))); eot_p[(s, len(streams[s]) - 1)] = ep.p_end; st["eot"] += 1
    return (streams, eot_p, st, ghosts) if drop_eot else (streams, eot_p, st)


def slot_activity(eps: List[Episode], speakers: Sequence[str], t0: float, L: float, K: int, n_slots: int = 2) -> List[List[int]]:
    """(K, n_slots) 0/1 — 창 기준 청크 [k·0.08, (k+1)·0.08) 가 슬롯 s 화자의 episode 와 겹치면 1(겹침 구간은 함께 1). 슬롯 = speakers 순서, 화자 없는 슬롯은 0."""
    act = [[0] * n_slots for _ in range(K)]; slot = {s: i for i, s in enumerate(speakers)}
    for ep in eps:
        if ep.speaker not in slot or ep.end <= t0 or ep.start >= t0 + L: continue
        for k in range(max(0, int((ep.start - t0) / CHUNK_S)), min(K, int(math.ceil((ep.end - t0) / CHUNK_S)))): act[k][slot[ep.speaker]] = 1
    return act


def load_revive(paths: Sequence[str]) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """stage3_revive_tokens.py 산출(<corpus>.revive.jsonl, 줄 = {"conv_id", "utts": {utt_id: [[tid, t], …]}}) → {conv_id: {utt_id: [(tid, t), …]}}."""
    out: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); out.setdefault(r["conv_id"], {}).update({u: [(int(t), float(e)) for t, e in v] for u, v in r["utts"].items()})
    return out


def revived_elements(streams: List[list], speakers: Sequence[str], spans: Sequence[Tuple[str, float, float]], t0: float) -> set:
    """되살린 발화(spans = [(화자, start, end)], 대화 시각)의 전사 원소 (슬롯, index). build_episodes 가 토큰 시각을 발화 구간으로 자르고
    같은 화자의 발화는 겹치지 않으므로, 그 화자 슬롯에서 끝 시각이 [start, end] 에 드는 전사 토큰이 그 발화의 것이다."""
    out = set()
    for s, who in enumerate(speakers):
        sp = [(a - t0 - 1e-6, b - t0 + 1e-6) for w, a, b in spans if w == who]
        if sp: out.update((s, j) for j, x in enumerate(streams[s]) if len(x) == 2 and any(a <= x[1] <= b for a, b in sp))
    return out


def revived_chunks(spans: Sequence[Tuple[str, float, float]], speakers: Sequence[str], t0: float, K: int, delay: int) -> set:
    """되살린 발화의 방출 청크 [⌊start/0.08⌋+δ, ⌊end/0.08⌋+δ](창 기준, K 이상은 flush = K) — 그 안의 <NEXT_AUDIO> label 을 −100 으로 둔다."""
    out = set()
    for w, a, b in spans:
        if w in speakers: out.update(range(min(chunk_of(max(0.0, a - t0), delay), K), min(chunk_of(max(0.0, b - t0), delay), K) + 1))
    return out


def tag_flush(chunks: List[tuple], srcs: List[list], K: int, sp: Specials) -> None:
    """flush 청크(k ≥ K)에 화자 태그를 다시 넣는다(제자리 수정): 직전 실제 청크까지의 마지막 화자와 다른 스트림의 원소 앞에 그 스트림 태그."""
    last = None
    for (k, emits), src in zip(chunks, srcs):
        if k >= K: break
        for e, q in zip(emits, src):
            if q is None and e in sp.spk: last = sp.spk.index(e)
    for c, ((k, emits), src) in enumerate(zip(chunks, srcs)):
        if k < K: continue
        ne, ns = [], []
        for e, q in zip(emits, src):
            if q is not None and q[0] != last: ne.append(sp.spk[q[0]]); ns.append(None); last = q[0]
            ne.append(e); ns.append(q)
        chunks[c] = (k, ne); srcs[c] = ns


class Recover(NamedTuple):
    """정체 회복 학습 설정(모듈 설명). p: 창마다 뒤집을 확률, boundary_frac: 회복을 교대 경계에 둘 비율(나머지 run 안),
    max_lag: run 안 회복의 최대 거리(원소 수), max_boundaries: 경계 회복을 몇 번째 교대 경계까지 고를지."""
    p: float = 0.0
    boundary_frac: float = 0.5
    max_lag: int = 8
    max_boundaries: int = 2


def pick_flip(seq: Sequence[Tuple[int, bool, bool]], rng, rc: Recover) -> Optional[Tuple[int, int, str]]:
    """seq: 실제 청크 방출 원소 [(슬롯, 텍스트 토큰인가, 쓸 수 있나(마스크 청크 밖))](방출 순) → (n0, n1, '시작/끝') | None.
    끝 종류를 먼저 고른다(boundary_frac) — 교대 경계가 드문 창(한국어 긴 턴·마스크)에서도 경계 회복 비율이 설정을 따르게. 그 종류로 회복할 수 있는 시작만 후보로 둔다.
    시작 n0: 'missed'(교대 지점 — 그 태그를 빼 직전 화자로 이어 쓴 것처럼) | 'spurious'(run 안 — 헛 태그를 넣은 것처럼), 가능한 쪽에서 반반.
    끝 n1: 'boundary'(n0 뒤 1–max_boundaries 번째 교대 지점) | 'inrun'(n0 뒤 max_lag 안의 run 안 원소). 고른 끝 종류가 창 어디에도 안 되면 다른 종류.
    n0·n1 은 텍스트 토큰이고 마스크 청크 밖."""
    import bisect
    n = len(seq); start = [i == 0 or seq[i][0] != seq[i - 1][0] for i in range(n)]; chg = [j for j in range(1, n) if start[j]]
    good = lambda i: seq[i][1] and seq[i][2]
    def ends(n0: int, end: str) -> List[int]:
        if end == "boundary": k = bisect.bisect_right(chg, n0); return [i for i in chg[k:k + rc.max_boundaries] if good(i)]
        return [i for i in range(n0 + 1, min(n, n0 + 1 + rc.max_lag)) if not start[i] and good(i)]
    first = "boundary" if rng.random() < rc.boundary_frac else "inrun"
    for end in (first, "inrun" if first == "boundary" else "boundary"):
        cand = {"missed": [i for i in chg if good(i) and ends(i, end)], "spurious": [i for i in range(1, n) if not start[i] and good(i) and ends(i, end)]}
        kinds = [k for k in ("missed", "spurious") if cand[k]]
        if kinds: break
    else: return None
    kind = rng.choice(kinds); n0 = rng.choice(cand[kind])
    return n0, rng.choice(ends(n0, end)), f"{kind}/{end}"


def retag(chunks: List[tuple], srcs: List[list], flip: set, sp: Specials) -> Tuple[List[tuple], List[list]]:
    """화자 태그를 다시 단다: 원소의 문맥 슬롯 = 자기 스트림, flip 의 (스트림, index) 는 반대 슬롯. 태그는 첫 방출 앞과 문맥 슬롯이 바뀔 때만
    (build_interleaved_events + tag_flush 와 같은 규칙 — flip 이 비면 원래 시퀀스와 같다)."""
    cur = None; oc: List[tuple] = []; os_: List[list] = []
    for (k, emits), src in zip(chunks, srcs):
        ne, ns = [], []
        for e, q in zip(emits, src):
            if q is None and e in sp.spk: continue
            if q is not None:
                c = 1 - q[0] if (q[0], q[1]) in flip else q[0]
                if c != cur: ne.append(sp.spk[c]); ns.append(None); cur = c
            ne.append(e); ns.append(q)
        oc.append((k, ne)); os_.append(ns)
    return oc, os_


def flatten_cst(chunks: List[tuple], srcs: List[list], K: int, prefix: List[int], audio_pad: int, sp: Specials):
    """build_mono_sequence(M=0) 와 같은 배치 + 위치별 출처. → (ids, is_input, chunk_of, src_at, chunk_at, n_rounds, n_flush).
    청크 k < K: audio_pad 다음 방출(끝이 <NEXT_AUDIO>). flush: <EMPTY_AUDIO> + flush 방출 전부 + <NEXT_AUDIO>, 끝에 <EMPTY_AUDIO><NEXT_AUDIO>.
    chunk_at = 방출 원소의 청크(flush 는 K), 그 외 −1."""
    ids = list(prefix); is_input = [True] * len(prefix); cof = [-1] * len(prefix); src_at: List = [None] * len(prefix); chunk_at = [-1] * len(prefix); flush = []
    for (k, emits), src in zip(chunks, srcs):
        if k < K:
            ids.append(audio_pad); is_input.append(True); cof.append(k); src_at.append(None); chunk_at.append(-1)
            ids += emits; is_input += [False] * len(emits); cof += [-1] * len(emits); src_at += list(src); chunk_at += [k] * len(emits)
        else:
            flush += [(e, q) for e, q in zip(emits, src) if e != sp.empty_audio]
    rounds = ([flush] if flush else []) + [[]]
    for r in rounds:
        ids.append(sp.empty_audio); is_input.append(True); cof.append(-1); src_at.append(None); chunk_at.append(-1)
        for e, q in r: ids.append(e); is_input.append(False); cof.append(-1); src_at.append(q); chunk_at.append(K)
        ids.append(sp.next_audio); is_input.append(False); cof.append(-1); src_at.append(None); chunk_at.append(K)
    return ids, is_input, cof, src_at, chunk_at, len(rounds), len(flush)


def decision_positions(ids: Sequence[int], is_input: Sequence[bool], labels: Sequence[int], labels_alt: Sequence[int], src_at: Sequence, spk_ids: Sequence[int],
                       trans_id: Optional[int], eot_id: int, word_start=None, heads=()) -> Tuple[List[int], List[int]]:
    """교대·머무름 결정 위치(C1, 순수 함수). 문맥 슬롯 cur 을 시퀀스를 따라 추적한다 — 첫 출력 위치부터 모든 슬롯 태그가(label −100 이든 아니든) cur 을 바꾸고
    (prefix 의 슬롯 선언 'language <SPK_A>{L} …' 은 문맥이 아니다), <TRANS> 는 구간(전사·번역)만 바꾼다. 글자 토큰 = 출처(src_at)가 있는 출력 원소 중 <EOT> 가 아닌 것.
    switch p: labels[p] == ids[p] 인 출력 슬롯 태그, cur 이 있고 그 슬롯과 다르다, labels_alt[p] == −100, 다음 위치가 단어 첫 글자 토큰
      → 그 <EOT> 만을 위해 연 태그(<SPK_x><EOT>), <SPK_x><TRANS>, 첫 태그, 단어 중간 겹침 churn 이 빠진다.
    hold p: 출력 <EOT>(문맥 슬롯 s) 뒤 첫 출력 결정 토큰(<NEXT_AUDIO>·오디오·<EMPTY_AUDIO> 입력은 건너뛴다)이 같은 슬롯 <SPK_s>(번역 모드 재개) 이거나
      cur == s 인 단어 첫 글자 토큰(받아쓰기 모드 — 태그 없이 계속)이고 labels[p] != −100, labels_alt[p] == −100.
    word_start(tid) → bool, None 이면 글자 토큰을 모두 단어 첫 토큰으로 본다. heads: episode 첫 전사 원소 (흐름, index) — 앞 공백 없이 시작하는 단위
    (words 행 합성 발화의 첫 토큰)도 턴 시작이면 단어 첫 토큰으로 본다(window_streams(4) heads). → (switch 위치, hold 위치)."""
    N = len(ids); slot = {int(t): k for k, t in enumerate(spk_ids)}; ws0 = word_start or (lambda t: True); hd = set(map(tuple, heads))
    wsp = lambda q: ws0(int(ids[q])) or (src_at[q] is not None and tuple(src_at[q][:2]) in hd)
    text = lambda q: not is_input[q] and src_at[q] is not None and ids[q] != eot_id
    switch: List[int] = []; hold: List[int] = []; cur = None; started = False; after = None             # after = 아직 첫 결정을 못 본 직전 출력 <EOT> 의 슬롯
    for p in range(N):
        t = ids[p]; started = started or not is_input[p]
        if not started: continue
        if t in slot:
            s = slot[t]
            if not is_input[p]:
                if after is not None and s == after and labels[p] != -100 and labels_alt[p] == -100: hold.append(p)
                after = None
                if labels[p] == t and cur is not None and cur != s and labels_alt[p] == -100 and p + 1 < N and text(p + 1) and wsp(p + 1): switch.append(p)
            cur = s; continue
        if is_input[p] or (src_at[p] is None and t != trans_id): continue                         # 오디오·<EMPTY_AUDIO> 입력, <NEXT_AUDIO> — 결정 토큰이 아니다
        if t == eot_id: after = cur; continue
        if after is not None and t != trans_id and cur == after and wsp(p) and labels[p] != -100 and labels_alt[p] == -100: hold.append(p)
        after = None
    return switch, hold


def weight_decisions(s: dict, sp: Specials, trans_id: Optional[int], eot_id: int, word_start, change_weight: float, hold_weight: float, heads=()) -> dict:
    """build_cst(4)_sequence 결과 s 의 pos_weight 를 교대·머무름 위치에서 max(기존, w) 로 올린다(제자리, w > 0 인 쪽만 — <EOT> decision_weight 위치는 그대로)
    → 덧붙일 키 dict(dec_kind (L,) 0 · 1 교대 · 2 머무름, n_switch, n_hold). 둘 다 0 이면 아무것도 안 하고 {}(반환 키가 예전 그대로)."""
    if not (change_weight > 0 or hold_weight > 0): return {}
    sw, ho = decision_positions(s["ids"], s["is_input"], s["labels"], s["labels_alt"], s["src_at"], sp.spk, trans_id, eot_id, word_start, heads)
    dk = [0] * len(s["ids"]); pw = s["pos_weight"]
    for ps, k, w in ((sw, 1, change_weight), (ho, 2, hold_weight)):
        for p in ps:
            dk[p] = k
            if w > 0: pw[p] = max(pw[p], float(w))
    return dict(dec_kind=dk, n_switch=len(sw), n_hold=len(ho), dec_heads=sorted(set(map(tuple, heads))))


def ghost_next(eot_ghost, src_at: Sequence, chunk_at: Sequence[int], ids: Sequence[int], is_input: Sequence[bool], K: int, sp: Specials) -> List[int]:
    """뺀 합성 <EOT>(ghost (슬롯, k_min, 꼬리))가 나왔을 청크 c = max(k_min, 꼬리 원소의 방출 청크 — flush 는 K) → c < K 이면 c 를 닫는 <NEXT_AUDIO> 위치(flush 는 없음)."""
    at = {(q[0], q[1]): p for p, q in enumerate(src_at) if q is not None}
    nx = {chunk_at[p]: p for p in range(len(ids)) if ids[p] == sp.next_audio and not is_input[p] and src_at[p] is None and 0 <= chunk_at[p] < K}
    out = []
    for _, k_min, tails in eot_ghost:
        c = max([int(k_min)] + [chunk_at[at[t]] for t in tails if t in at])
        if c < K: out.append(nx[c])
    return out


def event_id(streams: List[list]) -> int:
    """명시 청크 이벤트(<EOT>)의 토큰 id(이벤트가 없으면 −1) — decision_positions 가 <EOT> 를 글자와 가른다."""
    return next((int(x[0]) for ev in streams for x in ev if len(x) > 2), -1)


def build_cst_sequence(streams: List[list], eot_p: Dict[Tuple[int, int], Optional[float]], K: int, prefix: List[int], audio_pad: int, sp: Specials,
                       delay: int, mask_chunks=(), decision_weight: float = 0.0, recover: Optional[Recover] = None, rng=None, word_start=None,
                       lex_mask=(), lex_chunks=(), change_weight: float = 0.0, hold_weight: float = 0.0, eot_ghost=(), dec_heads=()) -> dict:
    """슬롯 이벤트열 → 학습 시퀀스(오디오와 무관한 순수 함수; recover·rng 를 주면 그 확률로 정체 회복 구간을 넣는다 — 모듈 설명).
    word_start(tid) → bool: 주면 뒤집기 시작·회복 지점을 단어 첫 토큰으로 제한한다(실제 오류는 단어 경계에서 난다 — 단어 중간 태그를 가르치지 않게).
    lex_mask: 전사 label 만 −100 인 원소 (슬롯, index), lex_chunks: <NEXT_AUDIO> label 을 −100 으로 둘 청크(K = flush) — 되살린 발화(revive, 모듈 설명).
    change_weight·hold_weight(C1, > 0 이면): 교대·머무름 결정 위치(decision_positions, 단어 첫 토큰은 word_start 또는 dec_heads — episode 첫 전사 원소) 가중
      + dec_kind·n_switch·n_hold·dec_heads 키.
    eot_ghost(C5, window_streams drop_eot 의 ghosts): 뺀 <EOT> 가 나왔을 창 안 청크를 닫는 <NEXT_AUDIO> label −100(ghost_next) + eot_ghost·ghost_pos 키.
    → dict(ids, is_input, chunk_of, labels, pos_weight, labels_alt, soft_w, soft_pos, eot_pos, src_at, chunk_at, chunks, srcs, st, n_rounds, n_flush, n_eot, n_soft,
           flip(None | dict(kind, keys, p_start, p_end)), lex[, dec_kind, n_switch, n_hold][, eot_ghost, ghost_pos])."""
    streams = [list(s) for s in streams] + [[] for _ in range(2 - len(streams))]
    chunks, st, srcs = build_interleaved_events(streams, (K - 0.5) * CHUNK_S, sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=0,
                                                add_spk_tags=True, with_src=True)
    tag_flush(chunks, srcs, K, sp)
    mask = set(mask_chunks); lex = set(lex_mask); flip = None
    if recover is not None and rng is not None and recover.p > 0 and len(streams) == 2 and rng.random() < recover.p:     # 뒤집기(1 − 슬롯)는 2 슬롯 창만
        E = [q for (k, _), src in zip(chunks, srcs) if k < K for q in src if q is not None]
        ck = {q: k for (k, _), src in zip(chunks, srcs) for q in src if q is not None}
        txt = lambda q: len(streams[q[0]][q[1]]) == 2 and (word_start is None or word_start(int(streams[q[0]][q[1]][0])))
        pick = pick_flip([(q[0], txt(q), ck[q] not in mask and (q[0], q[1]) not in lex) for q in E], rng, recover)
        if pick:
            n0, n1, kind = pick; keys = {(q[0], q[1]) for q in E[n0:n1]}
            chunks, srcs = retag(chunks, srcs, keys, sp); flip = dict(kind=kind, keys=keys, n0=tuple(E[n0]), n1=tuple(E[n1]))
    ids, is_input, cof, src_at, chunk_at, n_rounds, n_flush = flatten_cst(chunks, srcs, K, prefix, audio_pad, sp)
    N = len(ids)
    labels = [(-100 if inp or (0 <= c < K and c in mask) else t) for t, inp, c in zip(ids, is_input, chunk_at)]
    if lex or lex_chunks:                                                                       # 되살린 발화: 전사와 방출 구간 <NEXT_AUDIO> 만 −100(태그·<EOT> 는 학습)
        lk = set(lex_chunks)
        for p, q in enumerate(src_at):
            if not is_input[p] and ((q is not None and (q[0], q[1]) in lex) or (q is None and ids[p] == sp.next_audio and chunk_at[p] in lk)): labels[p] = -100
    if flip:                                                                                  # 구간 [첫 원소(앞 헛 태그 포함), 회복 지점의 첫 출력) 의 출력 label −100
        at = {(q[0], q[1]): p for p, q in enumerate(src_at) if q is not None}
        tag_before = lambda p: p > 0 and src_at[p - 1] is None and not is_input[p - 1] and ids[p - 1] in sp.spk
        a0, a1 = at[flip["n0"]], at[flip["n1"]]; a0 -= tag_before(a0); a1 -= tag_before(a1)
        for p in range(a0, a1):
            if not is_input[p]: labels[p] = -100
        flip.update(p_start=a0, p_end=a1)
    gh = ghost_next(eot_ghost, src_at, chunk_at, ids, is_input, K, sp) if eot_ghost else []
    for p in gh: labels[p] = -100                                                               # 뺀 합성 <EOT> 의 라운드: '여기는 <EOT> 아님'을 가르치지 않는다
    pos_weight = [0.0] * N; labels_alt = [-100] * N; soft_w = [1.0] * N; soft_pos: List[int] = []; eot_pos: List[int] = []
    for p, q in enumerate(src_at):
        if q is None or (q[0], q[1]) not in eot_p: continue
        eot_pos.append(p)
        if labels[p] == -100: continue                                                          # 마스크 청크
        s = q[0]; nxt = src_at[p + 1] if p + 1 < N else None
        opened = ids[p - 1] == sp.spk[s] and src_at[p - 1] is None and not (nxt is not None and nxt[0] == s)
        d = p - 1 if opened else p; pe = eot_p[(s, q[1])]
        if pe is None: labels[d] = labels[p] = -100; continue                                   # 미관측(대화 끝·결손 근처)
        alt = ids[p + 1]; assert alt != ids[d], f"soft 대안이 결정 토큰과 같다: pos {d}"
        labels_alt[d] = alt; soft_w[d] = float(pe); soft_pos.append(d)
        if decision_weight: pos_weight[d] = float(decision_weight)
    out = dict(ids=ids, is_input=is_input, chunk_of=cof, labels=labels, pos_weight=pos_weight, labels_alt=labels_alt, soft_w=soft_w, soft_pos=soft_pos,
               eot_pos=eot_pos, src_at=src_at, chunk_at=chunk_at, chunks=chunks, srcs=srcs, st=st, n_rounds=n_rounds, n_flush=n_flush, n_eot=len(eot_pos), n_soft=len(soft_pos),
               flip=flip, lex=lex, **(dict(eot_ghost=list(eot_ghost), ghost_pos=gh) if eot_ghost else {}))
    out.update(weight_decisions(out, sp, None, event_id(streams), word_start, change_weight, hold_weight, dec_heads))
    return out


def read_back(ids: Sequence[int], is_input: Sequence[bool], sp: Specials, eot_id: int) -> Tuple[List[List[int]], List[str]]:
    """출력 토큰열(입력 위치·<NEXT_AUDIO> 제외)을 태그 규칙으로 다시 읽는다 → (슬롯별 토큰열(<EOT> 포함), 위반 목록).
    위반: 'untagged'(태그 없이 시작) · 'dup_tag'(같은 태그 연속, 그 사이에 방출 없음)."""
    out: List[List[int]] = [[] for _ in sp.spk]; cur = None; bad: List[str] = []; since = 0
    for t, inp in zip(ids, is_input):
        if inp or t == sp.next_audio: continue
        if t in sp.spk:
            s = sp.spk.index(t)
            if s == cur and since == 0: bad.append("dup_tag")
            cur = s; since = 0; continue
        if cur is None: bad.append("untagged"); continue
        out[cur].append(int(t)); since += 1
    return out, bad


# ── Stage 4: 두 화자 × (전사·번역) 네 흐름 (decision-stage4-prep-20261019, decision-ast-monotonic-data 9)
def window_streams4(eps: List[Episode], speakers: Sequence[str], t0: float, L: float, eot_id: int, trans_of, drop_eot: bool = False,
                    heads: Optional[set] = None) -> Tuple[List[list], Dict[Tuple[int, int], Optional[float]], dict]:
    """window_streams + 번역 흐름: N = max(2, 화자 수) — 흐름 0..N−1 = 슬롯 전사, N..2N−1 = 그 번역(원소 (tid, t), 창 기준 시각). 2 화자면 0·1 = A·B 전사, 2·3 = 번역.
    trans_of(ep) → (토큰 [(tid, t 대화 시각, 입력만)], 번역 없는 행 구간 [(a, b, 사유)], 마지막 행에 번역이 있나).
    번역 시각은 trans_of 가 행 조각 구간으로 자른 것을 다시 episode 구간으로 자른다. <EOT> 는 그 episode 의 마지막 행에 번역이 있으면 번역 흐름,
    없으면 전사 흐름(Stage 3 와 같다)에 두고, 어느 쪽이든 전사·번역 꼬리 중 나중 것 뒤에 오도록 그 꼬리 원소를 eot_tail 로 남긴다
    (build_interleaved_events 는 같은 흐름의 앞 원소만 기다린다 — build_cst4_sequence 가 eot_after_tails 로 옮긴다). 청크는 Stage 3 규칙(말이 끝난 때) 그대로다.
    drop_eot(C5): 합성 episode 의 <EOT> 를 넣지 않고 정보에 ghost [(슬롯, k_min, 전사·번역 꼬리)], 통계에 eot_dropped(window_streams 와 같은 규칙).
    heads: window_streams 와 같다(전사 흐름의 episode 첫 원소).
    → (streams, {(흐름, index): p_end}, 정보 dict(tr_input, miss, miss_eot, eot_tail, st[, ghost]))."""
    N = max(2, len(speakers)); t1 = t0 + L; streams: List[list] = [[] for _ in range(2 * N)]; eot_p: Dict[Tuple[int, int], Optional[float]] = {}
    tr_input, miss, miss_eot, eot_tail = set(), [], set(), {}; st = dict(eot=0, eot_truncated=0, eot_no_tokens=0, eot_backchannel=0, tr_tokens=0, tr_input=0)
    slot = {s: i for i, s in enumerate(speakers)}; ghosts: List[tuple] = []
    if drop_eot: st["eot_dropped"] = 0
    for ep in eps:
        if ep.end <= t0 or ep.start >= t1 or ep.speaker not in slot: continue
        s = slot[ep.speaker]; toks = [(int(tid), float(t) - t0) for tid, t in ep.tokens if t0 <= t <= t1]
        if heads is not None and toks: heads.add((s, len(streams[s])))
        streams[s] += toks
        tr, gaps, last_ok = trans_of(ep); n_tr = 0
        for tid, t, inp in sorted(tr, key=lambda x: x[1]):
            t = min(max(float(t), ep.start), ep.end)
            if not t0 <= t <= t1: continue
            if inp: tr_input.add((N + s, len(streams[N + s]))); st["tr_input"] += 1
            streams[N + s].append((int(tid), t - t0)); n_tr += 1; st["tr_tokens"] += 1
        for a, b, why in gaps:
            if b > t0 and a < t1: miss.append((ep.speaker, a, b)); st[f"miss_{why}"] = st.get(f"miss_{why}", 0) + 1
        if ep.end > t1 + 1e-9: st["eot_truncated"] += 1; continue
        if not toks: st["eot_no_tokens"] += 1; continue
        if ep.outcome == "backchannel": st["eot_backchannel"] += 1; continue
        e = ep.end - t0; w = N + s if n_tr and last_ok else s; tails = [(s, len(streams[s]) - 1)] + ([(N + s, len(streams[N + s]) - 1)] if n_tr else [])
        if drop_eot and ep.outcome == "synthetic": ghosts.append((s, chunk_of(e + EOT_GAP_S), tails)); st["eot"] += 1; st["eot_dropped"] += 1; continue
        streams[w].append((int(eot_id), e, chunk_of(e + EOT_GAP_S))); key = (w, len(streams[w]) - 1); eot_p[key] = ep.p_end; eot_tail[key] = tails; st["eot"] += 1
        if not last_ok: miss_eot.add(key)
    return streams, eot_p, dict(tr_input=tr_input, miss=miss, miss_eot=miss_eot, eot_tail=eot_tail, st=st, **({"ghost": ghosts} if drop_eot else {}))


def eot_after_tails(chunks: List[tuple], srcs: List[list], eot_tail: Dict[Tuple[int, int], list]) -> int:
    """<EOT>(명시 이벤트)가 그 episode 의 전사·번역 꼬리(eot_tail) 중 나중 것보다 먼저 나왔으면 그 꼬리 바로 뒤로 옮긴다(제자리 수정) → 옮긴 수.
    꼬리는 <EOT> 의 k_min 이후 청크에 있으므로 청크 규칙(말이 끝난 때)은 그대로고, 다음 episode 원소는 시각이 더 늦어 흐름 안 순서도 그대로다."""
    n = 0
    for key, tails in eot_tail.items():
        pos = {q: (c, e) for c, src in enumerate(srcs) for e, q in enumerate(src) if q is not None}
        tp = max((pos[t] for t in tails if t in pos), default=None)
        if key not in pos or tp is None or tp < pos[key]: continue
        c0, e0 = pos[key]; tid = chunks[c0][1].pop(e0); srcs[c0].pop(e0); c1, e1 = tp
        if c1 == c0 and e1 > e0: e1 -= 1
        chunks[c1][1].insert(e1 + 1, tid); srcs[c1].insert(e1 + 1, key); n += 1
    return n


def tag_cst4(chunks: List[tuple], srcs: List[list], streams: List[list], sp: Specials, trans_id: int) -> Tuple[List[tuple], List[list]]:
    """build_interleaved_events(add_spk_tags=False) 결과(flush 포함)에 태그를 단다(흐름 2N 개 — N = len(streams) // 2). 구간(슬롯, 전사|번역)이 바뀔 때만:
    전사 → <SPK_x>, 같은 슬롯 전사 바로 뒤의 번역 → <TRANS>, 그 밖의 번역 → <SPK_x><TRANS>(첫 방출 앞도 마찬가지).
    <EOT>(명시 이벤트)는 화자 수준 — 문맥 슬롯이 같으면 구간(전사·번역)과 상관없이 태그 없이, 다르면 <SPK_x> 를 열고 낸다."""
    N = len(streams) // 2; cur = None; oc: List[tuple] = []; os_: List[list] = []
    for (k, emits), src in zip(chunks, srcs):
        ne, ns = [], []
        for e, q in zip(emits, src):
            if q is not None:
                s = q[0] % N
                if len(streams[q[0]][q[1]]) > 2:
                    if cur is None or cur[0] != s: ne.append(sp.spk[s]); ns.append(None); cur = (s, 0)
                else:
                    kind = int(q[0] >= N)
                    if cur != (s, kind):
                        tags = [sp.spk[s]] if kind == 0 else [trans_id] if cur == (s, 0) else [sp.spk[s], trans_id]
                        ne += tags; ns += [None] * len(tags); cur = (s, kind)
            ne.append(e); ns.append(q)
        oc.append((k, ne)); os_.append(ns)
    return oc, os_


def build_cst4_sequence(streams: List[list], eot_p: Dict[Tuple[int, int], Optional[float]], K: int, prefix: List[int], audio_pad: int, sp: Specials,
                        trans_id: int, delay: int, mask_chunks=(), decision_weight: float = 0.0, lex_mask=(), lex_chunks=(), tr_input=(),
                        miss_chunks=(), miss_eot=(), eot_tail: Optional[Dict[Tuple[int, int], list]] = None, word_start=None, change_weight: float = 0.0,
                        hold_weight: float = 0.0, eot_ghost=(), dec_heads=()) -> dict:
    """전사·번역 흐름 2N 개(window_streams4, 2 화자면 네 흐름) → 학습 시퀀스(오디오와 무관한 순수 함수). build_cst_sequence 와 같은 반환 키 + n_tr·eot_moved.
    label −100: 입력, 마스크 청크(전사 없는 음성), 되살린 전사(lex_mask)와 그 방출 청크 <NEXT_AUDIO>(lex_chunks),
      입력 문맥으로만 둘 번역(tr_input — 되살린 발화의 번역)과 그것을 여는 태그, 번역 없는 행의 방출 청크(miss_chunks)의 태그·<NEXT_AUDIO>·<EOT>,
      마지막 발화에 번역이 없는 episode 의 <EOT>(miss_eot). <EOT> soft label 은 Stage 3 규칙 그대로(태그를 그 <EOT> 만을 위해 열었으면 태그 위치).
    eot_tail(window_streams4): <EOT> 를 그 episode 의 전사·번역 꼬리 뒤로 옮긴다(eot_after_tails).
    change_weight·hold_weight·eot_ghost·dec_heads: build_cst_sequence 와 같다(word_start 는 교대·머무름의 단어 첫 토큰 판정에만 쓴다)."""
    NS = max(2, (len(streams) + 1) // 2); streams = [list(s) for s in streams] + [[] for _ in range(2 * NS - len(streams))]      # 슬롯 수 NS, 흐름 2·NS
    chunks, st, srcs = build_interleaved_events(streams, (K - 0.5) * CHUNK_S, sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=0,
                                                add_spk_tags=False, with_src=True)
    n_moved = eot_after_tails(chunks, srcs, eot_tail or {})
    chunks, srcs = tag_cst4(chunks, srcs, streams, sp, trans_id)
    ids, is_input, cof, src_at, chunk_at, n_rounds, n_flush = flatten_cst(chunks, srcs, K, prefix, audio_pad, sp)
    N = len(ids); mask = set(mask_chunks); lex = set(lex_mask); lk = set(lex_chunks); tri = set(tr_input); mk = set(miss_chunks); me = set(miss_eot)
    labels = [(-100 if inp or (0 <= c < K and c in mask) else t) for t, inp, c in zip(ids, is_input, chunk_at)]
    for p, q in enumerate(src_at):
        if is_input[p]: continue
        key = None if q is None else (q[0], q[1]); ev = key is not None and len(streams[q[0]][q[1]]) > 2
        if key in tri:                                                                          # 입력만 번역과 그것을 연 태그(<TRANS>, <SPK_x><TRANS>)
            labels[p] = -100; b = p - 1
            while b >= 0 and src_at[b] is None and not is_input[b] and (ids[b] == trans_id or ids[b] in sp.spk): labels[b] = -100; b -= 1
        elif key in lex or key in me: labels[p] = -100
        elif q is None and ids[p] == sp.next_audio and chunk_at[p] in lk: labels[p] = -100
        elif chunk_at[p] in mk and (q is None or ev): labels[p] = -100                           # 번역 없는 행: 태그·NEXT·EOT 를 가르치지 않는다
    gh = ghost_next(eot_ghost, src_at, chunk_at, ids, is_input, K, sp) if eot_ghost else []
    for p in gh: labels[p] = -100                                                               # 뺀 합성 <EOT> 의 라운드(C5)
    pos_weight = [0.0] * N; labels_alt = [-100] * N; soft_w = [1.0] * N; soft_pos: List[int] = []; eot_pos: List[int] = []
    for p, q in enumerate(src_at):
        if q is None or (q[0], q[1]) not in eot_p: continue
        eot_pos.append(p)
        if labels[p] == -100: continue
        s = q[0] % NS; nxt = src_at[p + 1] if p + 1 < N else None                               # <EOT> 뒤 <TRANS> = 같은 슬롯 번역이 이어진다(태그는 어차피 필요했다)
        opened = ids[p - 1] == sp.spk[s] and src_at[p - 1] is None and not (nxt is not None and nxt[0] % NS == s) and not (p + 1 < N and ids[p + 1] == trans_id)
        d = p - 1 if opened else p; pe = eot_p[(q[0], q[1])]
        if pe is None: labels[d] = labels[p] = -100; continue
        if labels[d] == -100: continue                                                          # 열린 태그가 가려진 구간이면 결정도 없다
        alt = ids[p + 1]; assert alt != ids[d], f"soft 대안이 결정 토큰과 같다: pos {d}"
        labels_alt[d] = alt; soft_w[d] = float(pe); soft_pos.append(d)
        if decision_weight: pos_weight[d] = float(decision_weight)
    out = dict(ids=ids, is_input=is_input, chunk_of=cof, labels=labels, pos_weight=pos_weight, labels_alt=labels_alt, soft_w=soft_w, soft_pos=soft_pos,
               eot_pos=eot_pos, src_at=src_at, chunk_at=chunk_at, chunks=chunks, srcs=srcs, st=st, n_rounds=n_rounds, n_flush=n_flush, n_eot=len(eot_pos),
               n_soft=len(soft_pos), flip=None, lex=lex, n_tr=sum(1 for ev in streams[NS:] for x in ev if len(x) == 2), eot_moved=n_moved,
               **(dict(eot_ghost=list(eot_ghost), ghost_pos=gh) if eot_ghost else {}))
    out.update(weight_decisions(out, sp, trans_id, event_id(streams), word_start, change_weight, hold_weight, dec_heads))
    return out


def read_back4(ids: Sequence[int], is_input: Sequence[bool], sp: Specials, trans_id: int, eot_id: int, n: int = 2) -> Tuple[List[List[int]], List[int], List[str]]:
    """출력 토큰열(입력·<NEXT_AUDIO> 제외)을 전사·번역 흐름 태그 규칙으로 다시 읽는다(슬롯 n 개)
    → (흐름별 글자 토큰 [슬롯 0..n−1 전사, 슬롯 0..n−1 번역], 슬롯별 <EOT> 수, 위반 목록).
    위반: untagged(태그 없이 시작) · trans_untagged(<TRANS> 앞에 화자 문맥 없음) · dup_tag(같은 구간 태그 연속, 사이에 방출 없음)."""
    out: List[List[int]] = [[] for _ in range(2 * n)]; eots = [0] * n; cur = None; bad: List[str] = []; since = 0
    for t, inp in zip(ids, is_input):
        if inp or t == sp.next_audio: continue
        if t in sp.spk:
            s = sp.spk.index(t)
            if cur == (s, 0) and since == 0: bad.append("dup_tag")
            cur = (s, 0); since = 0; continue
        if t == trans_id:
            if cur is None: bad.append("trans_untagged"); continue
            if cur[1] == 1 and since == 0: bad.append("dup_tag")
            cur = (cur[0], 1); since = 0; continue
        if cur is None: bad.append("untagged"); continue
        if t == eot_id: eots[cur[0]] += 1; since += 1; continue
        out[cur[0] + n * cur[1]].append(int(t)); since += 1
    return out, eots, bad


def lead_space_row(tok, ids, times, word_start) -> Tuple[list, list]:
    """번역 행의 첫 단어를 앞 공백 토큰으로 바꾼다 — 행 단위로 토큰화한 번역(ast_build_targets: ' '.join(단어))을 한 흐름에 이어 붙이면
    행 사이 단어가 붙는다('hurry,If', 2026-10-19 st4-p1 평가에서 참조 단어의 8 % 가 붙어 있었다). 첫 단어 토큰을 ' ' + 그 단어로 다시
    토큰화하고, 새 토큰은 원래 첫 단어의 시각(그 단어 마지막 토큰 시각)을 받는다. 이미 공백으로 시작하면 그대로."""
    ids, times = list(ids), list(times)
    if not ids or word_start(int(ids[0])): return ids, times
    k = next((j for j in range(1, len(ids)) if word_start(int(ids[j]))), len(ids))
    new = tok(" " + tok.decode([int(t) for t in ids[:k]]), add_special_tokens=False)["input_ids"]
    return [int(t) for t in new] + ids[k:], [times[k - 1]] * len(new) + times[k:]


class CstDialogueDataset(DialogueWindowDataset):
    """DialogueWindowDataset(창 선택·mono·마스크) + Stage 3 직렬화. exclude: 학습에서 뺄 conv_id(split held-out).
    windows: 주면 그 (conv_id, t0, L) 목록만 쓴다(p2_build_windows 산출물 등). max_speakers: 창 안 화자 상한(초과 창은 버림).
    decision_weight: EOT 결정 위치 가중(pos_weight 덮어쓰기, 0 = 기본 가중). backchannel: 맞장구 선택(모듈 설명, 기본 끔).
    eot_mode: 'soft'(p_end 그대로) | 'hard'(p_end ≥ 0.5 → 1, 아니면 0; 미관측은 그대로 마스크) — 파일럿 비교용.
    recover: 정체 회복 학습 설정(Recover, 기본 None = 끔) — __getitem__ 에서만 쓴다(모듈 설명).
    spk_activity: True 면 __getitem__ 이 화자 슬롯 활동 activity (K, n_slots) 를 낸다(모델 add_spk_head 보조 손실의 라벨).
    shrink_speakers(V2, 창 draw 와 함께): max_speakers 보다 많은 화자가 든 창을 버리지 않고 (max+1) 번째 화자가 처음 말하기 직전까지 줄인다(길이 ≥ window_s 하한일 때만).
    revive: 격리 발화 되살리기 토큰(load_revive, 기본 None = 끔) — 모듈 설명. 학습에서만 쓴다.
    Stage 4 번역(기본 끔): trans = 대화 발화 번역 타깃({행 id '<conv>|<화자>|<utt>': {목표 언어: (ids, 시각)}}, ast_dataset.load_ast_targets),
      trans_input = 되살린 발화의 번역(같은 형식 — 입력 문맥으로만, label −100), ast_frac = 학습 창을 번역 모드로 둘 확률(나머지는 Stage 3 형식),
      wav_aug = 학습 창 오디오 증강(callable(wav, rng), 예 phone_aug.PhoneAugment — 전화 대역, 2026-10-20). 시퀀스·K·라벨은 그대로, 기본 끔.
      trans_lead_space = 번역 행마다 첫 단어를 앞 공백 토큰으로(lead_space_row — 행을 이은 번역 흐름에서 단어가 붙지 않게, 기본 끔: st4-p1 까지의 형식),
      tgt_of = 원천 언어 → 목표 언어 후보(기본 한·독 → 영, 영 → 독·한 중 하나). 번역 모드의 시퀀스는 build_cst4_sequence(네 흐름) 다.
    교차 언어 대화(meta.spk_lang = {화자: 언어} 에 두 언어 — sim_build_dialogues --lang2): 슬롯을 언어로 고정하고(어느 언어를 A 로 둘지는 학습 rng,
      평가·점검은 창 id 해시), prefix 는 'language <SPK_A>{L_A} <SPK_B>{L_B}[\\ntranslate <SPK_A>{L_B} <SPK_B>{L_A}]' — 각 화자의 번역은 상대 언어다(slots).
    change_weight·hold_weight(V3-p1 C1, 기본 0 = 끔): 교대 태그·<EOT> 뒤 머무름 결정 가중(받아쓰기·번역 모드 모두) — 켜면 항목에 dec_kind·n_switch·n_hold.
    eot_drop_synth(V3-p1 C5, 기본 0 = 끔): meta.mask_eot 대화 창을 학습 rng 로 이 확률만큼 골라 합성 <EOT> 를 문맥에서 뺀다(모듈 설명) — 켜면 항목에 n_eot_drop.
    xl_order(V3 R4, 기본 'lang'): 'arrival' 이면 교차 언어 창도 먼저 말한 화자 = A, prefix 는 두 언어만(cst_head_text_xl — 'language English German[\ntranslate auto]').
      어느 슬롯이 어느 언어인지·번역 방향은 모델이 정한다. 같은 언어 창은 그대로다."""

    def __init__(self, dialogues: List[str], tok, delays=(4,), exclude: Optional[set] = None, windows: Optional[List[Tuple[str, float, float]]] = None,
                 max_speakers: int = 2, decision_weight: float = 0.0, backchannel: Optional[Tuple[float, int]] = None, eot_mode: str = "soft",
                 recover: Optional[Recover] = None, spk_activity: bool = False, revive: Optional[Dict[str, Dict[str, list]]] = None,
                 trans: Optional[Dict[str, Dict[str, tuple]]] = None, trans_input: Optional[Dict[str, Dict[str, tuple]]] = None, ast_frac: float = 0.0,
                 tgt_of: Optional[Dict[str, Sequence[str]]] = None, trans_lead_space: bool = False, wav_aug=None, declare_all_p: float = 0.0,
                 declare: Optional[int] = None, shrink_speakers: bool = False, change_weight: float = 0.0, hold_weight: float = 0.0, eot_drop_synth: float = 0.0, xl_order: str = "lang", **kw):
        if eot_mode not in ("soft", "hard"): raise ValueError(f"eot_mode: soft | hard — {eot_mode}")
        if xl_order not in ("lang", "arrival"): raise ValueError(f"xl_order lang|arrival — {xl_order}")
        self.xl_order = xl_order                                                                # V3 R4: 'arrival' 이면 교차 언어 창도 먼저 말한 순서 슬롯(slots·cst_prefix)
        if not 2 <= max(2, int(max_speakers)) <= len(SLOTS): raise ValueError(f"max_speakers ≤ {len(SLOTS)}: {max_speakers}")
        if not 0.0 <= declare_all_p <= 1.0 or (declare is not None and not 2 <= int(declare) <= len(SLOTS)): raise ValueError(f"declare_all_p {declare_all_p} · declare {declare}")
        if change_weight < 0 or hold_weight < 0 or not 0.0 <= eot_drop_synth <= 1.0: raise ValueError(f"change_weight·hold_weight ≥ 0 · eot_drop_synth [0, 1] — {change_weight} {hold_weight} {eot_drop_synth}")
        if recover is not None and not (0 <= recover.p <= 1 and 0 <= recover.boundary_frac <= 1 and recover.max_lag >= 1 and recover.max_boundaries >= 1):
            raise ValueError(f"recover 범위: {recover}")
        self.revived: Dict[str, Dict[str, Tuple[str, float, float]]] = {}                          # conv_id → {utt_id: (화자, start, end)} — 창 선택(super) 때는 비어 있다
        super().__init__(dialogues, tok, delays=delays, **kw)
        ex = set(exclude or ()); gone = [c for c in self.dlgs if c in ex]
        for c in gone: del self.dlgs[c]
        if windows is not None:
            self.items = [(c, round(float(t0), 3), round(float(L), 3)) for c, t0, L in windows if c in self.dlgs]; self.item_draw = [0] * len(self.items); self._est = None
        else: self.keep_items([it[0] in self.dlgs for it in self.items])
        if revive:                                                                                 # 창을 고른 뒤 — 창 목록은 끈 것과 같다
            n_miss = 0
            for cid, d in self.dlgs.items():
                m = revive.get(cid, {}); miss = set(d.meta.get("missing", []))
                for i, u in enumerate(d.utterances, 1):
                    if u.tokens or not u.flags or not set(u.flags) <= REVIVABLE or i in miss: continue    # 결손 조각은 오디오가 없다
                    if u.utt_id not in m: n_miss += 1; continue
                    u.tokens = list(m[u.utt_id]); self.revived.setdefault(cid, {})[u.utt_id] = (u.speaker, u.start, u.end)
            self._est = None
            self.stats.update(cst_revived=sum(len(v) for v in self.revived.values()), cst_revive_missing=n_miss)
        self.eot_mode = eot_mode; self.recover = recover if recover is not None and recover.p > 0 else None; self.spk_activity = bool(spk_activity)
        self.change_weight, self.hold_weight, self.eot_drop_synth = float(change_weight), float(hold_weight), float(eot_drop_synth)   # V3-p1 C1·C5(기본 끔)
        conv = lambda n: int(tok.convert_tokens_to_ids(n))
        self.n_slots = max(2, int(max_speakers)); self.spk_ids = tuple(conv(t) for t in SLOTS[: self.n_slots]); self.eot_id = conv("<EOT>"); self.decision_weight = float(decision_weight)
        self.declare_all_p, self.declare = float(declare_all_p), (None if declare is None else int(declare))
        self.delay_ids = {d: conv(f"<DELAY_{d}>") for d in self.delays}
        self.csp = Specials(next_audio=self.sp.next_audio, empty_audio=self.sp.empty_audio, spk=self.spk_ids)
        self.trans, self.trans_input, self.ast_frac = trans or {}, trans_input or {}, float(ast_frac)
        self.wav_aug = wav_aug                                                                     # 오디오 증강(phone_aug.PhoneAugment 등, 학습 전용) — 시퀀스·K 그대로
        self.trans_lead_space = bool(trans_lead_space)                                             # 행 사이 공백(lead_space_row) — Stage 4 본 학습부터
        self.tgt_of = dict(tgt_of) if tgt_of else {"Korean": ["English"], "German": ["English"], "English": ["German", "Korean"]}
        self.trans_id = conv("<TRANS>") if self.trans or self.trans_input else None
        self.has_xl = any(self.xl_langs(d) for d in self.dlgs.values())
        self.backchannel = tuple(backchannel) if backchannel else None
        self.eps: Dict[str, List[Episode]] = {c: dialogue_episodes(d, self.backchannel) for c, d in self.dlgs.items()}
        keep, n_drop, n_shrunk, lo = [], 0, 0, float(getattr(self, "window_s", (0.0,))[0])
        for j, (cid, t0, L) in enumerate(self.items):
            first = speaker_firsts(self.eps[cid], t0, L)
            if len(first) <= max_speakers: keep.append(True); continue
            L2 = math.floor((first[max_speakers][0] - t0) * 1000.0 - 1.0) / 1000.0                 # (max+1) 번째 화자가 처음 말하기 1 ms 전까지 — 내림(반올림하면 그 화자의 첫 발화가 창에 들어
                                                                                                    # 화자가 max+1 명이 된다: 2026-10-20 v2-p1 첫 step 의 slot_activity IndexError)
            if shrink_speakers and L2 >= lo and windows is None and len(speaker_firsts(self.eps[cid], t0, L2)) <= max_speakers:
                self.items[j] = (cid, t0, L2); keep.append(True); n_shrunk += 1
            else: keep.append(False); n_drop += 1
        self.keep_items(keep); self.stats.update(cst_windows=len(self.items), cst_dropped_speakers=n_drop, cst_shrunk_speakers=n_shrunk, cst_dialogues=len(self.dlgs), cst_excluded=len(gone))

    def cst_prefix(self, lang, delay: int, tgt=None) -> List[int]:
        """lang·tgt: 언어 하나(두 슬롯 같음) 또는 슬롯 순서 목록(선언 슬롯 수 = 길이). tgt 없으면 translate 줄 없음. tgt 가 문자열이면 선언 슬롯마다 같은 목표."""
        langs = (lang, lang) if isinstance(lang, str) else tuple(lang); tgts = None if not tgt else (tgt,) * len(langs) if isinstance(tgt, str) else tuple(tgt)
        c = self.__dict__.setdefault("_cst_head", {}); key = (langs, delay, tgts)
        if key not in c:
            head = cst_head_text_xl(langs, bool(tgts)) if self.xl_order == "arrival" and len(set(langs)) > 1 else cst_head_text(list(langs), list(tgts) if tgts else None)   # R4: 교차 언어는 슬롯에 언어를 묶지 않는다
            c[key] = self._pre + self.tok(head, add_special_tokens=False)["input_ids"] + [self.delay_ids[delay]]
        return list(c[key])

    @property
    def trans_on(self) -> bool:
        return self.trans_id is not None and self.ast_frac > 0

    @staticmethod
    def xl_langs(d: Dialogue) -> Optional[Dict[str, str]]:
        """교차 언어 대화면 {화자: 언어}(두 화자, 두 언어), 아니면 None."""
        sl = (d.meta or {}).get("spk_lang")
        return sl if sl and len(d.speakers) == 2 and all(s in sl for s in d.speakers) and len({sl[s] for s in d.speakers}) == 2 else None

    def slots(self, i: int, rng=None) -> Tuple[List[str], List[str]]:
        """창 i 의 슬롯 (화자 [A, B…], 선언 슬롯의 언어 목록). 같은 언어: 창 안에서 먼저 말한 사람 = A(window_speakers — 한 화자 창은 화자 하나),
        선언 슬롯 수 = max(2, 화자 수) — 학습(rng)에서는 declare_all_p 확률로 n_slots 전부(추론 때 화자 수를 모르면 최대를 선언하므로), 평가는 declare 가 있으면
        max(그 수, 화자 수). 2 슬롯 데이터셋(n_slots 2)은 예전과 같고 난수도 더 뽑지 않는다.
        교차 언어: 두 화자를 언어 이름순으로 두고 rng(학습) 또는 창 id 해시(평가·점검 — 재현)로 반을 뒤집는다(언어·위치 편향 방지).
        xl_order 'arrival'(V3 R4): 교차 언어도 창 안에서 먼저 말한 사람 = A(창에서 말하지 않은 화자는 뒤) — 난수를 뽑지 않고, prefix 는 cst_head_text_xl."""
        cid, t0, L = self.items[i]; d = self.dlgs[cid]; sl = self.xl_langs(d)
        if sl is None:
            spk = window_speakers(self.eps[cid], t0, L); D = max(2, len(spk))
            if self.declare is not None: D = max(D, self.declare)
            elif rng is not None and self.n_slots > D and self.declare_all_p > 0 and rng.random() < self.declare_all_p: D = self.n_slots
            return spk, [d.lang] * D
        sp = sorted(d.speakers, key=lambda s: sl[s])
        if self.xl_order == "arrival":
            w = [s for s in window_speakers(self.eps[cid], t0, L) if s in sl]; sp = w + [s for s in sp if s not in w]
            return sp, [sl[s] for s in sp]
        if (rng.random() < 0.5) if rng is not None else zlib.crc32(f"{cid}@{t0}".encode()) % 2: sp = sp[::-1]
        return sp, [sl[s] for s in sp]

    def tgt_options(self, i: int) -> List[str]:
        """번역 모드 목표: 같은 언어 = tgt_of[언어], 교차 언어 = ['xl'](각 슬롯이 상대 언어로)."""
        d = self.dlgs[self.items[i][0]]
        return ["xl"] if self.xl_langs(d) else list(self.tgt_of.get(d.lang, ()))

    def _rows(self, cid: str) -> Dict[Tuple[str, str], dict]:
        """대화의 행 (화자, 원 utt_id — 정제가 나눈 조각 #n 을 묶은 것, 번역 타깃의 행 id 와 같은 단위 — stage4_dialogue_units)
        → dict(a, e, kind 'n'(일반: 토큰 있고 격리 아님) | 'r'(되살림 — 번역은 입력만), eps [그 행 조각이 든 ep_id, 시작순], spans [조각 (start, end), 시작순]).
        격리된 채 남은 조각은 행이 아니다(마스크 구간). 2026-10-19 점검: 일반·되살림 조각이 섞인 행 0, 행이 episode 여럿에 걸침 1–15 %."""
        c = self.__dict__.setdefault("_row_cache", {})
        if cid not in c:
            rv = self.revived.get(cid, {}); piece = {}
            for u in self.dlgs[cid].utterances:
                k = "r" if u.utt_id in rv else "n" if u.tokens and not u.flags else None
                if k: piece[u.utt_id] = (u.utt_id.split("#")[0], k, u.start, u.end)
            rows: Dict[Tuple[str, str], dict] = {}
            for ep in self.eps[cid]:
                for uid in ep.utt_ids:
                    if uid not in piece: continue
                    b, k, a, e = piece[uid]; r = rows.setdefault((ep.speaker, b), dict(a=a, e=e, kind=k, eps=[], spans=[]))
                    r["a"], r["e"] = min(r["a"], a), max(r["e"], e); r["spans"].append((a, e))
                    if r["kind"] != k: r["kind"] = "r"                                          # 섞인 행은 입력만(점검 0 건)
                    if not r["eps"] or r["eps"][-1] != ep.ep_id: r["eps"].append(ep.ep_id)
            for r in rows.values(): r["spans"].sort()
            c[cid] = rows
        return c[cid]

    def trans_of(self, cid: str, tgt, t0: float):
        """window_streams4 의 trans_of(ep): episode 안 행마다 번역(되살린 행은 trans_input 에서, 입력만). tgt = 목표 언어 | {화자: 목표 언어}(교차 언어).
        확정 시각은 전사 토큰처럼 행 조각 구간으로 자른다(clamp_to_spans — 정렬 토큰이 조각 시작보다 앞서면 전사는 시작으로 당겨지는데 번역만 앞서지 않게).
        행이 episode 여럿에 걸치면 번역 토큰은 확정 시각이 든(그 시각 이후 처음 끝나는) episode 하나에만 준다 — 같은 번역을 두 번 넣지 않게.
        번역 없는 행(판정 탈락)과 창 시작 전에 시작한 행(번역이 창 밖 원문에 기댄다)은 그 episode 안 구간으로 낸다(사유 miss·cut)."""
        rows, E = self._rows(cid), self.eps[cid]
        def f(ep: Episode):
            toks, gaps, last_ok, seen = [], [], True, set(); tg = tgt.get(ep.speaker) if isinstance(tgt, dict) else tgt
            for uid in ep.utt_ids:
                key = (ep.speaker, uid.split("#")[0]); r = rows.get(key)
                if r is None or key in seen: continue
                seen.add(key); inp = r["kind"] == "r"
                tt = None if r["a"] < t0 - 1e-6 else (self.trans_input if inp else self.trans).get(f"{cid}|{key[0]}|{key[1]}", {}).get(tg)
                if tt is None: gaps.append((max(r["a"], ep.start), min(r["e"], ep.end), "cut" if r["a"] < t0 - 1e-6 else "miss")); last_ok = False; continue
                if self.trans_lead_space: tt = lead_space_row(self.tok, tt[0], tt[1], self.word_start)
                own = (lambda t: ep.ep_id) if len(r["eps"]) == 1 else (lambda t: next((j for j in r["eps"] if t <= E[j].end + 1e-6), r["eps"][-1]))
                ts = [clamp_to_spans(float(t), r["spans"]) for t in tt[1]]
                toks += [(int(i), t, inp) for i, t in zip(tt[0], ts) if own(t) == ep.ep_id]; last_ok = True
            return toks, gaps, last_ok
        return f

    def _untranscribed(self, d: Dialogue, u) -> bool:
        return u.utt_id not in self.revived.get(d.conv_id, ()) and super()._untranscribed(d, u)        # 되살린 발화는 마스크 구간이 아니다

    def cst_sequence(self, i: int, delay: int, rng=None, recover: Optional[Recover] = None, tgt: Optional[str] = None, drop_eot: Optional[bool] = None) -> dict:
        """창 i·δ 의 시퀀스. rng 를 주면 recover(기본 self.recover)로 정체 회복 구간을 넣을 수 있다 — 평가·점검은 rng 없이(뒤집지 않음).
        tgt(목표 언어, 교차 언어는 'xl')를 주거나, rng 가 있고 번역이 켜져 있으면 ast_frac 확률로 번역 모드(cst4_sequence) — tgt 는 tgt_options 에서 고른다.
        슬롯(화자·언어)은 slots(i, rng) — 교차 언어 대화만 rng 로 A/B 를 정한다.
        drop_eot(C5): None 이면 eot_drop_synth > 0 이고 rng 가 있고 meta.mask_eot 대화일 때만 rng 로 뽑는다(그 밖에는 난수를 뽑지 않는다), True/False 면 그대로(점검·dTF)."""
        cid, t0, L = self.items[i]; d = self.dlgs[cid]
        if tgt is None and rng is not None and self.trans_on and rng.random() < self.ast_frac:
            opts = self.tgt_options(i)
            if opts: tgt = rng.choice(opts)
        spk, langs = self.slots(i, rng)
        if drop_eot is None: drop_eot = bool(self.eot_drop_synth > 0 and rng is not None and d.meta.get("mask_eot") and rng.random() < self.eot_drop_synth)
        if tgt: return self.cst4_sequence(i, delay, tgt, spk, langs, drop_eot=drop_eot)
        K = int(round(L / CHUNK_S)); eps = self.eps[cid]
        hd = set() if self.change_weight > 0 or self.hold_weight > 0 else None                   # C1: episode 첫 전사 원소(앞 공백 없는 단위 첫 토큰도 턴 시작이면 단어 첫 토큰)
        streams, eot_p, wst, *gh = window_streams(eps, spk, t0, L, self.eot_id, drop_eot=bool(drop_eot), heads=hd)
        if self.eot_mode == "hard": eot_p = {k: (None if v is None else float(v >= 0.5)) for k, v in eot_p.items()}
        rv = [x for x in self.revived.get(cid, {}).values() if x[2] > t0 and x[1] < t0 + L]
        s = build_cst_sequence(streams, eot_p, K, self.cst_prefix(langs, delay), self.audio_pad, self.csp, delay,
                               mask_chunks=self.mask_chunks(d, t0, L, K, delay), decision_weight=self.decision_weight,
                               recover=recover if recover is not None else self.recover, rng=rng, word_start=self._ws(),
                               lex_mask=revived_elements(streams, spk, rv, t0) if rv else (), lex_chunks=revived_chunks(rv, spk, t0, K, delay) if rv else (),
                               change_weight=self.change_weight, hold_weight=self.hold_weight, eot_ghost=gh[0] if gh else (), dec_heads=hd or ())
        return dict(s, K=K, cid=cid, t0=t0, L=L, lang=d.lang, corpus=d.corpus, speakers=spk, langs=langs, streams=streams, win=wst, tgt=None, n_tr=0)

    def cst4_sequence(self, i: int, delay: int, tgt: str, spk: Optional[List[str]] = None, langs: Optional[List[str]] = None, drop_eot: bool = False) -> dict:
        """번역 모드: 두 슬롯의 전사 + 번역 네 흐름(window_streams4 → build_cst4_sequence). 같은 언어는 두 슬롯 모두 tgt 로,
        교차 언어(tgt 'xl')는 각 슬롯이 상대 슬롯 언어로 번역한다. spk·langs 를 안 주면 slots(i)(rng 없음 — 재현). drop_eot: 합성 <EOT> 를 뺀다(C5)."""
        if self.trans_id is None: raise ValueError("번역 타깃(trans)이 없다")
        cid, t0, L = self.items[i]; d = self.dlgs[cid]; K = int(round(L / CHUNK_S)); eps = self.eps[cid]
        if spk is None: spk, langs = self.slots(i)
        xl = len(set(langs)) > 1; tg = [langs[1], langs[0]] if xl else [tgt] * len(langs)
        hd = set() if self.change_weight > 0 or self.hold_weight > 0 else None
        streams, eot_p, info = window_streams4(eps, spk, t0, L, self.eot_id, self.trans_of(cid, dict(zip(spk, tg)) if xl else tgt, t0), drop_eot=bool(drop_eot), heads=hd)
        if self.eot_mode == "hard": eot_p = {k: (None if v is None else float(v >= 0.5)) for k, v in eot_p.items()}
        rv = [x for x in self.revived.get(cid, {}).values() if x[2] > t0 and x[1] < t0 + L]; mc = revived_chunks(info["miss"], spk, t0, K, delay)
        s = build_cst4_sequence(streams, eot_p, K, self.cst_prefix(langs, delay, tg), self.audio_pad, self.csp, self.trans_id, delay,
                                mask_chunks=self.mask_chunks(d, t0, L, K, delay), decision_weight=self.decision_weight,
                                lex_mask=revived_elements(streams[: len(streams) // 2], spk, rv, t0) if rv else (), lex_chunks=revived_chunks(rv, spk, t0, K, delay) if rv else (),
                                tr_input=info["tr_input"], miss_chunks=mc, miss_eot=info["miss_eot"], eot_tail=info["eot_tail"], word_start=self._ws(),
                                change_weight=self.change_weight, hold_weight=self.hold_weight, eot_ghost=info.get("ghost", ()), dec_heads=hd or ())
        win = dict(info["st"], tr_input=len(info["tr_input"]), eot_moved=s["eot_moved"])
        return dict(s, K=K, cid=cid, t0=t0, L=L, lang=d.lang, corpus=d.corpus, speakers=spk, langs=langs, streams=streams, win=win, tgt=tuple(tg), tr_input=info["tr_input"],
                    eot_tail=info["eot_tail"], eot_p=eot_p, miss_chunks=mc)

    def word_start(self, tid: int) -> bool:
        """단어 첫 토큰인가(디코드가 공백으로 시작) — 뒤집기 시작·회복 지점 후보, 교대·머무름 결정 위치(캐시)."""
        c = self.__dict__.setdefault("_word_start", {})
        if tid not in c: c[tid] = self.tok.decode([tid]).startswith(" ")
        return c[tid]

    def _ws(self):
        return self.word_start if hasattr(self.tok, "decode") else None                        # decode 없는 tokenizer(시험용) → 단어 경계 판정 없음

    def target_problems(self, i: int, delay: int, rng=None, recover: Optional[Recover] = None) -> List[str]:
        """시퀀스를 다시 읽어 슬롯별 토큰열(<EOT> 포함)이 이벤트열과 같은지, 태그 규칙·soft 결정 위치가 맞는지 검사(rng·recover → 뒤집은 시퀀스도)."""
        return self.problems_of(self.cst_sequence(i, delay, rng=rng, recover=recover))

    def problems_of(self, s: dict) -> List[str]:
        """cst_sequence 결과 하나의 검사(target_problems 본체 — dry-run 은 이미 만든 시퀀스를 넘긴다).
        뒤집은 시퀀스(flip): 슬롯별 다시 읽기는 뒤집은 문맥 슬롯 기준, 구간 출력 label 이 모두 −100, 회복 지점 label 이 정답
        (inrun = 그 화자 태그, boundary = 태그 없이 그 원소)인지 본다."""
        if s.get("tgt"): return self.problems_of4(s)
        P = []
        got, bad = read_back(s["ids"], s["is_input"], self.csp, self.eot_id)
        P += sorted(set(bad)); f = s.get("flip"); lex = s.get("lex") or ()
        if lex and any(s["labels"][p] != -100 for p, q in enumerate(s["src_at"]) if q is not None and (q[0], q[1]) in lex): P.append("revive:unmasked")
        if f is None: want = [[x[0] for x in ev] for ev in s["streams"]] + [[]] * (len(self.spk_ids) - len(s["streams"]))   # 화자 없는 슬롯은 비어야 한다
        else:
            want = [[] for _ in self.spk_ids]
            for p, q in enumerate(s["src_at"]):
                if q is not None: want[1 - q[0] if (q[0], q[1]) in f["keys"] else q[0]].append(s["ids"][p])
            a0, a1 = f["p_start"], f["p_end"]; ids, lab = s["ids"], s["labels"]
            if any(lab[p] != -100 for p in range(a0, a1) if not s["is_input"][p]): P.append("flip:unmasked")
            if lab[a1] != ids[a1]: P.append("flip:recover_label")
            want_tag = f["kind"].endswith("inrun")
            if want_tag != (ids[a1] in self.spk_ids) or (want_tag and ids[a1] != self.spk_ids[f["n1"][0]]) or (not want_tag and s["src_at"][a1] != f["n1"]):
                P.append(f"flip:recover:{f['kind']}")
        for k in range(len(self.spk_ids)):
            if got[k] != want[k]: P.append(f"order:{SLOTS[k]}")
        if len(s["langs"]) < max(2, len(s["speakers"])): P.append("declare:fewer_slots_than_speakers")
        for d in s["soft_pos"]:
            if s["labels"][d] != s["ids"][d] or not 0.0 <= s["soft_w"][d] <= 1.0 or s["labels_alt"][d] == s["ids"][d]: P.append(f"soft:{d}")
            if s["ids"][d] not in (self.eot_id,) + self.spk_ids: P.append(f"soft_tok:{d}")
        if s["n_eot"] + s["win"].get("eot_dropped", 0) != s["win"]["eot"]: P.append("eot_count")
        return P + self.decision_problems(s)

    def decision_problems(self, s: dict) -> List[str]:
        """C1·C5 검사(켰을 때만 — 끄면 []): 'switch_w'·'hold_w' 가중·dec_kind 위치가 decision_positions 와 같은가, pos_weight > 0 위치 = soft 결정(decision_weight) ∪ 켠 쪽 위치,
        교대 태그는 직전 문맥 슬롯과 다른가, 머무름은 같은 슬롯 <EOT> 바로 뒤 첫 결정인가. 'eot_ghost': ghost 수 = eot_dropped, ghost 라운드 <NEXT_AUDIO> label −100."""
        P: List[str] = []; ids, inp, lab = s["ids"], s["is_input"], s["labels"]
        if self.change_weight > 0 or self.hold_weight > 0:
            if "dec_kind" not in s: return ["switch_w:missing"]
            sw, ho = decision_positions(ids, inp, lab, s["labels_alt"], s["src_at"], self.spk_ids, self.trans_id, self.eot_id, self._ws(), s.get("dec_heads", ()))
            dk = list(s["dec_kind"])
            if [p for p, k in enumerate(dk) if k == 1] != sw or s["n_switch"] != len(sw): P.append("switch_w")
            if [p for p, k in enumerate(dk) if k == 2] != ho or s["n_hold"] != len(ho): P.append("hold_w")
            want = (set(s["soft_pos"]) if self.decision_weight else set()) | (set(sw) if self.change_weight > 0 else set()) | (set(ho) if self.hold_weight > 0 else set())
            if {p for p, w in enumerate(s["pos_weight"]) if w > 0} != want: P.append("switch_w:weighted")
            if any(s["pos_weight"][p] < w for ps, w in ((sw, self.change_weight), (ho, self.hold_weight)) for p in ps): P.append("switch_w:value")
            ctx, cur, prev = {}, None, None                                                     # 출력 위치마다 (그 토큰 앞 문맥 슬롯, 직전 출력 결정 토큰)
            for p, t in enumerate(ids):
                if inp[p] or (s["src_at"][p] is None and t not in self.spk_ids and t != self.trans_id): continue
                ctx[p] = (cur, prev); prev = p
                if t in self.spk_ids: cur = self.spk_ids.index(t)
            if any(ctx[p][0] is None or ctx[p][0] == self.spk_ids.index(ids[p]) for p in sw): P.append("switch_w:same_slot")
            for p in ho:
                q = ctx[p][1]; s0 = ctx[q][0] if q is not None else None
                if q is None or ids[q] != self.eot_id or s0 != (self.spk_ids.index(ids[p]) if ids[p] in self.spk_ids else ctx[p][0]): P.append("hold_w:not_after_eot"); break
        gh = s.get("eot_ghost") or ()
        if len(gh) != s["win"].get("eot_dropped", 0): P.append("eot_ghost:count")
        if gh:
            want = ghost_next(gh, s["src_at"], s["chunk_at"], ids, inp, s["K"], self.csp)
            if want != s.get("ghost_pos") or any(lab[p] != -100 or ids[p] != self.csp.next_audio for p in want): P.append("eot_ghost:label")
        return P

    def problems_of4(self, s: dict) -> List[str]:
        """번역 모드 시퀀스 검사: 네 흐름 다시 읽기(흐름별 글자 토큰·슬롯별 <EOT> 수), 태그 규칙, <EOT> 가 그 episode 의 전사·번역 꼬리 뒤인가,
        입력만 번역·되살린 전사의 label −100, soft 결정 위치."""
        n = len(s["streams"]) // 2; got, eots, bad = read_back4(s["ids"], s["is_input"], self.csp, self.trans_id, self.eot_id, n=n); P = sorted(set(bad))
        want = [[x[0] for x in ev if len(x) == 2] for ev in s["streams"]]
        for k in range(2 * n):
            if got[k] != want[k]: P.append(f"order:{'ABCD'[k % n]}{('', '+tr')[k // n]}")
        if eots != [sum(1 for ev in (s["streams"][k], s["streams"][n + k]) for x in ev if len(x) > 2) for k in range(n)]: P.append("eot_slots")
        if len(s["langs"]) < max(2, len(s["speakers"])): P.append("declare:fewer_slots_than_speakers")
        at = {(q[0], q[1]): p for p, q in enumerate(s["src_at"]) if q is not None}
        if any(k in at and any(t in at and at[t] > at[k] for t in tails) for k, tails in (s.get("eot_tail") or {}).items()): P.append("eot_before_tail")
        hide = set(s.get("lex") or ()) | set(s.get("tr_input") or ())
        if hide and any(s["labels"][p] != -100 for p, q in enumerate(s["src_at"]) if q is not None and (q[0], q[1]) in hide): P.append("input_only:unmasked")
        for d in s["soft_pos"]:
            if s["labels"][d] != s["ids"][d] or not 0.0 <= s["soft_w"][d] <= 1.0 or s["labels_alt"][d] == s["ids"][d]: P.append(f"soft:{d}")
            if s["ids"][d] not in (self.eot_id,) + self.spk_ids: P.append(f"soft_tok:{d}")
        if s["n_eot"] + s["win"].get("eot_dropped", 0) != s["win"]["eot"]: P.append("eot_count")
        return P + self.decision_problems(s)

    def est_lens(self) -> np.ndarray:
        est = super().est_lens() + 16 + 8 * (self.n_slots - 2)                                   # 두 슬롯 prefix 가 Phase 2 prefix 보다 길다(슬롯 하나에 언어·번역 줄 약 4 토큰씩)
        if not self.trans_on: return est
        if getattr(self, "_est4", None) is not None and len(self._est4) == len(self.items): return self._est4
        per = {}                                                                                 # 번역 모드로 뽑힐 수 있으니 창마다 번역 토큰 n(행의 목표 언어 중 최대) + 태그 2 + n/4 를 더한다
        for cid in {c for c, _, _ in self.items}:                                                # (전사·번역이 번갈면 태그가 는다 — 2026-10-19 시험 dry-run 에서 +2 만으로는 1 % 과소 추정 11 창)
            n_of = lambda x, w, b: max((len(v[0]) for v in (self.trans_input if x["kind"] == "r" else self.trans).get(f"{cid}|{w}|{b}", {}).values()), default=0)
            r = [(x["a"], x["e"], 2 + n + n // 4) for (w, b), x in self._rows(cid).items() for n in (n_of(x, w, b),)]
            per[cid] = (np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.array([x[2] for x in r]))
        add = np.array([int(per[c][2][(per[c][1] > t0) & (per[c][0] < t0 + L)].sum()) if len(per[c][0]) else 0 for c, t0, L in self.items], dtype=np.int64)
        self._est4 = est + add + 12                                                              # translate 줄
        return self._est4

    def __getitem__(self, i):
        delay = random.choice(self.delays); cid = self.items[i][0]
        drop = (random.random() < self.eot_drop_synth) if self.eot_drop_synth > 0 and self.dlgs[cid].meta.get("mask_eot") else None   # C5 동전은 따로 — 형식 난수(slots·recover·tgt)는 예전 조건 그대로
        s = self.cst_sequence(i, delay, rng=random if (self.recover or self.trans_on or self.has_xl) else None, **({"drop_eot": drop} if drop is not None else {}))
        wav = np.asarray(crop_audio(self.mono(s["cid"]), s["t0"], s["t0"] + s["L"]), dtype=np.float32)
        if self.wav_aug is not None: wav = self.wav_aug(wav, random)                              # 끄면 난수를 더 뽑지 않는다
        return dict(wav=torch.from_numpy(np.ascontiguousarray(wav)), ids=torch.tensor(s["ids"]), is_audio=torch.tensor([c >= 0 for c in s["chunk_of"]]),
                    chunk_of=torch.tensor(s["chunk_of"]), labels=torch.tensor(s["labels"]), pos_weight=torch.tensor(s["pos_weight"], dtype=torch.float32),
                    labels_alt=torch.tensor(s["labels_alt"]), soft_w=torch.tensor(s["soft_w"], dtype=torch.float32),
                    lang=s["lang"], delay=delay, name=s["corpus"], id=f"{s['cid']}@{s['t0']}", n_text=s["st"].tokens, overflow=s["st"].overflow_tokens,
                    n_flush=s["n_flush"], K=s["K"], rounds=s["n_rounds"], n_eot=s["n_eot"], n_soft=s["n_soft"], n_spk=len(s["speakers"]), n_flip=int(s["flip"] is not None), n_rev=len(s["lex"]), n_tr=int(s.get("n_tr", 0)),
                    **({"activity": torch.tensor(slot_activity(self.eps[s["cid"]], s["speakers"], s["t0"], s["L"], s["K"], self.n_slots), dtype=torch.float)} if self.spk_activity else {}),
                    **({"dec_kind": torch.tensor(s["dec_kind"], dtype=torch.int8), "n_switch": s["n_switch"], "n_hold": s["n_hold"]} if "dec_kind" in s else {}),   # C1(켰을 때만)
                    **({"n_eot_drop": int(s["win"].get("eot_dropped", 0))} if self.eot_drop_synth > 0 else {}))                                               # C5(켰을 때만)


def collate_cst(batch):
    """collate_streams + pos_weight·labels_alt·soft_w_full (B, L)(packing ROW_KEYS) + sem_only_mask(유효 위치 True: <SEM_END> 중립, <EOT> 학습)
    + (spk_activity 면) activity (B, Kmax, 2)·activity_mask (B, Kmax) — 인코더 출력 (B, Kmax) 과 줄을 맞춘다(packing.pack_batch 가 그대로 편다)
    + (교대·머무름 가중을 켠 항목이 있으면) dec_kind (B, L) int8(0 = 결정 아님·패딩 — 모델 forward 통계)·n_switch·n_hold, (EOT-문맥 dropout 을 켰으면) n_eot_drop."""
    out = collate_streams(batch); B, L = out["ids"].shape
    pw = torch.zeros(B, L); la = torch.full((B, L), -100, dtype=torch.long); sw = torch.ones(B, L)
    for i, b in enumerate(batch):
        n = len(b["ids"]); pw[i, :n] = b["pos_weight"]; la[i, :n] = b["labels_alt"]; sw[i, :n] = b["soft_w"]
    out.update(pos_weight=pw, labels_alt=la, soft_w_full=sw, sem_only_mask=out["mask"].bool())
    for k in ("n_eot", "n_soft", "n_flip", "n_rev", "n_tr"): out[k] = sum(int(b.get(k, 0)) for b in batch)
    if any("dec_kind" in b for b in batch):                                                     # 없으면 키를 만들지 않는다(끈 배치는 예전 그대로)
        dk = torch.zeros(B, L, dtype=torch.int8)
        for i, b in enumerate(batch):
            if "dec_kind" in b: dk[i, : len(b["ids"])] = b["dec_kind"]
        out["dec_kind"] = dk
    for k in ("n_switch", "n_hold", "n_eot_drop"):
        if any(k in b for b in batch): out[k] = sum(int(b.get(k, 0)) for b in batch)
    if "activity" in batch[0]:
        Km = max(int(b["K"]) for b in batch); act = torch.zeros(B, Km, batch[0]["activity"].shape[1]); am = torch.zeros(B, Km)
        for i, b in enumerate(batch):
            k = b["activity"].shape[0]; assert k == int(b["K"]), f"activity 길이 {k} ≠ K {b['K']}"; act[i, :k] = b["activity"]; am[i, :k] = 1
        out.update(activity=act, activity_mask=am)
    return out
