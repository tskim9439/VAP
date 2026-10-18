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
세 화자 이상이 말하는 창은 버린다(회의 2 화자 창은 P2 의 생성기가 따로 만든다).
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
"""
import json
import math
import random
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch

from .dialogue import Dialogue, Episode, CHUNK_S, build_episodes, chunk_of
from .dialogue_dataset import DialogueWindowDataset
from .dialogue_mix import crop_audio
from .eot_soft import assign_p_end, classify, P_END
from .interleave import Specials
from .lane_alloc import EOT_GAP_S
from .semcommit_dataset import build_interleaved_events
from ..uslm.mono_data import collate_streams

SLOTS = ("<SPK_A>", "<SPK_B>")
SR = 16000
REVIVABLE = frozenset({"asr_disagree"})                         # 되살릴 격리 사유(시각·화자는 믿을 수 있고 전사만 의심) — implausible_rate·TN 사유는 그대로 마스크


def cst_head_text(langs: Sequence[str], tgts: Optional[Sequence[str]] = None) -> str:
    """두 슬롯 prefix 본문 'language <SPK_A>Korean <SPK_B>Korean[\\ntranslate <SPK_A>English <SPK_B>English]<asr_text>'
    (ast_dataset.ast_head_text 의 한 슬롯 형식을 슬롯 순서 A, B 로 넓힌 것)."""
    s = "language " + " ".join(f"{t}{l}" for t, l in zip(SLOTS, langs))
    if tgts: s += "\ntranslate " + " ".join(f"{t}{l}" for t, l in zip(SLOTS, tgts))
    return s + "<asr_text>"


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


def window_speakers(eps: List[Episode], t0: float, L: float) -> List[str]:
    """창 [t0, t0+L) 에서 말하는 화자를 먼저 말한 순서로(창 안 첫 발화 시각, 같으면 화자 ID)."""
    first: Dict[str, float] = {}
    for ep in eps:
        if ep.end > t0 and ep.start < t0 + L: first[ep.speaker] = min(first.get(ep.speaker, 1e18), max(ep.start, t0))
    return sorted(first, key=lambda s: (first[s], s))


def window_streams(eps: List[Episode], speakers: Sequence[str], t0: float, L: float, eot_id: int) -> Tuple[List[list], Dict[Tuple[int, int], Optional[float]], Dict[str, int]]:
    """episode → 슬롯별 이벤트열(창 기준 시각). 원소 = (tid, t) 전사 토큰 | (eot_id, t_end, k_min) 명시 청크 이벤트.
    → (streams, {(슬롯, 원소 index): p_end}, 통계)."""
    t1 = t0 + L; streams: List[list] = [[] for _ in speakers]; eot_p: Dict[Tuple[int, int], Optional[float]] = {}; st = dict(eot=0, eot_truncated=0, eot_no_tokens=0, eot_backchannel=0)
    slot = {s: i for i, s in enumerate(speakers)}
    for ep in eps:
        if ep.end <= t0 or ep.start >= t1 or ep.speaker not in slot: continue
        s = slot[ep.speaker]; toks = [(int(tid), float(t) - t0) for tid, t in ep.tokens if t0 <= t <= t1]
        streams[s] += toks
        if ep.end > t1 + 1e-9: st["eot_truncated"] += 1; continue
        if not toks: st["eot_no_tokens"] += 1; continue
        if ep.outcome == "backchannel": st["eot_backchannel"] += 1; continue
        e = ep.end - t0; streams[s].append((int(eot_id), e, chunk_of(e + EOT_GAP_S))); eot_p[(s, len(streams[s]) - 1)] = ep.p_end; st["eot"] += 1
    return streams, eot_p, st


def slot_activity(eps: List[Episode], speakers: Sequence[str], t0: float, L: float, K: int) -> List[List[int]]:
    """(K, 2) 0/1 — 창 기준 청크 [k·0.08, (k+1)·0.08) 가 슬롯 s 화자의 episode 와 겹치면 1(겹침 구간은 둘 다 1). 슬롯 = speakers 순서."""
    act = [[0, 0] for _ in range(K)]; slot = {s: i for i, s in enumerate(speakers)}
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


def build_cst_sequence(streams: List[list], eot_p: Dict[Tuple[int, int], Optional[float]], K: int, prefix: List[int], audio_pad: int, sp: Specials,
                       delay: int, mask_chunks=(), decision_weight: float = 0.0, recover: Optional[Recover] = None, rng=None, word_start=None,
                       lex_mask=(), lex_chunks=()) -> dict:
    """슬롯 이벤트열 → 학습 시퀀스(오디오와 무관한 순수 함수; recover·rng 를 주면 그 확률로 정체 회복 구간을 넣는다 — 모듈 설명).
    word_start(tid) → bool: 주면 뒤집기 시작·회복 지점을 단어 첫 토큰으로 제한한다(실제 오류는 단어 경계에서 난다 — 단어 중간 태그를 가르치지 않게).
    lex_mask: 전사 label 만 −100 인 원소 (슬롯, index), lex_chunks: <NEXT_AUDIO> label 을 −100 으로 둘 청크(K = flush) — 되살린 발화(revive, 모듈 설명).
    → dict(ids, is_input, chunk_of, labels, pos_weight, labels_alt, soft_w, soft_pos, eot_pos, src_at, chunk_at, chunks, srcs, st, n_rounds, n_flush, n_eot, n_soft,
           flip(None | dict(kind, keys, p_start, p_end)), lex)."""
    streams = [list(s) for s in streams] + [[] for _ in range(2 - len(streams))]
    chunks, st, srcs = build_interleaved_events(streams, (K - 0.5) * CHUNK_S, sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=0,
                                                add_spk_tags=True, with_src=True)
    tag_flush(chunks, srcs, K, sp)
    mask = set(mask_chunks); lex = set(lex_mask); flip = None
    if recover is not None and rng is not None and recover.p > 0 and rng.random() < recover.p:
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
    return dict(ids=ids, is_input=is_input, chunk_of=cof, labels=labels, pos_weight=pos_weight, labels_alt=labels_alt, soft_w=soft_w, soft_pos=soft_pos,
                eot_pos=eot_pos, src_at=src_at, chunk_at=chunk_at, chunks=chunks, srcs=srcs, st=st, n_rounds=n_rounds, n_flush=n_flush, n_eot=len(eot_pos), n_soft=len(soft_pos),
                flip=flip, lex=lex)


def read_back(ids: Sequence[int], is_input: Sequence[bool], sp: Specials, eot_id: int) -> Tuple[List[List[int]], List[str]]:
    """출력 토큰열(입력 위치·<NEXT_AUDIO> 제외)을 태그 규칙으로 다시 읽는다 → (슬롯별 토큰열(<EOT> 포함), 위반 목록).
    위반: 'untagged'(태그 없이 시작) · 'dup_tag'(같은 태그 연속, 그 사이에 방출 없음)."""
    out: List[List[int]] = [[], []]; cur = None; bad: List[str] = []; since = 0
    for t, inp in zip(ids, is_input):
        if inp or t == sp.next_audio: continue
        if t in sp.spk:
            s = sp.spk.index(t)
            if s == cur and since == 0: bad.append("dup_tag")
            cur = s; since = 0; continue
        if cur is None: bad.append("untagged"); continue
        out[cur].append(int(t)); since += 1
    return out, bad


class CstDialogueDataset(DialogueWindowDataset):
    """DialogueWindowDataset(창 선택·mono·마스크) + Stage 3 직렬화. exclude: 학습에서 뺄 conv_id(split held-out).
    windows: 주면 그 (conv_id, t0, L) 목록만 쓴다(p2_build_windows 산출물 등). max_speakers: 창 안 화자 상한(초과 창은 버림).
    decision_weight: EOT 결정 위치 가중(pos_weight 덮어쓰기, 0 = 기본 가중). backchannel: 맞장구 선택(모듈 설명, 기본 끔).
    eot_mode: 'soft'(p_end 그대로) | 'hard'(p_end ≥ 0.5 → 1, 아니면 0; 미관측은 그대로 마스크) — 파일럿 비교용.
    recover: 정체 회복 학습 설정(Recover, 기본 None = 끔) — __getitem__ 에서만 쓴다(모듈 설명).
    spk_activity: True 면 __getitem__ 이 화자 슬롯 활동 activity (K, 2) 를 낸다(모델 add_spk_head 보조 손실의 라벨).
    revive: 격리 발화 되살리기 토큰(load_revive, 기본 None = 끔) — 모듈 설명. 학습에서만 쓴다."""

    def __init__(self, dialogues: List[str], tok, delays=(4,), exclude: Optional[set] = None, windows: Optional[List[Tuple[str, float, float]]] = None,
                 max_speakers: int = 2, decision_weight: float = 0.0, backchannel: Optional[Tuple[float, int]] = None, eot_mode: str = "soft",
                 recover: Optional[Recover] = None, spk_activity: bool = False, revive: Optional[Dict[str, Dict[str, list]]] = None, **kw):
        if eot_mode not in ("soft", "hard"): raise ValueError(f"eot_mode: soft | hard — {eot_mode}")
        if recover is not None and not (0 <= recover.p <= 1 and 0 <= recover.boundary_frac <= 1 and recover.max_lag >= 1 and recover.max_boundaries >= 1):
            raise ValueError(f"recover 범위: {recover}")
        self.revived: Dict[str, Dict[str, Tuple[str, float, float]]] = {}                          # conv_id → {utt_id: (화자, start, end)} — 창 선택(super) 때는 비어 있다
        super().__init__(dialogues, tok, delays=delays, **kw)
        ex = set(exclude or ()); gone = [c for c in self.dlgs if c in ex]
        for c in gone: del self.dlgs[c]
        if windows is not None: self.items = [(c, round(float(t0), 3), round(float(L), 3)) for c, t0, L in windows if c in self.dlgs]
        else: self.items = [it for it in self.items if it[0] in self.dlgs]
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
        conv = lambda n: int(tok.convert_tokens_to_ids(n))
        self.spk_ids = (conv("<SPK_A>"), conv("<SPK_B>")); self.eot_id = conv("<EOT>"); self.decision_weight = float(decision_weight)
        self.delay_ids = {d: conv(f"<DELAY_{d}>") for d in self.delays}
        self.csp = Specials(next_audio=self.sp.next_audio, empty_audio=self.sp.empty_audio, spk=self.spk_ids)
        self.backchannel = tuple(backchannel) if backchannel else None
        self.eps: Dict[str, List[Episode]] = {c: dialogue_episodes(d, self.backchannel) for c, d in self.dlgs.items()}
        keep, n_drop = [], 0
        for it in self.items:
            if len(window_speakers(self.eps[it[0]], it[1], it[2])) <= max_speakers: keep.append(it)
            else: n_drop += 1
        self.items = keep; self.stats.update(cst_windows=len(keep), cst_dropped_speakers=n_drop, cst_dialogues=len(self.dlgs), cst_excluded=len(gone))

    def cst_prefix(self, lang: str, delay: int) -> List[int]:
        c = self.__dict__.setdefault("_cst_head", {})
        if (lang, delay) not in c: c[(lang, delay)] = self._pre + self.tok(cst_head_text([lang, lang]), add_special_tokens=False)["input_ids"] + [self.delay_ids[delay]]
        return list(c[(lang, delay)])

    def _untranscribed(self, d: Dialogue, u) -> bool:
        return u.utt_id not in self.revived.get(d.conv_id, ()) and super()._untranscribed(d, u)        # 되살린 발화는 마스크 구간이 아니다

    def cst_sequence(self, i: int, delay: int, rng=None, recover: Optional[Recover] = None) -> dict:
        """창 i·δ 의 시퀀스. rng 를 주면 recover(기본 self.recover)로 정체 회복 구간을 넣을 수 있다 — 평가·점검은 rng 없이(뒤집지 않음)."""
        cid, t0, L = self.items[i]; d = self.dlgs[cid]; K = int(round(L / CHUNK_S)); eps = self.eps[cid]
        spk = window_speakers(eps, t0, L); streams, eot_p, wst = window_streams(eps, spk, t0, L, self.eot_id)
        if self.eot_mode == "hard": eot_p = {k: (None if v is None else float(v >= 0.5)) for k, v in eot_p.items()}
        rv = [x for x in self.revived.get(cid, {}).values() if x[2] > t0 and x[1] < t0 + L]
        s = build_cst_sequence(streams, eot_p, K, self.cst_prefix(d.lang, delay), self.audio_pad, self.csp, delay,
                               mask_chunks=self.mask_chunks(d, t0, L, K, delay), decision_weight=self.decision_weight,
                               recover=recover if recover is not None else self.recover, rng=rng, word_start=self.word_start if hasattr(self.tok, "decode") else None,
                               lex_mask=revived_elements(streams, spk, rv, t0) if rv else (), lex_chunks=revived_chunks(rv, spk, t0, K, delay) if rv else ())
        return dict(s, K=K, cid=cid, t0=t0, L=L, lang=d.lang, corpus=d.corpus, speakers=spk, streams=streams, win=wst)

    def word_start(self, tid: int) -> bool:
        """단어 첫 토큰인가(디코드가 공백으로 시작) — 뒤집기 시작·회복 지점 후보(캐시)."""
        c = self.__dict__.setdefault("_word_start", {})
        if tid not in c: c[tid] = self.tok.decode([tid]).startswith(" ")
        return c[tid]

    def target_problems(self, i: int, delay: int, rng=None, recover: Optional[Recover] = None) -> List[str]:
        """시퀀스를 다시 읽어 슬롯별 토큰열(<EOT> 포함)이 이벤트열과 같은지, 태그 규칙·soft 결정 위치가 맞는지 검사(rng·recover → 뒤집은 시퀀스도)."""
        return self.problems_of(self.cst_sequence(i, delay, rng=rng, recover=recover))

    def problems_of(self, s: dict) -> List[str]:
        """cst_sequence 결과 하나의 검사(target_problems 본체 — dry-run 은 이미 만든 시퀀스를 넘긴다).
        뒤집은 시퀀스(flip): 슬롯별 다시 읽기는 뒤집은 문맥 슬롯 기준, 구간 출력 label 이 모두 −100, 회복 지점 label 이 정답
        (inrun = 그 화자 태그, boundary = 태그 없이 그 원소)인지 본다."""
        P = []
        got, bad = read_back(s["ids"], s["is_input"], self.csp, self.eot_id)
        P += sorted(set(bad)); f = s.get("flip"); lex = s.get("lex") or ()
        if lex and any(s["labels"][p] != -100 for p, q in enumerate(s["src_at"]) if q is not None and (q[0], q[1]) in lex): P.append("revive:unmasked")
        if f is None: want = [[x[0] for x in ev] for ev in s["streams"]] + [[]] * (2 - len(s["streams"]))      # 한 화자 창: 슬롯 B 는 비어야 한다
        else:
            want = [[], []]
            for p, q in enumerate(s["src_at"]):
                if q is not None: want[1 - q[0] if (q[0], q[1]) in f["keys"] else q[0]].append(s["ids"][p])
            a0, a1 = f["p_start"], f["p_end"]; ids, lab = s["ids"], s["labels"]
            if any(lab[p] != -100 for p in range(a0, a1) if not s["is_input"][p]): P.append("flip:unmasked")
            if lab[a1] != ids[a1]: P.append("flip:recover_label")
            want_tag = f["kind"].endswith("inrun")
            if want_tag != (ids[a1] in self.spk_ids) or (want_tag and ids[a1] != self.spk_ids[f["n1"][0]]) or (not want_tag and s["src_at"][a1] != f["n1"]):
                P.append(f"flip:recover:{f['kind']}")
        for k in range(2):
            if got[k] != want[k]: P.append(f"order:{SLOTS[k]}")
        for d in s["soft_pos"]:
            if s["labels"][d] != s["ids"][d] or not 0.0 <= s["soft_w"][d] <= 1.0 or s["labels_alt"][d] == s["ids"][d]: P.append(f"soft:{d}")
            if s["ids"][d] not in (self.eot_id,) + self.spk_ids: P.append(f"soft_tok:{d}")
        if s["n_eot"] != s["win"]["eot"]: P.append("eot_count")
        return P

    def est_lens(self) -> np.ndarray:
        return super().est_lens() + 16                                                           # 두 슬롯 prefix 가 Phase 2 prefix 보다 길다

    def __getitem__(self, i):
        delay = random.choice(self.delays); s = self.cst_sequence(i, delay, rng=random if self.recover else None)
        wav = np.asarray(crop_audio(self.mono(s["cid"]), s["t0"], s["t0"] + s["L"]), dtype=np.float32)
        return dict(wav=torch.from_numpy(np.ascontiguousarray(wav)), ids=torch.tensor(s["ids"]), is_audio=torch.tensor([c >= 0 for c in s["chunk_of"]]),
                    chunk_of=torch.tensor(s["chunk_of"]), labels=torch.tensor(s["labels"]), pos_weight=torch.tensor(s["pos_weight"], dtype=torch.float32),
                    labels_alt=torch.tensor(s["labels_alt"]), soft_w=torch.tensor(s["soft_w"], dtype=torch.float32),
                    lang=s["lang"], delay=delay, name=s["corpus"], id=f"{s['cid']}@{s['t0']}", n_text=s["st"].tokens, overflow=s["st"].overflow_tokens,
                    n_flush=s["n_flush"], K=s["K"], rounds=s["n_rounds"], n_eot=s["n_eot"], n_soft=s["n_soft"], n_spk=len(s["speakers"]), n_flip=int(s["flip"] is not None), n_rev=len(s["lex"]),
                    **({"activity": torch.tensor(slot_activity(self.eps[s["cid"]], s["speakers"], s["t0"], s["L"], s["K"]), dtype=torch.float)} if self.spk_activity else {}))


def collate_cst(batch):
    """collate_streams + pos_weight·labels_alt·soft_w_full (B, L)(packing ROW_KEYS) + sem_only_mask(유효 위치 True: <SEM_END> 중립, <EOT> 학습)
    + (spk_activity 면) activity (B, Kmax, 2)·activity_mask (B, Kmax) — 인코더 출력 (B, Kmax) 과 줄을 맞춘다(packing.pack_batch 가 그대로 편다)."""
    out = collate_streams(batch); B, L = out["ids"].shape
    pw = torch.zeros(B, L); la = torch.full((B, L), -100, dtype=torch.long); sw = torch.ones(B, L)
    for i, b in enumerate(batch):
        n = len(b["ids"]); pw[i, :n] = b["pos_weight"]; la[i, :n] = b["labels_alt"]; sw[i, :n] = b["soft_w"]
    out.update(pos_weight=pw, labels_alt=la, soft_w_full=sw, sem_only_mask=out["mask"].bool())
    for k in ("n_eot", "n_soft", "n_flip", "n_rev"): out[k] = sum(int(b.get(k, 0)) for b in batch)
    if "activity" in batch[0]:
        Km = max(int(b["K"]) for b in batch); act = torch.zeros(B, Km, batch[0]["activity"].shape[1]); am = torch.zeros(B, Km)
        for i, b in enumerate(batch):
            k = b["activity"].shape[0]; assert k == int(b["K"]), f"activity 길이 {k} ≠ K {b['K']}"; act[i, :k] = b["activity"]; am[i, :k] = 1
        out.update(activity=act, activity_mask=am)
    return out
