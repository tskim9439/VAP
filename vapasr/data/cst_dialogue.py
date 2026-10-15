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
"""
import random
from typing import Dict, List, Optional, Sequence, Tuple

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
    if backchannel is None: assign_p_end(eps, d.observed()); return eps
    bc = [is_backchannel(e, backchannel) for e in eps]
    for e, b in zip(eps, bc):
        if b: e.outcome, e.p_end = "backchannel", None; continue
        e.outcome = classify(e, [o for o, ob in zip(eps, bc) if o.speaker == e.speaker or not ob], d.observed()); e.p_end = P_END[e.outcome]
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
                       delay: int, mask_chunks=(), decision_weight: float = 0.0) -> dict:
    """슬롯 이벤트열 → 학습 시퀀스(오디오와 무관한 순수 함수).
    → dict(ids, is_input, chunk_of, labels, pos_weight, labels_alt, soft_w, soft_pos, eot_pos, src_at, chunk_at, chunks, srcs, st, n_rounds, n_flush, n_eot, n_soft)."""
    streams = [list(s) for s in streams] + [[] for _ in range(2 - len(streams))]
    chunks, st, srcs = build_interleaved_events(streams, (K - 0.5) * CHUNK_S, sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=0,
                                                add_spk_tags=True, with_src=True)
    tag_flush(chunks, srcs, K, sp)
    ids, is_input, cof, src_at, chunk_at, n_rounds, n_flush = flatten_cst(chunks, srcs, K, prefix, audio_pad, sp)
    mask = set(mask_chunks); N = len(ids)
    labels = [(-100 if inp or (0 <= c < K and c in mask) else t) for t, inp, c in zip(ids, is_input, chunk_at)]
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
                eot_pos=eot_pos, src_at=src_at, chunk_at=chunk_at, chunks=chunks, srcs=srcs, st=st, n_rounds=n_rounds, n_flush=n_flush, n_eot=len(eot_pos), n_soft=len(soft_pos))


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
    eot_mode: 'soft'(p_end 그대로) | 'hard'(p_end ≥ 0.5 → 1, 아니면 0; 미관측은 그대로 마스크) — 파일럿 비교용."""

    def __init__(self, dialogues: List[str], tok, delays=(4,), exclude: Optional[set] = None, windows: Optional[List[Tuple[str, float, float]]] = None,
                 max_speakers: int = 2, decision_weight: float = 0.0, backchannel: Optional[Tuple[float, int]] = None, eot_mode: str = "soft", **kw):
        if eot_mode not in ("soft", "hard"): raise ValueError(f"eot_mode: soft | hard — {eot_mode}")
        super().__init__(dialogues, tok, delays=delays, **kw)
        ex = set(exclude or ()); gone = [c for c in self.dlgs if c in ex]
        for c in gone: del self.dlgs[c]
        if windows is not None: self.items = [(c, round(float(t0), 3), round(float(L), 3)) for c, t0, L in windows if c in self.dlgs]
        else: self.items = [it for it in self.items if it[0] in self.dlgs]
        self.eot_mode = eot_mode; conv = lambda n: int(tok.convert_tokens_to_ids(n))
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

    def cst_sequence(self, i: int, delay: int) -> dict:
        cid, t0, L = self.items[i]; d = self.dlgs[cid]; K = int(round(L / CHUNK_S)); eps = self.eps[cid]
        spk = window_speakers(eps, t0, L); streams, eot_p, wst = window_streams(eps, spk, t0, L, self.eot_id)
        if self.eot_mode == "hard": eot_p = {k: (None if v is None else float(v >= 0.5)) for k, v in eot_p.items()}
        s = build_cst_sequence(streams, eot_p, K, self.cst_prefix(d.lang, delay), self.audio_pad, self.csp, delay,
                               mask_chunks=self.mask_chunks(d, t0, L, K, delay), decision_weight=self.decision_weight)
        return dict(s, K=K, cid=cid, t0=t0, L=L, lang=d.lang, corpus=d.corpus, speakers=spk, streams=streams, win=wst)

    def target_problems(self, i: int, delay: int) -> List[str]:
        """시퀀스를 다시 읽어 슬롯별 토큰열(<EOT> 포함)이 이벤트열과 같은지, 태그 규칙·soft 결정 위치가 맞는지 검사."""
        return self.problems_of(self.cst_sequence(i, delay))

    def problems_of(self, s: dict) -> List[str]:
        """cst_sequence 결과 하나의 검사(target_problems 본체 — dry-run 은 이미 만든 시퀀스를 넘긴다)."""
        P = []
        got, bad = read_back(s["ids"], s["is_input"], self.csp, self.eot_id)
        P += sorted(set(bad))
        for k, ev in enumerate(s["streams"]):
            if got[k] != [x[0] for x in ev]: P.append(f"order:{SLOTS[k]}")
        for d in s["soft_pos"]:
            if s["labels"][d] != s["ids"][d] or not 0.0 <= s["soft_w"][d] <= 1.0 or s["labels_alt"][d] == s["ids"][d]: P.append(f"soft:{d}")
            if s["ids"][d] not in (self.eot_id,) + self.spk_ids: P.append(f"soft_tok:{d}")
        if s["n_eot"] != s["win"]["eot"]: P.append("eot_count")
        return P

    def est_lens(self) -> np.ndarray:
        return super().est_lens() + 16                                                           # 두 슬롯 prefix 가 Phase 2 prefix 보다 길다

    def __getitem__(self, i):
        delay = random.choice(self.delays); s = self.cst_sequence(i, delay)
        wav = np.asarray(crop_audio(self.mono(s["cid"]), s["t0"], s["t0"] + s["L"]), dtype=np.float32)
        return dict(wav=torch.from_numpy(np.ascontiguousarray(wav)), ids=torch.tensor(s["ids"]), is_audio=torch.tensor([c >= 0 for c in s["chunk_of"]]),
                    chunk_of=torch.tensor(s["chunk_of"]), labels=torch.tensor(s["labels"]), pos_weight=torch.tensor(s["pos_weight"], dtype=torch.float32),
                    labels_alt=torch.tensor(s["labels_alt"]), soft_w=torch.tensor(s["soft_w"], dtype=torch.float32),
                    lang=s["lang"], delay=delay, name=s["corpus"], id=f"{s['cid']}@{s['t0']}", n_text=s["st"].tokens, overflow=s["st"].overflow_tokens,
                    n_flush=s["n_flush"], K=s["K"], rounds=s["n_rounds"], n_eot=s["n_eot"], n_soft=s["n_soft"], n_spk=len(s["speakers"]))


def collate_cst(batch):
    """collate_streams + pos_weight·labels_alt·soft_w_full (B, L)(packing ROW_KEYS) + sem_only_mask(유효 위치 True: <SEM_END> 중립, <EOT> 학습)."""
    out = collate_streams(batch); B, L = out["ids"].shape
    pw = torch.zeros(B, L); la = torch.full((B, L), -100, dtype=torch.long); sw = torch.ones(B, L)
    for i, b in enumerate(batch):
        n = len(b["ids"]); pw[i, :n] = b["pos_weight"]; la[i, :n] = b["labels_alt"]; sw[i, :n] = b["soft_w"]
    out.update(pos_weight=pw, labels_alt=la, soft_w_full=sw, sem_only_mask=out["mask"].bool())
    for k in ("n_eot", "n_soft"): out[k] = sum(int(b.get(k, 0)) for b in batch)
    return out
