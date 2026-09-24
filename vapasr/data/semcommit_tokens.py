"""Semantic commit 토큰 registry — <SEM_END>(의미 단위 확정) (raw/inbox/streaming_asr_semantic_commit_plan.md §2·§13).

턴 종료는 Phase 2 의 <EOT>(151722) 하나로 통일한다(2026-09-24 결정): v0.2 의 <TURN_END>(151724)는 폐기 — 새 모델 tokenizer 에는 붙이지 않고,
v0.2 체크포인트(config.sem_registry 에 <TURN_END>)만 읽기·평가용으로 인식한다. Phase 1(mono) 학습은 기본이 <SEM_END> 만이고, mono 에서 턴 종료를
함께 가르칠 때(--turn-end)는 <EOT> 행을 쓴다(config.sem_registry 에 <EOT> 가 들어가고 decode 차단에서 뺀다).

id 정책: Phase 1(interleave_data.SPECIAL_TOKENS, 151705–151716) → Phase 2(dialogue_tokens.PHASE2_SPECIALS, 151717–151722) → SEM(151723) 순서로
한 번에 add_tokens 해서 id 를 고정한다. Phase 1 tokenizer 에 SEM 만 바로 붙이면 151717/151718 을 받아 FROZEN_REGISTRY 의 <SPK_3>/<SPK_4> 와 충돌하고,
SPECIAL_TOKENS 에 넣으면 from_legacy 의 sp_ids 대조와 Phase 2 id 가 모두 깨진다 → SPECIAL_TOKENS·FROZEN_REGISTRY 는 건드리지 않는다.
mono(lanes=0) 모델에서 Phase 2 id 는 '예약'만 된다(임베딩 행은 평균+잡음, decode 차단, config.sem_reserved 에 기록 → 나중 add_phase2_tokens 가 재초기화;
mono 에서 턴 종료로 학습한 <EOT> 는 예약에서 빠진다). 임베딩 여유 행(151936 − 151724 = 212)이 있어 resize 는 필요 없다.
모델 쪽 적용은 VapAsrForStreamingASR.add_semantic_tokens, 재로드는 vapasr.hf.load_tokenizer(config.sem_registry)."""
from typing import Dict, Iterable, List, Optional, Tuple
import torch
from ..uslm.interleave_data import SPECIAL_TOKENS as PHASE1_SPECIALS
from .dialogue_tokens import PHASE2_SPECIALS, FROZEN_REGISTRY

SEM_SPECIALS = ["<SEM_END>"]
SEM_REGISTRY = {"<SEM_END>": 151723}                           # Qwen3-ASR-0.6B tokenizer(기본 길이 151705) 기준 — FROZEN_REGISTRY 바로 뒤
TURN_TOKEN = "<EOT>"                                            # 턴 종료 토큰 = Phase 2 <EOT>(FROZEN_REGISTRY 151722)
LEGACY_TURN_TOKEN, LEGACY_TURN_ID = "<TURN_END>", 151724         # v0.2 체크포인트 전용(읽기·평가) — 새 tokenizer 에는 붙이지 않는다
ALL_SEM_SPECIALS = list(PHASE1_SPECIALS) + list(PHASE2_SPECIALS) + SEM_SPECIALS
BASE_VOCAB = 151705                                              # 첫 특수 토큰(<NEXT_AUDIO>) id = 기본 어휘 길이

def add_semantic_specials(tok) -> Dict[str, int]:
    """Phase 1 + Phase 2 + SEM 특수 토큰을 이 순서로 tokenizer 에 추가(이미 있으면 그대로) → 전체 {name: id}."""
    tok.add_tokens(ALL_SEM_SPECIALS, special_tokens=True)
    return {t: tok.convert_tokens_to_ids(t) for t in ALL_SEM_SPECIALS}

def assert_frozen_semantic(ids: Dict[str, int]) -> None:
    """실제 Qwen tokenizer 로 얻은 ids 가 동결 id(Phase 1/2 = FROZEN_REGISTRY, SEM = SEM_REGISTRY)와 같은지 검사."""
    want = {**FROZEN_REGISTRY, **SEM_REGISTRY}; bad = {k: (ids.get(k), v) for k, v in want.items() if ids.get(k) != v}
    assert not bad, f"semantic registry 불일치 (tokenizer, 동결): {bad}"

def sem_ids_of(registry: Dict[str, int]) -> Tuple[int, Optional[int]]:
    """config.sem_registry(학습한 이벤트 토큰) → (<SEM_END> id, 턴 종료 id 또는 None). 턴 종료 = <EOT>(mono 에서 --turn-end 로 학습), v0.2 체크포인트는 <TURN_END>.
    sp_ids 처럼 <EOT> 가 들어 있지만 턴으로 학습하지 않은 사전을 넘기지 않는다(그런 사전이면 <EOT> 를 턴으로 오인한다)."""
    turn = registry.get(TURN_TOKEN, registry.get(LEGACY_TURN_TOKEN))
    return int(registry["<SEM_END>"]), (None if turn is None else int(turn))

# ── 모델 쪽 순수 도우미 (add_semantic_tokens 가 쓴다; 모델 없이 단위 테스트)
def init_rows_mean_noise(W: torch.Tensor, rows: Iterable[int], upto: int, seed: int = 0) -> List[int]:
    """W (V,D) 의 rows 만 mean(W[:upto]) + 0.02·randn(seeded) 으로 덮는다(no_grad, 다른 행은 그대로). rows 는 오름차순으로 처리 → 같은 seed 면 결정적.
    upto 는 기본 어휘 길이(특수 토큰 앞) — 이미 학습된 특수 토큰 행이 평균에 섞이지 않게 한다(add_phase2_tokens 와 같은 규약)."""
    rows = sorted(set(int(r) for r in rows))
    if not rows: return rows
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mu = W[:upto].float().mean(0).cpu()
        for r in rows: W[r] = (mu + 0.02 * torch.randn(mu.shape, generator=g)).to(W.device, W.dtype)   # 잡음은 CPU generator → GPU 모델에서도 같은 값
    return rows

def semantic_blocked_ids(blocked_ids: Iterable[int], ids: Dict[str, int], lanes: int, block_reserved_phase2: bool = True, turn: bool = False) -> List[int]:
    """decode 차단 목록 갱신: <SEM_END> 는 항상 방출 가능(차단 해제), mono(lanes=0) 이면 학습되지 않은 Phase 2 예약 행(<SPK_3..6>,<ONSET>,<EOT>)을 차단.
    turn(mono 에서 <EOT> 를 턴 종료로 학습)이면 <EOT> 는 차단하지 않는다."""
    free = {n for n in SEM_SPECIALS} | ({TURN_TOKEN} if turn else set())
    out = set(int(b) for b in blocked_ids) - {ids[n] for n in free if n in ids}
    if block_reserved_phase2 and lanes == 0: out |= {ids[n] for n in PHASE2_SPECIALS if n in ids and n not in free}
    return sorted(out)

def new_special_rows(ids: Dict[str, int], known: Optional[Dict[str, int]]) -> List[int]:
    """이미 config.sp_ids 에 있는 이름을 뺀 새 행 id (known 에 있는 이름은 id 가 같아야 한다 — 다른 tokenizer 를 넘긴 실수 방지)."""
    known = known or {}; bad = {n: (ids[n], known[n]) for n in ids if n in known and int(known[n]) != int(ids[n])}
    assert not bad, f"tokenizer 와 config.sp_ids 의 기존 특수 토큰 id 가 다름 (tokenizer, config): {bad}"
    return sorted(int(ids[n]) for n in ids if n not in known)
