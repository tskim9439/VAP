"""Phase 2 토큰 registry — 정본 §1: lane 1/2 = 기존 <SPK_A>/<SPK_B>, lane 3–6 = <SPK_3..6> 신규, 구조 토큰 <ONSET>/<EOT> 신규.
Phase 1 registry(vapasr.uslm.interleave_data.SPECIAL_TOKENS)는 바꾸지 않고 그 위에 더한다. 실제 ID 는 tokenizer 가 부여하며
HF config `sp_ids` 로 동결하는 migration 은 D1 항목이다(정본 §11)."""
from typing import Dict
from ..uslm.interleave_data import SPECIAL_TOKENS as PHASE1_SPECIALS
from .dialogue_interleave import LaneSpecials

R_LANES = 6
LANE_TOKENS = ["<SPK_A>", "<SPK_B>"] + [f"<SPK_{i}>" for i in range(3, R_LANES + 1)]
PHASE2_SPECIALS = [t for t in LANE_TOKENS if t not in PHASE1_SPECIALS] + ["<ONSET>", "<EOT>"]

def add_phase2_specials(tok) -> Dict[str, int]:
    """Phase 1 특수 토큰 + Phase 2 토큰을 tokenizer 에 추가(이미 있으면 그대로) → {name: id}."""
    tok.add_tokens(PHASE1_SPECIALS + PHASE2_SPECIALS, special_tokens=True)
    return {t: tok.convert_tokens_to_ids(t) for t in PHASE1_SPECIALS + PHASE2_SPECIALS}

def lane_specials_of(ids: Dict[str, int], R: int = R_LANES) -> LaneSpecials:
    return LaneSpecials(next_audio=ids["<NEXT_AUDIO>"], empty_audio=ids["<EMPTY_AUDIO>"], lanes=[ids[t] for t in LANE_TOKENS[:R]], onset=ids["<ONSET>"], eot=ids["<EOT>"])

FROZEN_REGISTRY = {"<NEXT_AUDIO>": 151705, "<EMPTY_AUDIO>": 151706, "<SPK_A>": 151707, "<SPK_B>": 151708, **{f"<DELAY_{d}>": 151708 + d for d in range(1, 9)},
                   "<SPK_3>": 151717, "<SPK_4>": 151718, "<SPK_5>": 151719, "<SPK_6>": 151720, "<ONSET>": 151721, "<EOT>": 151722}   # 2026-09-16 실측(experiments/p2_registry.py, Qwen3-ASR-0.6B tokenizer, E2 config 일치)

def load_frozen_registry(path: str = None) -> Dict[str, int]:
    """동결된 Phase 2 registry. 기본은 코드 상수 FROZEN_REGISTRY 이며, schemas/phase2-registry.json 이 있으면 그것과 같은지 검사한다(서버 사본의 schemas/ 가 root 소유라 파일 동기화가 막혀 상수를 정본으로 둔다)."""
    import json, os
    path = path or os.path.join(os.path.dirname(__file__), "schemas", "phase2-registry.json")
    if os.path.exists(path):
        ids = {k: int(v) for k, v in json.load(open(path))["ids"].items()}
        assert ids == FROZEN_REGISTRY, f"phase2-registry.json 과 FROZEN_REGISTRY 불일치: {ids} vs {FROZEN_REGISTRY}"
    return dict(FROZEN_REGISTRY)
