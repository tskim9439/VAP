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

def load_frozen_registry(path: str = None) -> Dict[str, int]:
    """동결된 Phase 2 registry(vapasr/data/schemas/phase2-registry.json). experiments/p2_registry.py 가 tokenizer 실측으로 만든다."""
    import json, os
    path = path or os.path.join(os.path.dirname(__file__), "schemas", "phase2-registry.json")
    return {k: int(v) for k, v in json.load(open(path))["ids"].items()}
