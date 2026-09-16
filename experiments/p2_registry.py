#!/usr/bin/env python
"""Phase 2 토큰 registry 동결 — Qwen3-ASR tokenizer 에 Phase 1(+Phase 2) 특수 토큰을 추가해 실제 id 를 재고, E2 체크포인트 config 와 대조한 뒤
vapasr/data/schemas/phase2-registry.json 에 쓴다 (정본 output-phase2-lane-plan §1·§11: lane 1/2 = <SPK_A>/<SPK_B> 재사용, lane 3–6·<ONSET>·<EOT> 신규).
  python experiments/p2_registry.py --tokenizer $MXC_QWEN_ASR_DIR [--e2 /soundai/Model/VAPASR/hf-E2/final] [--write]"""
import os, sys, json, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from vapasr.data.dialogue_tokens import add_phase2_specials, PHASE2_SPECIALS, LANE_TOKENS
from vapasr.uslm.interleave_data import SPECIAL_TOKENS as PHASE1
ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", default=os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")); ap.add_argument("--e2", default="/soundai/Model/VAPASR/hf-E2/final"); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer); base_len = len(tok); ids = add_phase2_specials(tok)
rep = dict(tokenizer=a.tokenizer, base_vocab_len=base_len, after_len=len(tok), phase1={k: ids[k] for k in PHASE1}, phase2={k: ids[k] for k in PHASE2_SPECIALS}, lanes={f"lane{i+1}": ids[t] for i, t in enumerate(LANE_TOKENS)})
cfg = os.path.join(a.e2, "config.json")
if os.path.exists(cfg):
    c = json.load(open(cfg)); sp = c.get("sp_ids", {}); rep["e2_sp_ids"] = sp; rep["e2_vocab_rows"] = c.get("thinker", {}).get("text_config", {}).get("vocab_size")
    mism = {k: (sp[k], ids[k]) for k in sp if k in ids and sp[k] != ids[k]}; rep["e2_mismatch"] = mism
    assert not mism, f"E2 config sp_ids 와 tokenizer id 불일치: {mism}"
    assert max(ids.values()) < int(rep["e2_vocab_rows"] or 151936), "임베딩 행 부족"
    rep["e2_blocked_ids"] = c.get("blocked_ids")
out = dict(schema="vapasr-phase2-registry-1.0", ids={**{k: ids[k] for k in PHASE1}, **{k: ids[k] for k in PHASE2_SPECIALS}}, lanes=rep["lanes"], R=len(LANE_TOKENS), source=rep)
print(json.dumps(rep, ensure_ascii=False, indent=1))
if a.write:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vapasr", "data", "schemas", "phase2-registry.json")
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False); print("wrote", p)
