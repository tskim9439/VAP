"""출력 형식 플래그(outfmt-v1) — 정규형(normalized) ↔ 표기형(written)을 system prompt 텍스트로 고른다.

  normalized  빈 system 칸 = 지금까지의 모든 학습·평가 prefix 와 바이트 동일(<|im_start|>system\\n<|im_end|>\\n<|im_start|>assistant\\n).
              EN = asr-tn-v1 lexical(소문자·구두점 없음·숫자는 읽는 형태), KO = asr-tn-v1(구두점 없음, 숫자는 원본 전사의 읽는 형태).
  written     system 칸 = WRITTEN_INSTRUCTION(동결). 구두점·대소문자·아라비아 숫자를 쓰는 표기형(KO = Qwen3-ASR display_text, EN = pnc_text).

어휘·특수 토큰은 바꾸지 않는다(vocab_sha256·동결 id 불변). 지시문 토큰은 사전학습 임베딩이다.
system 칸은 Qwen3-ASR 의 context(문맥 바이어싱) 칸이기도 하다 — 나중에 문맥을 함께 넣게 되면 '첫 줄 = 형식 지시문, 빈 줄 뒤 = context' 로 나눈다.
지시문 문자열을 바꾸면 OUTFMT_PROTOCOL 을 올린다(학습 지문 config.semcommit['output_format'] 과 모델 config.output_formats 에 기록된다)."""
import hashlib, json
from typing import Dict, List

OUTFMT_PROTOCOL = "outfmt-v1"
NORMALIZED, WRITTEN = "normalized", "written"
FORMATS = (NORMALIZED, WRITTEN)
WRITTEN_INSTRUCTION = "Output format: written (punctuation, capitalization, numerals)."
_SYSTEM = {NORMALIZED: "", WRITTEN: WRITTEN_INSTRUCTION}


def header_text(fmt: str = NORMALIZED) -> str:
    """chat 머리 문자열(system + assistant 시작). normalized 는 기존 _pre 문자열 그대로."""
    if fmt not in _SYSTEM: raise ValueError(f"unknown output format: {fmt!r} (지원: {FORMATS})")
    return f"<|im_start|>system\n{_SYSTEM[fmt]}<|im_end|>\n<|im_start|>assistant\n"


def header_ids(tok, fmt: str = NORMALIZED) -> List[int]:
    """형식의 머리 토큰 id. 전체 문자열을 한 번에 토큰화한다(Qwen chat template 과 같다). Qwen3-ASR tokenizer 에서 written 은
    normalized[:3] + tok(지시문) + normalized[3:] 과 같다(tests/test_semcommit_format.py 가 실 tokenizer 로 확인)."""
    return list(tok(header_text(fmt), add_special_tokens=False)["input_ids"])


def header_sha256(tok, fmt: str = WRITTEN) -> str:
    return hashlib.sha256(json.dumps(header_ids(tok, fmt)).encode()).hexdigest()


def output_formats_config(tok) -> Dict[str, dict]:
    """모델 config.output_formats — 평가·추론이 표기형 prefix 를 tokenizer 없이 복원할 수 있게(학습에 쓴 형식만)."""
    return {WRITTEN: dict(instruction=WRITTEN_INSTRUCTION, header_ids=header_ids(tok, WRITTEN), protocol=OUTFMT_PROTOCOL)}
