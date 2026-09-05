"""텍스트 정규화 — 모든 코퍼스의 학습 타깃과 채점을 한 규약으로 (plans/stage1-mono-pilot.md §3.3).

규약은 사전학습 모델의 출력 표기에 맞춘다(2026-09-05 실측, Qwen3-ASR·Nemotron RNN-T 모두):
  EN: 숫자는 **단어**("three hundred dollars"), 아포스트로피 유지("don't"). 대문자·구두점은 모델이 내지만
      LibriSpeech(주 데이터)에 없어 타깃에서는 뺀다 — 채점에서 양쪽 모두 같은 규칙으로 지운다.
  KO: 숫자는 **한글 읽기**("십만 원", "이십육 일"). KsponSpeech 는 (철자)/(발음) 이중표기의 발음형이 곧 읽기이므로
      숫자·라틴을 포함한 이중표기만 발음형, 나머지는 철자형 → 한글 전용(AI Hub 와 동일). 구두점 제거, 띄어쓰기는 원문.

target_*: 학습 타깃(약한 정규화)   score_*: 지표용(강한 정규화, 가설·대조군 출력에도 동일 적용)
"""
import re, unicodedata
from typing import Optional
from .kspon import normalize_kspon

_TAG = re.compile(r"<[^>]{1,20}>")                                   # Nemotron <en-US>/<ko-KR>, 특수 토큰
_APOS = re.compile(r"(?<=\w)['’](?=\w)")                             # 단어 내부 아포스트로피만 보존
_EN_PUNCT = re.compile(r"[^a-z0-9'\s]")                              # 아포스트로피 외 전부
_KO_PUNCT = re.compile(r"[^\w\s]")

def _num2words_en(t: str) -> str:
    """숫자 → 단어. '%' → percent, '$5' → five dollars. num2words 가 없으면 그대로(설치: pip install num2words)."""
    try: from num2words import num2words
    except ImportError: return t
    t = re.sub(r"\$\s?(\d+(?:\.\d+)?)", lambda m: f"{num2words(float(m.group(1)) if '.' in m.group(1) else int(m.group(1)))} dollars", t)
    t = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: f"{num2words(float(m.group(1)) if '.' in m.group(1) else int(m.group(1)))} percent", t)
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), t)
    t = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", lambda m: num2words(float(m.group(0).replace(",", "")) if "." in m.group(0) else int(m.group(0).replace(",", ""))), t)
    return t.replace("-", " ")                                        # twenty-one → twenty one

def target_en(t: str) -> str:
    t = unicodedata.normalize("NFKC", _TAG.sub(" ", t)).lower()
    t = _APOS.sub("'", t); t = _num2words_en(t)
    t = re.sub(r"[-–—/]", " ", t)                                     # mm-mm → mm mm, and/or → and or
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)                           # 따옴표로 쓰인 아포스트로피 제거
    t = _EN_PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def target_ko(raw: str, corpus: str = "kspon") -> str:
    """kspon: (철자)/(발음) 자동 선택 + 표지 제거. 그 외(aihub 등): 구두점 제거만."""
    t = normalize_kspon(raw, form="auto") if corpus == "kspon" else raw
    t = unicodedata.normalize("NFKC", _TAG.sub(" ", t)); t = _KO_PUNCT.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def target(t: str, lang: str, corpus: Optional[str] = None) -> str:
    return target_ko(t, corpus or "kspon") if lang == "Korean" else target_en(t)

def score_en(t: str) -> str:       return target_en(t)                  # 대조군 출력의 대문자·구두점·숫자도 같은 규칙으로
def score_ko(t: str, spaces: bool) -> str:
    t = unicodedata.normalize("NFKC", _TAG.sub(" ", t)); t = _KO_PUNCT.sub(" ", t); t = re.sub(r"\s+", " ", t).strip()
    return t if spaces else t.replace(" ", "")
