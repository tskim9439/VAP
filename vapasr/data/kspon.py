"""KsponSpeech 리더·정규화 (Stage 1). plans/stage1-mono-pilot.md §3.1·3.3.

- 오디오: raw PCM 16 kHz · 16-bit LE · mono · headerless. 홀수 바이트 파일은 마지막 1 바이트 제외. `.wav` 도 RIFF 헤더가 없다(실측).
- 전사(.trn): `경로 :: 텍스트`. 정규화는 (철자)/(발음) 중 **왼쪽 철자형**, 표지 b/ l/ o/ n/ u/ 제거, 기호 + * / 제거(단어는 유지), 구두점 제거.
  동봉 jsonl 은 발음형을 고른 사례가 있어 기준으로 쓰지 않는다.
"""
import os, re
from typing import List, Tuple, Optional
import numpy as np

SR, BPS = 16000, 2
DUAL = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")           # (철자)/(발음)
NOISE = re.compile(r"(?<!\S)[blonu]/(?!\S)")               # 단독 표지
FILLER = re.compile(r"(\S+?)/(?=\s|$)")                     # 아/ 그/ → 단어 유지
PUNCT = re.compile(r"(?<!\d)[.,](?!\d)|(?<=\d)[.,](?!\d)|(?<!\d)[.,](?=\d)|[?!]")   # 숫자 사이의 . , (0.1, 1,000) 는 남긴다

def normalize_kspon_v1(raw: str) -> Tuple[str, List[str]]:
    """asr-tn-v1.0.0: (정규화 문자열, malformed 사유 목록). 쌍마다 철자형에 digit·Latin 이 있으면 발음형, 아니면 철자형.
    malformed = 빈 철자/발음형, DUAL 치환 뒤 남는 괄호(중첩·미닫힘·홑괄호). 사유가 있으면 호출자가 quarantine 한다."""
    bad: List[str] = []
    def pick(m):
        a, b = m.group(1), m.group(2)                      # 괄호 안 공백도 원문 띄어쓰기의 일부라 strip 하지 않는다("(2월)/(이 월 )달에" → "이 월 달에")
        if not a.strip() or not b.strip(): bad.append("empty_side")
        return b if re.search(r"[0-9A-Za-z]", a) else a
    s = DUAL.sub(pick, raw)
    if "(" in s or ")" in s: bad.append("unpaired_paren")
    s = NOISE.sub(" ", s); s = PUNCT.sub("", s); s = FILLER.sub(r"\1", s); s = s.replace("+", "").replace("*", "")
    return re.sub(r"\s+", " ", s).strip(), bad

def normalize_kspon(raw: str, form: str = "auto") -> str:
    """form: auto(기본) — 숫자·라틴을 포함한 이중표기는 발음형, 나머지는 철자형 → 한글 전용(Qwen3-ASR·Nemotron 출력 규약과 동일, 2026-09-05 실측)
             spelling — 항상 왼쪽 철자형 | pron — 항상 오른쪽 발음형."""
    def pick(m):
        a, b = m.group(1), m.group(2)
        if form == "spelling": return a
        if form == "pron": return b
        return b if re.search(r"[0-9A-Za-z]", a) else a
    s = DUAL.sub(pick, raw)
    s = NOISE.sub(" ", s); s = PUNCT.sub("", s)                  # 구두점을 먼저 지워야 '뭐/.' 의 표지가 잡힌다
    s = FILLER.sub(r"\1", s); s = s.replace("+", "").replace("*", "")
    return re.sub(r"\s+", " ", s).strip()

def read_text(path: str) -> str:
    for enc in ("utf-8", "cp949"):
        try: return open(path, encoding=enc).read()
        except UnicodeDecodeError: continue
    raise RuntimeError(f"인코딩 판별 실패: {path}")

def read_trn(path: str) -> List[Tuple[str, str]]:
    """[(상대 경로, 원문 전사)]"""
    rows = []
    for line in read_text(path).splitlines():
        if " :: " not in line: continue
        p, t = line.split(" :: ", 1); rows.append((p.strip(), t.strip()))
    return rows

def resolve_path(root: str, rel: str) -> Optional[str]:
    """trn 의 상대 경로를 이 서버의 실제 경로로. eval 은 trn 에 'KsponSpeech_eval/' 접두사가 붙지만 서버에는 없다."""
    for cand in (rel, rel.replace("KsponSpeech_eval/", "", 1), os.path.join(os.path.basename(os.path.dirname(rel)), os.path.basename(rel))):
        p = os.path.join(root, cand)
        if os.path.exists(p): return p
    return None

def pcm_duration(path: str) -> Tuple[float, bool]:
    """(초, 홀수 바이트 여부) — 바이트 수만 본다."""
    n = os.path.getsize(path); return (n // BPS) / SR, n % 2 == 1

def read_pcm(path: str) -> Tuple[np.ndarray, bool]:
    """→ (float32 [-1,1] mono @16 kHz, odd)"""
    b = open(path, "rb").read(); odd = len(b) % 2 == 1
    if odd: b = b[:-1]
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0, odd
