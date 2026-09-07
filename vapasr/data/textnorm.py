"""텍스트 정규화 규약 `asr-tn-v1.0.0` 구현 — 모든 코퍼스의 학습 타깃과 채점을 한 규약으로.
정본: wiki/outputs/output-asr-tn-v1-spec.md (이 파일은 그 문서의 golden 출력을 만족해야 한다. 라이브러리 기본 동작은 정본이 아니다.)

  EN 타깃(LibriSpeech): 태그 제거 → NFKC → 소문자 → ’→' (단어 내부만 유지) → 구두점·하이픈·대시·slash → 공백. **숫자는 변환하지 않는다**:
      LibriSpeech 는 발음 단어로 전사돼 있어야 하므로 digit 가 남으면 그 행을 quarantine 한다(1984 = 연도? 정수? 음성 없이 결정 불가).
  EN 채점(score_en) 과 숫자를 명시적으로 허용한 코퍼스: 지원 문법(정수·천단위·소수·퍼센트·USD 정수·서수)에 **통째로** 맞는 토큰만
      canonical spoken form 으로 바꾸고, 그 밖의 숫자(연도 문맥·시각·전화·분수·소수 통화·단위 결합 …)는 부분 변환 없이 원형을 남긴다.
  KO 타깃(KsponSpeech): (철자)/(발음) 쌍마다 철자형에 digit·Latin 이 있으면 발음형, 아니면 철자형. 표지 b/ l/ o/ n/ u/ 와 filler 의 slash, + * 구두점 제거,
      원문 띄어쓰기 유지. 이중표기 밖 digit·malformed 표기·허용 문자 밖 문자 → quarantine. 독립 Latin 은 **원형(대소문자 포함) 유지**.
  채점: score_en / score_ko(공백 포함 CER-official, 공백 제거 CER-nospace). 학습 타깃에 score_* 를 쓰지 않는다.

target_* 는 문자열만 돌려주고, quarantine 판정은 target_flags() 로 따로 한다(빈 문자열·digit·허용 문자 밖·malformed).
버전: 타깃 문자열이나 token ID 가 한 건이라도 바뀌면 minor 이상을 올리고 manifest·정렬을 새 경로에 만든다(patch 는 문서·메시지·audit 만).
"""
import re, os, hashlib, unicodedata
from typing import Optional, Set, Dict, Tuple
from .kspon import normalize_kspon_v1
from .nikl import normalize_nikl
try:
    from num2words import num2words
    from importlib.metadata import version as _pkg_version
    NUMERIC_BACKEND_VERSION = _pkg_version("num2words")
except Exception as e:                                                   # 의존성 없으면 즉시 실패(동결 관문 1)
    raise ImportError("asr-tn-v1.0.0 은 num2words==0.5.14 가 필요합니다: pip install num2words==0.5.14") from e

TEXTNORM_VERSION = "asr-tn-v1.1.0"        # v1.1.0(2026-09-07): 새 코퍼스 파서 nikl·switchboard·mnsc 추가. 기존 코퍼스 타깃은 v1.0.0 과 동일
TEXTNORM_ID_SHORT = "asr-tn-v1"                                          # 정렬 경로 등에 쓰는 major 식별자
NUMERIC_BACKEND_PINNED = "0.5.14"
NUMERIC_CORPORA_EN: Set[str] = set()                                    # v1.1.0: 숫자 표기를 허용한 EN 학습 코퍼스 없음(LibriSpeech·Switchboard·MNSC 모두 digit 0, 남으면 quarantine)

_TAG = re.compile(r"<[^>\s]{1,20}>")                                     # <en-US>, <ko-KR>, 특수 토큰
_WS = re.compile(r"\s+")
def _clean(t: str) -> str: return _WS.sub(" ", unicodedata.normalize("NFKC", _TAG.sub(" ", t))).strip()

# ───────────────────────── EN 숫자 canonical form (지원 문법만, 토큰 단위 전체 일치) ─────────────────────────
_INT = r"(?:0|[1-9]\d{0,8})"                                             # 0–999,999,999, 선행 0 불허
_GRP = r"(?:[1-9]\d{0,2}(?:,\d{3}){1,2})"                                # 천 단위 쉼표(정확히 3 자리 grouping), ≤ 999,999,999
_NUM = rf"(?:{_GRP}|{_INT})"
_DEC = rf"{_NUM}\.\d{{1,6}}"
_RE_INT, _RE_DEC = re.compile(rf"^{_NUM}$"), re.compile(rf"^({_NUM})\.(\d{{1,6}})$")
_RE_PCT = re.compile(rf"^({_NUM}(?:\.\d{{1,6}})?) ?%$"); _RE_USD = re.compile(rf"^\$({_NUM})$"); _RE_ORD = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")
_DIGIT_WORD = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

def _cardinal(s: str) -> str:
    """정수 문자열 → American cardinal, 'and'·쉼표 없이, 하이픈은 공백."""
    w = num2words(int(s.replace(",", ""))); return _WS.sub(" ", w.replace(",", " ").replace("-", " ").replace(" and ", " ")).strip()
def _ordinal_suffix(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
def canon_number_en(tok: str) -> Optional[str]:
    """지원 숫자 토큰 → spoken form. 지원 문법에 통째로 맞지 않으면 None(부분 변환 금지)."""
    if _RE_INT.match(tok): return _cardinal(tok)
    m = _RE_DEC.match(tok)
    if m: return _cardinal(m.group(1)) + " point " + " ".join(_DIGIT_WORD[d] for d in m.group(2))
    m = _RE_PCT.match(tok)
    if m: return (canon_number_en(m.group(1)) or "") + " percent"
    m = _RE_USD.match(tok)
    if m: return _cardinal(m.group(1)) + (" dollar" if int(m.group(1).replace(",", "")) == 1 else " dollars")
    m = _RE_ORD.match(tok)
    if m and m.group(2) == _ordinal_suffix(int(m.group(1))) and not (len(m.group(1)) == 2 and m.group(1)[0] == "0"):
        return num2words(int(m.group(1)), to="ordinal").replace("-", " ")
    return None

_PEEL = re.compile(r"""^(?P<pre>["'(\[]*)(?P<core>.*?)(?P<post>[.,;:!?"')\]]*)$""", re.S)
def canon_numbers_en(t: str) -> str:
    """공백 토큰 단위로 지원 숫자만 바꾼다. '5 %' 처럼 숫자와 % 사이 공백 1 개는 허용."""
    t = re.sub(r"(\d) %", r"\1%", t); out = []
    for tok in t.split(" "):
        if not any(c.isdigit() for c in tok): out.append(tok); continue
        m = _PEEL.match(tok); c = canon_number_en(m.group("core"))
        out.append(m.group("pre") + c + m.group("post") if c is not None else tok)
    return " ".join(out)

# ───────────────────────── EN ─────────────────────────
_EN_PUNCT = re.compile(r"[^a-z0-9'\s]")                                   # 아포스트로피와 digit(quarantine 판정용) 외 전부 → 공백
def _en_surface(t: str, numbers: bool) -> str:
    t = _clean(t).lower().replace("’", "'").replace("‘", "'")
    if numbers: t = canon_numbers_en(t)
    t = re.sub(r"[-–—/]", " ", t)                                         # 하이픈·대시·slash → 공백 (twenty-one → twenty one)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)                               # 단어 내부 아포스트로피만 유지
    return _WS.sub(" ", _EN_PUNCT.sub(" ", t)).strip()
def target_en(t: str, corpus: str = "librispeech") -> str:
    """EN 학습 타깃. 숫자 변환은 NUMERIC_CORPORA_EN 에 등록된 코퍼스에서만(v1.0.0: 없음). 남은 digit 는 target_flags 가 quarantine 으로 표시."""
    return _en_surface(t, numbers=corpus in NUMERIC_CORPORA_EN)
def score_en(t: str) -> str:
    """EN 채점 정규화 — 참조·가설·대조군 출력에 동일 적용. 지원 숫자만 canonical form, 미지원은 원형."""
    return _en_surface(t, numbers=True)

# ───────────────────────── KO ─────────────────────────
_KO_PUNCT = re.compile(r"[^\w\s]")
_KO_ALLOWED = re.compile(r"^[가-힣ᄀ-ᇿ㄰-㆏A-Za-z ]*$")
KO_PARSERS = {"kspon": lambda r: normalize_kspon_v1(r), "nikl": lambda r: normalize_nikl(r)}   # corpus → (텍스트, malformed 사유)
def target_ko(raw: str, corpus: str = "kspon") -> str:
    """KO 학습 타깃. kspon: 이중표기 선택 + 표지 제거. nikl: original_form 표지 제거(익명화·불명확은 quarantine). 그 외(aihub 등): 태그·구두점 제거만. 독립 Latin 은 원형 유지."""
    t = KO_PARSERS[corpus](raw)[0] if corpus in KO_PARSERS else raw
    return _WS.sub(" ", _KO_PUNCT.sub(" ", _clean(t))).strip()
def score_ko(t: str, spaces: bool) -> str:
    """CER-official(spaces=True, 원문 공백 포함) / CER-nospace(spaces=False). 숫자·Latin 을 한글로 추측 변환하지 않는다."""
    t = _WS.sub(" ", _KO_PUNCT.sub(" ", _clean(t))).strip()
    return t if spaces else t.replace(" ", "")

# ───────────────────────── 공통 ─────────────────────────
def target(t: str, lang: str, corpus: Optional[str] = None) -> str:
    return target_ko(t, corpus or "kspon") if lang == "Korean" else target_en(t, corpus or "librispeech")

def target_flags(t: str, lang: str, raw: Optional[str] = None, corpus: Optional[str] = None) -> Set[str]:
    """quarantine 사유 집합(비어 있으면 학습 가능). empty · digit · charset(허용 문자 밖) · control · malformed_dual(kspon) · not_idempotent."""
    f: Set[str] = set()
    if lang == "Korean" and raw is not None and (corpus or "kspon") in KO_PARSERS:
        for reason in KO_PARSERS[corpus or "kspon"](raw)[1]: f.add({"empty_side": "malformed_dual", "unpaired_paren": "malformed_dual"}.get(reason, reason))
    if not t: f.add("empty"); return f
    if any(unicodedata.category(c) == "Cc" for c in t): f.add("control")
    if re.search(r"\d", t): f.add("digit")
    if lang == "Korean":
        if not _KO_ALLOWED.match(t): f.add("charset")
        if target_ko(t, "aihub") != t: f.add("not_idempotent")
    else:
        if not re.match(r"^[a-z' ]*$", t) or re.search(r"(?<![a-z])'|'(?![a-z])", t): f.add("charset")
        if target_en(t, "librispeech") != t: f.add("not_idempotent")
    return f

def has_standalone_latin(t: str) -> bool: return re.search(r"[A-Za-z]", t) is not None

# ───────────────────────── fingerprint ─────────────────────────
def _sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths: h.update(open(p, "rb").read())
    return h.hexdigest()
def code_sha256() -> str:
    d = os.path.dirname(os.path.abspath(__file__)); return _sha256_files([os.path.join(d, "textnorm.py"), os.path.join(d, "kspon.py")])
def tokenizer_sha256(tok_dir: str) -> str:
    """Qwen3-ASR 로컬 디렉토리의 tokenizer 정의(vocab.json·merges.txt·tokenizer_config.json, 있으면 tokenizer.json)."""
    fs = [os.path.join(tok_dir, f) for f in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json") if os.path.exists(os.path.join(tok_dir, f))]
    assert fs, f"tokenizer 파일 없음: {tok_dir}"; return _sha256_files(fs)
def fingerprint(tok_dir: Optional[str] = None) -> Dict[str, str]:
    """코드 쪽 fingerprint. manifest 빌더가 source_transcript/manifest/quarantined_ids/git 을 채워 stats.json 에 기록한다."""
    fp = dict(textnorm_version=TEXTNORM_VERSION, textnorm_sha256=code_sha256(), numeric_backend_version=f"num2words {NUMERIC_BACKEND_VERSION}")
    if tok_dir: fp["tokenizer_json_sha256"] = tokenizer_sha256(tok_dir)
    return fp
def check_fingerprint(fp: Optional[Dict[str, str]], where: str, allow_legacy_env: str = "VAPASR_ALLOW_LEGACY_TN") -> None:
    """정렬기·학습기 시작 전 검사: fingerprint 가 없거나 코드와 다르면 실패. 동결 전 산출물(align2 등) 재현은 env {allow_legacy_env}=1 로만 허용."""
    if not fp or "textnorm_version" not in fp:
        if os.environ.get(allow_legacy_env) == "1": print(f"!! {where}: textnorm fingerprint 없음 — 동결 전(legacy) 타깃으로 진행({allow_legacy_env}=1)", flush=True); return
        raise RuntimeError(f"{where}: textnorm fingerprint 없음(동결 전 산출물). 새 규약으로 manifest·정렬을 다시 만들거나 {allow_legacy_env}=1 로 재현 실행")
    mine = fingerprint()
    for k in ("textnorm_version", "textnorm_sha256", "numeric_backend_version"):
        if fp.get(k) != mine[k]: raise RuntimeError(f"{where}: textnorm fingerprint 불일치 {k}: 산출물 {fp.get(k)} ≠ 코드 {mine[k]}")
