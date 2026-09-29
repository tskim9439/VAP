"""Artificial Analysis AA-WER v2 와 비교 가능한 영어 채점(근사) — VoxPopuli-Cleaned-AA 등 영어 평가의 보조 지표.

AA 방식(데이터셋 카드, https://artificialanalysis.ai/articles/aa-wer-v2): OpenAI Whisper EnglishTextNormalizer 를 바탕으로 한 정규화 + jiwer WER,
세트 안에서는 발화별 WER 을 오디오 길이로 가중 평균한다. 여기서 재현하는 것:
  1. 시각의 ':00' 제거('7:00pm' → '7pm') — Whisper 정규화가 '7 0 pm' 으로 만들기 전에. 선행 0 숫자열('007')은 한 자리씩 미리 띄운다
     (Whisper 는 쓴 '007' 을 '7' 로 줄이지만 말한 'zero zero seven' 은 '007' 로 둔다 — AA 의 '코드·식별자 선행 0 보존')
  2. Whisper EnglishTextNormalizer(vapasr/data/whisper_normalizer, openai/whisper 원본 그대로): 소문자, 괄호 안·필러 제거, 약어·호칭 표준화,
     수 단어 → 숫자, 영국 → 미국 철자(english.json)
  3. 숫자 한 자리씩 띄우기(숫자 묶음 차이 '1405 553 272' vs '1405553272' 무시, 선행 0 보존)
AA 가 공개하지 않은 추가 규칙(발화 기호 '+'·'_' 정규화, 추가 영/미 철자 쌍, 고유명사 허용 철자)은 넣지 않았다 — 리더보드 수치와 완전히 같지는 않다.
내부 주 지표는 textnorm.score_en 기반 micro WER(single_turn_eval.score_pair)이고, 이 모듈은 외부 비교용이다.
"""
import hashlib
import re
from pathlib import Path

AA_VERSION = "aa-wer-approx-v1"
_DIR = Path(__file__).resolve().parent
_NORMALIZER = None
_TIME00 = re.compile(r"(?<!\d)(\d{1,2}):00(?!\d)")
_LEAD0 = re.compile(r"(?<![\d.,])0\d+(?!\d|[.,]\d)")          # '007' 같은 선행 0 숫자열(소수 '0.5' 는 제외)
_DIGIT = re.compile(r"\d")
_WS = re.compile(r"\s+")


def _normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        from .whisper_normalizer import EnglishTextNormalizer
        _NORMALIZER = EnglishTextNormalizer()
    return _NORMALIZER


def aa_normalize(text: str) -> str:
    """AA 비교용 정규화 문자열(공백 구분 토큰)."""
    t = _TIME00.sub(r"\1", text or "")
    t = _LEAD0.sub(lambda m: " ".join(m.group(0)), t)     # Whisper 는 쓴 '007' 을 '7' 로, 말한 'zero zero seven' 은 '007' 로 둔다 → 미리 띄워 선행 0 보존
    t = _normalizer()(t)
    t = _DIGIT.sub(lambda m: " " + m.group(0) + " ", t)
    return _WS.sub(" ", t).strip()


def _edit_counts(ref, hyp):
    """토큰 목록 편집 카운트(S/D/I). rapidfuzz 가 있으면 그것(single_turn_eval.score_pair 와 같은 경로), 없으면 DP."""
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        Levenshtein = None
    c = dict(substitutions=0, deletions=0, insertions=0)
    if Levenshtein is not None:
        names = {"replace": "substitutions", "delete": "deletions", "insert": "insertions"}
        for op in Levenshtein.editops(ref, hyp):
            c[names[op.tag]] += 1
        return c
    n, m = len(ref), len(hyp)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]              # (총, S, D, I)
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]; continue
            s, dl, ins = d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]
            d[i][j] = min((s[0] + 1, s[1] + 1, s[2], s[3]), (dl[0] + 1, dl[1], dl[2] + 1, dl[3]), (ins[0] + 1, ins[1], ins[2], ins[3] + 1))
    _, c["substitutions"], c["deletions"], c["insertions"] = d[n][m]
    return c


def aa_counts(reference: str, hypothesis: str) -> dict:
    """한 발화의 AA 정규화 WER 카운트."""
    ref, hyp = aa_normalize(reference).split(), aa_normalize(hypothesis).split()
    c = _edit_counts(ref, hyp)
    c.update(n_ref=len(ref), n_hyp=len(hyp), errors=c["substitutions"] + c["deletions"] + c["insertions"])
    c["rate"] = c["errors"] / len(ref) if ref else None
    return c


def aa_summary(rows) -> dict:
    """rows: reference·hypothesis·audio_s 가 있는 dict 들(한 세트·한 조건). micro WER 과 AA 방식(오디오 길이 가중 발화별 WER 평균).
    정규화 뒤 참조가 빈 발화는 가중 평균에서 빼고 empty_ref 로 센다."""
    rows = list(rows)
    e = n = s = dl = ins = empty = 0
    wsum = dsum = 0.0
    for r in rows:
        c = aa_counts(r["reference"], r["hypothesis"])
        e += c["errors"]; n += c["n_ref"]; s += c["substitutions"]; dl += c["deletions"]; ins += c["insertions"]
        if c["n_ref"]:
            wsum += float(r["audio_s"]) * c["errors"] / c["n_ref"]; dsum += float(r["audio_s"])
        else:
            empty += 1
    return dict(utterances=len(rows), empty_ref=empty, substitutions=s, deletions=dl, insertions=ins, errors=e, n_ref=n,
                wer_micro=e / n if n else None, wer_duration_weighted=wsum / dsum if dsum else None, audio_s=dsum)


def aa_fingerprint() -> dict:
    """채점 재현 지문: 이 파일 + 벤더링한 Whisper 정규화 파일들의 sha256."""
    files = [_DIR / "aa_wer.py"] + sorted((_DIR / "whisper_normalizer").glob("*.py")) + [_DIR / "whisper_normalizer" / "english.json"]
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return dict(aa_version=AA_VERSION, aa_code_sha256=h.hexdigest())
