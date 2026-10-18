"""표기형(written, outfmt-v1) semcommit 행 — 정규형 words.jsonl 행에서 구두점·대소문자·아라비아 숫자를 쓰는 타깃을 만든다(SemCommitDataset 이 적재 때 호출).

written_row(r, tok) → (row_w, word_map) | (None, 사유).  word_map: None = 단어 index 그대로(KO), list = lexical 단어 i → 표기형 단어 index(단위 안쪽 단어는 None, EN).
  KO  speechlm 행의 원 tokens 그대로(= Qwen3-ASR display_text; 단어는 이미 끝 구두점 토큰을 span 에 포함한 lexical 단어). normalize_ko_row 를 건너뛴다.
      display_text 가 없으면(kspon 등 asr-tn-v1 행) ko_no_written_source, 공백을 뺀 decode(tokens) ≠ display_text 면 ko_display_mismatch.
  EN  pnc_text(segment raw_text 를 이은 것; voxpopuli·yodas·mnsc 만 있다)의 공백 토큰 = 단위. 단위의 lexical 읽기(target_en; 숫자는
      canon_number_en · num2words · en_GB('and' 읽기) · 연도 · 서수 · % → percent 후보)를 이어 붙인 것이 행의 lexical 단어열과 **정확히** 같게 DP 로 배정한다.
      대소문자·구두점은 pnc 그대로, 숫자는 읽기가 정확히 맞을 때만 되살린다. 한 곳이라도 실패하면 표기형 없음(정규형만 쓴다):
      en_no_pnc · en_align · en_no_case(영단어 4개 이상인데 대문자 0 — yodas 소문자 raw) · en_word_cover · en_selfcheck · written_decode.
      붙임 오류('layer.Many')는 문장 끝 부호 뒤에 공백을 넣어 고친다. 순수 구두점 토큰: 여는 부호('(' '[' '{' '“' '‘' '«', 짝수 번째 '"')는
      다음 단위 앞에 공백 없이 붙인다('( we' → '(we' — <SEM_END> 가 다음 절의 여는 부호 뒤로 가지 않게). 나머지('—' ')' '.' 닫는 '"')는 앞 단위에
      ' T' 로 붙는다(첫 단위 앞이면 앞머리). 빌더 규칙(lexical_words: 구두점만인 단어는 앞 단어에 합친다)과 같은 단어가 나오는 배치다(en_selfcheck).
토큰 시각: 단위 토큰 = 그 단위 lexical 토큰 시각을 순서대로 펼친 것(semcommit_tn._spread — 마지막 토큰 = 단위 끝 시각).
  digit hold: 숫자가 든 단위(EN)·단어(KO)는 모든 토큰을 단위 끝 시각에 둔다 → 숫자는 그 음성 구간이 끝난 뒤 한 번에 나온다(스트리밍 ITN hold).
  단어 끝 시각(= <SEM_END>·B/N 결정 청크): EN 은 구성상 정규형과 같다(단위 끝 = 마지막 lexical 토큰 시각). KO 는 끝 구두점 토큰 시각이 앞 lexical 토큰과
  같을 때만 같다(데이터 성질 — 실데이터 스냅샷 20260929-0928 에서 차이 0). 다른 행은 SemCommitDataset 이 버린다(written_cand_time_shift ·
  written_tail_shift) → 학습에 쓰는 표기형은 SEM 청크·결정 청크·마지막 방출 청크가 정규형과 같고, K·정규형 시퀀스는 플래그와 무관하다.
<SEM_END>·B/N 결정 위치 규칙은 그대로(단어 마지막 토큰 바로 뒤) — 표기형 단어 span 이 끝 구두점까지 포함하므로 '구두점 뒤'가 된다.
  평가 주의(구두점 지름길): 표기형에서는 A 단어의 대부분이 문장 끝 부호로 끝난다(mxc 표본 A 중 EN 97.8 %·KO 93.2 %, 역으로 문장 끝 부호 단어 중 A
  EN 96.9 %·KO 95.7 %). 표기형 모드의 SEM 지표는 구두점 예측 품질을 크게 반영하므로 정규형 지표와 따로, 같은 스트림의 짝 비교로 보고한다.
remap_labels: after_word 를 word_map 으로 옮기고, 단위 안쪽 단어의 후보는 버린다(사유별 수를 돌려준다).
지문: written_rules_sha256() = 빌더 이름·규칙 상수·정규식 + 고정 탐침 문장의 en_units/number_readings 결과(textnorm.target_en·num2words 판본 차이도 잡힌다)
  → output_format_settings 의 rules_sha256. 규칙을 바꾸면 WRITTEN_BUILDER 도 올린다(tests/test_semcommit_format.py 의 WRITTEN_GOLDEN 이 잡는다)."""
import functools
import hashlib
import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from num2words import num2words

from .output_format import OUTFMT_PROTOCOL, WRITTEN_INSTRUCTION, header_sha256
from .semcommit_tn import _spread
from .semcommit_words import COMMA, EDGE_PUNCT, FINAL, _pieces, lexical_words, piece_bytes, punct_tags, split_words
from .textnorm import _PEEL, canon_number_en, target_en

WRITTEN_BUILDER = "semcommit-written-v1"
DIGIT_HOLD = True
MIN_CASE_WORDS = 4                                   # 영단어가 이만큼 이상인데 대문자가 없으면 대소문자 정보가 없는 raw(en_no_case)
_DIGIT = re.compile(r"\d")
_WS = re.compile(r"\s+")
_GLUE = re.compile(r"(?<=[a-z])([.?!])(?=[A-Z][a-z])")
_INT_TOKEN = re.compile(r"\d{1,9}|\d{1,3}(?:,\d{3})+")
_ORD_TOKEN = re.compile(r"(\d{1,2})(st|nd|rd|th)")
_OPEN = frozenset("([{“‘«")                         # 여는 부호(순수 구두점 토큰일 때 다음 단위에 붙인다); '"' 는 짝수 번째(여는 쪽)만


def _row_bytes(tok, ids) -> Optional[List[bytes]]:
    if not callable(getattr(tok, "convert_ids_to_tokens", None)): return None
    bs = [piece_bytes(p) for p in _pieces(tok, list(ids))]
    return None if any(b is None for b in bs) else bs


def _decode(bs: List[bytes]) -> Optional[str]:
    try: return b"".join(bs).decode("utf-8")
    except UnicodeDecodeError: return None


# ── KO ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _written_ko(r: dict, tok, digit_hold: bool):
    disp = r.get("display_text")
    if not disp: return None, "ko_no_written_source"
    toks = r["tokens"]; bs = _row_bytes(tok, [int(t[0]) for t in toks]); txt = _decode(bs) if bs is not None else None
    if txt is None: return None, "written_decode"
    if _WS.sub("", txt) != _WS.sub("", disp): return None, "ko_display_mismatch"
    if not digit_hold: return r, None
    hold = {j: float(w["end_time"]) for w in r["words"] if _DIGIT.search(w["text"]) for j in range(int(w["a"]), int(w["b"]))}
    if not hold: return r, None
    return dict(r, tokens=[[t[0], hold[j]] if j in hold else t for j, t in enumerate(toks)]), None


# ── EN ───────────────────────────────────────────────────────────────────────────────────────────────────────
def en_corpus(set_name: Optional[str]) -> str:
    """words 행의 set 이름 → textnorm 코퍼스 키(숫자 변환 여부만 다르다: yodas 만 NUMERIC_CORPORA_EN)."""
    s = (set_name or "").lower()
    return next((c for c in ("yodas", "voxpopuli", "mnsc") if c in s), "librispeech")


@functools.lru_cache(maxsize=1 << 16)
def _key(T: str, corpus: str) -> Tuple[str, ...]:
    return tuple(target_en(T, corpus).split())


def _words_of(s: str) -> Tuple[str, ...]:
    return tuple(re.sub(r"[-,]", " ", s.lower()).split())


@functools.lru_cache(maxsize=1 << 14)
def number_readings(core: str) -> Tuple[Tuple[str, ...], ...]:
    """숫자 토큰 core('2014', '5%', '21st', '1,200', '$3') → 가능한 lexical 읽기들(단어 tuple). 지원 밖이면 ()."""
    out: List[Tuple[str, ...]] = []
    def add(s):
        k = _words_of(s)
        if k and k not in out: out.append(k)
    c = canon_number_en(core)
    if c: add(c)
    base, pct = (core[:-1].rstrip(), True) if core.endswith("%") else (core, False)
    if _INT_TOKEN.fullmatch(base):
        n = int(base.replace(",", ""))
        for kw in ({}, {"lang": "en_GB"}, {"to": "year"}):
            try: add(num2words(n, **kw) + (" percent" if pct else ""))
            except Exception: pass
    m = _ORD_TOKEN.fullmatch(core)
    if m:
        try: add(num2words(int(m.group(1)), to="ordinal"))
        except Exception: pass
    return tuple(out)


def en_units(pnc: str, corpus: str) -> List[dict]:
    """pnc 텍스트 → 단위 [{surface, opts(lexical 읽기 후보), digit}]. 순수 구두점 토큰 T:
      첫 단위 앞          앞머리(lead + T + ' ' — 빌더도 스트림 첫 구두점 단어를 다음 단어에 합친다)
      여는 부호(단위 뒤)   다음 단위 앞에 공백 없이('(' + 'we' → '(we'); 뒤에 단위가 없거나 다른 구두점이 이어지면 앞 단위로 돌린다
      그 밖               앞 단위 surface 에 ' T'"""
    units: List[dict] = []; lead = ""; pend = ""; quotes = 0
    for T in pnc.split():
        key = _key(T, corpus)
        if not key:
            opening = all(c in _OPEN for c in T) or (set(T) == {'"'} and quotes % 2 == 0)
            quotes += T.count('"')
            if not units: lead += T + " "
            elif opening and not pend: pend = T
            else:
                if pend: units[-1]["surface"] += " " + pend; pend = ""
                units[-1]["surface"] += " " + T
            continue
        quotes += T.count('"')
        digit = bool(_DIGIT.search(T)); opts = [] if _DIGIT.search(" ".join(key)) else [key]
        if digit:
            for x in number_readings(_PEEL.match(T).group("core")):
                if x not in opts: opts.append(x)
        units.append(dict(surface=lead + pend + T, opts=opts, digit=digit)); lead = pend = ""
    if pend: units[-1]["surface"] += " " + pend
    return units


def align_units(units: List[dict], lex: List[str]) -> Optional[List[Tuple[int, int]]]:
    """단위마다 읽기 하나를 골라 이어 붙인 것 == lex 가 되게 하는 배정 → 단위별 lexical span [(i0, i1)], 없으면 None.
    DP(위치 집합), 동률이면 앞 후보·앞 경로(결정적)."""
    L = len(lex); states: Dict[int, Optional[tuple]] = {0: None}; back = []
    for u in units:
        nxt: Dict[int, tuple] = {}
        for i in states:
            for opt in u["opts"]:
                j = i + len(opt)
                if j <= L and j not in nxt and tuple(lex[i:j]) == opt: nxt[j] = (i,)
        if not nxt: return None
        back.append(nxt); states = nxt
    if L not in states: return None
    spans, j = [], L
    for nxt in reversed(back): i = nxt[j][0]; spans.append((i, j)); j = i
    return spans[::-1]


def _encode_many(tok, texts: List[str]) -> List[List[int]]:
    try:
        res = tok(texts, add_special_tokens=False)["input_ids"]
        if isinstance(res, list) and len(res) == len(texts) and all(isinstance(x, list) for x in res): return [list(x) for x in res]
    except Exception:
        pass
    return [list(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]


def _written_en(r: dict, tok, digit_hold: bool):
    pnc = r.get("pnc_text")
    if not pnc: return None, "en_no_pnc"
    toks, words = r["tokens"], r["words"]
    if not words or int(words[0]["a"]) != 0 or int(words[-1]["b"]) != len(toks) or any(int(a["b"]) != int(b["a"]) for a, b in zip(words, words[1:])):
        return None, "en_word_cover"                                              # 토큰 전체를 단어가 빈틈없이 덮어야 단위로 갈아 끼울 수 있다
    pnc = _GLUE.sub(r"\1 ", pnc)
    lex = [w["text"] for w in words]
    units = en_units(pnc, en_corpus(r.get("set")))
    spans = align_units(units, lex)
    if spans is None: return None, "en_align"
    if sum(1 for w in lex if re.search(r"[a-z]", w)) >= MIN_CASE_WORDS and not re.search(r"[A-Z]", pnc): return None, "en_no_case"
    b0 = _row_bytes(tok, [int(toks[0][0])])
    if b0 is None: return None, "written_decode"
    first_lead = b0[0][:1] == b" "
    enc = _encode_many(tok, [(" " if (k or first_lead) else "") + u["surface"] for k, u in enumerate(units)])
    new_toks: List[list] = []; new_words: List[dict] = []; word_map: List[Optional[int]] = [None] * len(lex)
    for k, (u, (i0, i1), ids) in enumerate(zip(units, spans, enc)):
        if not ids: return None, "en_selfcheck"
        w0, w1 = words[i0], words[i1 - 1]; src = [float(t[1]) for t in toks[int(w0["a"]):int(w1["b"])]]
        times = [src[-1]] * len(ids) if (digit_hold and u["digit"]) else _spread(src, len(ids))
        a = len(new_toks); new_toks += [[int(i), t] for i, t in zip(ids, times)]
        m = EDGE_PUNCT.match(u["surface"])
        new_words.append(dict(i=k, text=m.group(2), a=a, b=len(new_toks), end_time=times[-1], seg=w1.get("seg", 0), tags=punct_tags(m.group(3))))
        word_map[i1 - 1] = k
    if any(t1[1] < t0[1] for t0, t1 in zip(new_toks, new_toks[1:])): return None, "en_selfcheck"
    ws = split_words([t[0] for t in new_toks], [t[1] for t in new_toks], tok)                # 이 tokenizer 로 다시 나눠도 같은 단어(빌더 규칙 = speechlm lexical)
    lw = lexical_words([dict(w) for w in ws])[0] if ws else None
    if lw is None or [(w["text"], w["a"], w["b"]) for w in lw] != [(w["text"], w["a"], w["b"]) for w in new_words]: return None, "en_selfcheck"
    return dict(r, tokens=new_toks, words=new_words, text=" ".join(u["surface"] for u in units), output_format="written"), word_map


def written_row(r: dict, tok, digit_hold: bool = DIGIT_HOLD):
    """정규형 words 행(KO 는 normalize_ko_row **전** 원 행) → (표기형 행, word_map) 또는 (None, 사유)."""
    lang = r.get("lang")
    if lang == "Korean": return _written_ko(r, tok, digit_hold)
    if lang == "English": return _written_en(r, tok, digit_hold)
    return None, "written_lang"


def remap_labels(lab: Optional[dict], word_map: Optional[List[Optional[int]]]):
    """labels 행의 after_word 를 표기형 단어 index 로 → (새 labels 행, 버린 후보 수 Counter{grade}). word_map None = 그대로.
    범위 밖 after_word 는 ValueError('after_word_range: …' — build_semcommit_tokens 와 같은 사유 코드)."""
    dropped: Counter = Counter()
    if lab is None or word_map is None: return lab, dropped
    cands = []
    for c in lab.get("candidates", []):
        i = int(c["after_word"])
        if not 0 <= i < len(word_map): raise ValueError(f"after_word_range: {i} of {len(word_map)} words")
        j = word_map[i]
        if j is None: dropped[c["grade"]] += 1; continue
        cands.append(dict(c, after_word=j))
    return dict(lab, candidates=cands), dropped


_RULE_PROBES = (                                    # 규칙 지문용 고정 탐침(코퍼스별 숫자 처리·여는/닫는 부호·붙임 오류·앞머리 구두점)
    ("In 2014, we adopted it. Waxman-Markey is law.", "voxpopuli"),
    ("— So ( we won ) \"yes\" , the 3rd of 1,200 votes: 21% in the 1990s!", "yodas"),
    ("It is a layer.Many people agree ... “Really?” he said [twice] $3 G20 1.8%.", "mnsc"),
    ("Mr. Smith's 2nd car, 45 miles ; 'ok' — fine?", "librispeech"),
)


@functools.lru_cache(maxsize=1)
def written_rules_sha256() -> str:
    """표기형 규칙 지문: 빌더 이름·규칙 상수·정규식 + 탐침 문장의 en_units / 숫자 읽기 결과. 규칙·textnorm.target_en·num2words 판본이 바뀌면
    값이 바뀐다(tokenizer 차이는 지문의 vocab_sha256 이 잡는다). KO 규칙(display 대조 + digit hold)은 builder·digit_hold 로만 드러난다."""
    probe = [[(u["surface"], [list(o) for o in u["opts"]], u["digit"]) for u in en_units(t, c)] for t, c in _RULE_PROBES]
    nums = {x: [list(o) for o in number_readings(x)] for x in ("2014", "1,200", "21st", "5%", "$3", "1990s", "1.8%", "G20", "100")}
    rec = dict(builder=WRITTEN_BUILDER, min_case_words=MIN_CASE_WORDS, glue=_GLUE.pattern, int_token=_INT_TOKEN.pattern, ord_token=_ORD_TOKEN.pattern,
               open=sorted(_OPEN), edge=EDGE_PUNCT.pattern, final=sorted(FINAL), comma=sorted(COMMA), probe=probe, nums=nums)
    return hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def output_format_settings(tok, written_frac: Dict[str, float], digit_hold: bool = DIGIT_HOLD) -> dict:
    """semcommit_fingerprint 에 더할 설정. 모든 비율이 0 이면 {} — 지문 키가 기존 run 과 같아야 재개된다(fingerprint_diff 는 모든 키를 비교).
    rules_sha256 = written_rules_sha256() — WRITTEN_BUILDER 를 올리지 않은 규칙 변경도 재개 때 드러난다."""
    fr = {k: float(v) for k, v in sorted(written_frac.items())}
    if not any(v > 0 for v in fr.values()): return {}
    return {"output_format": dict(protocol=OUTFMT_PROTOCOL, instruction=WRITTEN_INSTRUCTION, header_sha256=header_sha256(tok), written_frac=fr,
                                  builder=WRITTEN_BUILDER, digit_hold=bool(digit_hold), rules_sha256=written_rules_sha256())}
