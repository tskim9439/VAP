"""`asr-tn-v1.0.0` golden regression (wiki/outputs/output-asr-tn-v1-spec.md 'Golden regression 최소 집합' + 추가 사례).
실행: python tests/test_textnorm.py (또는 pytest). num2words==0.5.14 필요. tokenizer-ID snapshot 은 tests/golden/asr-tn-v1.0.0.json 이 있고
MXC_QWEN_ASR_DIR(또는 QWEN_ASR_DIR) 이 잡히면 검사한다(없으면 문자열만). snapshot 생성: python tests/test_textnorm.py --write-golden"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.textnorm import (TEXTNORM_VERSION, NUMERIC_BACKEND_VERSION, NUMERIC_BACKEND_PINNED, target_en, target_ko, score_en, score_ko,
                                  canon_number_en, target_flags, fingerprint)
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", f"{TEXTNORM_VERSION}.json")

EN_TARGET = [("DON'T STOP!", "don't stop"), ("I’VE TWENTY-ONE BOOKS", "i've twenty one books"), ('"HELLO"—SHE SAID', "hello she said"),
             ("<en-US> It's fine, Mr. O'Neil’s book", "it's fine mr o'neil's book"), ("'quoted' don't", "quoted don't"), ("and/or  mm-mm", "and or mm mm")]
EN_NUM = [("0", "zero"), ("101", "one hundred one"), ("12,345", "twelve thousand three hundred forty five"), ("1.05", "one point zero five"),
          ("1.10", "one point one zero"), ("1.05%", "one point zero five percent"), ("$1", "one dollar"), ("$5", "five dollars"), ("21st", "twenty first"),
          ("1000", "one thousand"), ("999,999,999", "nine hundred ninety nine million nine hundred ninety nine thousand nine hundred ninety nine"), ("13th", "thirteenth"), ("2nd", "second")]
EN_UNSUPPORTED = ["007", "0012", "10:30", "1/2", "$5.50", "5kg", "A320", "11st", "22th", "-5", "3-5", "€5", "1e-5", "1000000000", "0.1234567", "1,00", "12,3456"]
KO_TARGET = [("(100명)/(백 명)", "백 명"), ("(0.1프로)/(영 점 일 프로)", "영 점 일 프로"), ("(TV)/(티비)", "티비"), ("(3G)/(쓰리 지)", "쓰리 지"), ("(서울)/(서울)", "서울"),
             ("b/ 안녕하세요", "안녕하세요"), ("아/ 그게 아니고", "아 그게 아니고"), ("TV를 봤어", "TV를 봤어"), ("pc방 가자, tv 봐.", "pc방 가자 tv 봐"),
             ("b/ 아 나 (3DS)/(쓰리 DS) 갖고 싶다.", "아 나 쓰리 DS 갖고 싶다"), ("(엄마)/(음마) 뭐/ 그니까+ 그* 래", "엄마 뭐 그니까 그 래"), ("o/ n/ 어. 수빈이, 이케 해서", "어 수빈이 이케 해서"),
             ("<ko-KR> 안녕하세요.", "안녕하세요")]
KO_QUARANTINE = [("100명이 왔어", {"digit"}), ("(3주)/(3 주)", {"digit"}), ("(안녕)/(", {"malformed_dual"}), ("()/(이십)", {"malformed_dual"}), ("", {"empty"}), ("b/ o/", {"empty"}), ("안녕_하세요", {"charset"})]

def test_version():
    assert TEXTNORM_VERSION == "asr-tn-v1.0.0"; assert NUMERIC_BACKEND_VERSION == NUMERIC_BACKEND_PINNED, NUMERIC_BACKEND_VERSION
    fp = fingerprint(); assert fp["textnorm_version"] == TEXTNORM_VERSION and len(fp["textnorm_sha256"]) == 64
def test_en_target():
    for raw, exp in EN_TARGET: assert target_en(raw) == exp, (raw, target_en(raw), exp)
    assert target_en("SHE PAID 5 DOLLARS IN 1984") == "she paid 5 dollars in 1984"                       # 숫자 미변환(quarantine 대상)
    assert "digit" in target_flags(target_en("SHE PAID 5 DOLLARS IN 1984"), "English")
    assert target_flags(target_en("DON'T STOP!"), "English") == set()
def test_en_numbers():
    for raw, exp in EN_NUM: assert canon_number_en(raw) == exp, (raw, canon_number_en(raw), exp)
    for raw in EN_UNSUPPORTED: assert canon_number_en(raw) is None, (raw, canon_number_en(raw))
    assert score_en("It's 101, not $5.50 at 10:30!") == "it's one hundred one not 5 50 at 10 30"       # 미지원 숫자는 부분 변환 없이 표면 규칙만
    assert score_en("Total 12,345 (5 %)") == "total twelve thousand three hundred forty five five percent"
    assert score_en("at 10:30 on 09/07/2026") == "at 10 30 on 09 07 2026"                 # 미지원: 부분 변환 없이 표면 규칙만
    assert score_en("Hello, it's 5 o'clock.") == "hello it's five o'clock"
def test_ko_target():
    for raw, exp in KO_TARGET: assert target_ko(raw, "kspon") == exp, (raw, target_ko(raw, "kspon"), exp)
    for raw, exp in KO_TARGET: assert target_flags(target_ko(raw, "kspon"), "Korean", raw, "kspon") == set(), raw
def test_ko_quarantine():
    for raw, exp in KO_QUARANTINE:
        t = target_ko(raw, "kspon"); f = target_flags(t, "Korean", raw, "kspon"); assert exp <= f, (raw, t, f, exp)
def test_ko_score():
    assert score_ko("<ko-KR> 안녕, pc 방!", spaces=True) == "안녕 pc 방"; assert score_ko("<ko-KR> 안녕, pc 방!", spaces=False) == "안녕pc방"
    assert score_ko("십만 원", True) == target_ko("(10만원)/(십만 원)")
def test_idempotent():
    for _, exp in EN_TARGET: assert target_en(exp) == exp and score_en(exp) == exp
    for _, exp in EN_NUM: assert score_en(exp) == exp
    for _, exp in KO_TARGET: assert target_ko(exp, "aihub") == exp and target_ko(exp, "kspon") == exp
def _tokenizer():
    d = os.environ.get("MXC_QWEN_ASR_DIR", os.environ.get("QWEN_ASR_DIR"))
    if not d or not os.path.isdir(d): return None
    from transformers import AutoTokenizer; return AutoTokenizer.from_pretrained(d)
def _snapshot(tok):
    ids = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    return dict(version=TEXTNORM_VERSION, en=[[r, e, ids(e)] for r, e in EN_TARGET + [(r, e) for r, e in EN_NUM]], ko=[[r, e, ids(e)] for r, e in KO_TARGET])
def test_token_ids():
    tok = _tokenizer()
    if tok is None or not os.path.exists(GOLDEN): print("  (tokenizer-ID snapshot 생략: tokenizer 또는 golden 없음)"); return
    g = json.load(open(GOLDEN, encoding="utf-8")); s = _snapshot(tok); assert g["version"] == TEXTNORM_VERSION
    for k in ("en", "ko"):
        for (r, e, i), (r2, e2, i2) in zip(g[k], s[k]): assert (r, e, i) == (r2, e2, i2), (k, r, e, i, e2, i2)

if __name__ == "__main__":
    if "--write-golden" in sys.argv:
        tok = _tokenizer(); assert tok is not None, "tokenizer 없음"; os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        json.dump(_snapshot(tok), open(GOLDEN, "w", encoding="utf-8"), ensure_ascii=False, indent=0); print("golden →", GOLDEN); sys.exit(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns: f(); print("ok", f.__name__)
    print(f"{TEXTNORM_VERSION}: {len(fns)} tests passed")
