"""text_stream.TextAssembler 테스트 (audio-ui-spec §4.4, E6/E7). 실제 s4 tokenizer.json 으로 바이트 분할 id 를 만든다."""
import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_stream import TextAssembler, decode_replace, incomplete_tail_len  # noqa: E402

TOK = os.path.expanduser("~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/tokenizer.json")
SPK_A, SPK_B, TRANS, NEXT, SEM_END, ASR_TEXT = 151707, 151708, 151726, 151705, 151723, 151704
norm = lambda s: re.sub(r"\s+", " ", s).strip()


def _tok_available():
    try:
        return os.path.exists(TOK) and not (os.stat(TOK).st_flags & 0x40000000)   # iCloud dataless 면 읽지 않는다(멈춤)
    except OSError:
        return False


@pytest.fixture(scope="module")
def tok():
    if not _tok_available(): pytest.skip("s4 tokenizer.json 없음 또는 dataless")
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(TOK)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}
    added = set(t.get_added_tokens_decoder())

    def token_bytes(i):                                   # 엔진의 token_bytes 와 같은 규칙: 추가·특수·표 밖 id → b""
        if i in added: return b""
        p = t.id_to_token(i)
        return b"" if p is None else bytes(u2b[c] for c in p)
    t.token_bytes = token_bytes
    t.ids = lambda s: t.encode(s, add_special_tokens=False).ids
    return t


def assemble(tb, rounds, keep_tr=True):
    ta = TextAssembler(tb, (SPK_A, SPK_B), TRANS, keep_tr=keep_tr)
    ds, dt = [], []
    for r in rounds:
        a, b = ta.push(r); ds.append(a); dt.append(b)
    a, b = ta.flush(); ds.append(a); dt.append(b)
    src, tr, st = ta.finish()
    assert src == "".join(ds) and tr == "".join(dt)                  # 증분의 연결 == 최종 텍스트
    assert st["final_src"] == "" and st["final_tr"] == ""            # flush() 를 먼저 불렀으므로
    return src, tr, st, ds, dt


def split_runs(ids):
    src, tr, run = [], [], 0
    for t in ids:
        if t in (SPK_A, SPK_B): run = 0; continue
        if t == TRANS: run = 1; continue
        (tr if run else src).append(t)
    return src, tr


# ---------------------------------------------------------------- 도우미 함수
def test_incomplete_tail_and_decode_replace():
    assert incomplete_tail_len(b"abc") == 0
    assert incomplete_tail_len("가".encode()[:2]) == 2 and incomplete_tail_len("가".encode()[:1]) == 1
    assert incomplete_tail_len("가".encode()) == 0 and incomplete_tail_len("😀".encode()[:3]) == 3
    assert incomplete_tail_len(b"a\x80") == 0 and incomplete_tail_len(b"\xff") == 0
    for b, want in ((b"ab\x80c", 1), (b"\xed\xa0\x80", 3), (b"\xf0\x9f\x98", 3), (b"\xe3\x81\xe3\x81\x82", 2), ("정상".encode(), 0)):
        s, n = decode_replace(b)
        assert s == b.decode("utf-8", "replace") and n == want


# ---------------------------------------------------------------- 실제 토크나이저
def test_hangul_byte_split_holdback(tok):
    ids = tok.ids("오늘은 날씨가 좋네요")
    assert 37195 in ids and 254 in ids                               # ' \xeb\x82' + '\xa0' = ' 날'
    src, _, st, ds, _ = assemble(tok.token_bytes, [[SPK_A]] + [[i] for i in ids])
    assert all("\ufffd" not in d for d in ds) and src == "오늘은 날씨가 좋네요" and st["replaced_bytes"] == 0
    j = 1 + ids.index(37195)
    assert ds[j] == "" and ds[j + 1] == " 날"                        # 반쪽 바이트와 끝 공백을 보류했다가 함께 낸다


def test_tag_routing_critic_example(tok):
    """decode-contract §e: 태그는 구간이 바뀔 때만, 상태는 프레임을 넘어 유지."""
    I = tok.ids
    rounds = [[SPK_A] + I(" it") + [TRANS] + I(" Es"), [SPK_A] + I(" is"), I(" about") + [TRANS] + I(" geht um"),
              [SPK_A] + I(" shared"), I(" values") + [TRANS] + I(" gemeinsame Werte")]
    src, tr, st, ds, dt = assemble(tok.token_bytes, rounds)
    assert src == "it is about shared values" and tr == "Es geht um gemeinsame Werte"
    assert ds[0] == "it" and dt[0] == "Es" and ds[2] == " about" and dt[2] == " geht um"


def test_tokens_before_any_tag_are_src_and_specials_render_nothing(tok):
    ids = tok.ids("hello") + [SEM_END, 151727, 151800, 151935, ASR_TEXT, NEXT] + tok.ids(" world")
    src, tr, st, *_ = assemble(tok.token_bytes, [ids])
    assert src == "hello world" and tr == ""                         # E7: 151727–151935·<SEM_END>·<asr_text> 는 글자가 없다
    assert st["n_src"] == len(ids)


def test_asr_mode_drops_trans_run(tok):
    I = tok.ids
    rounds = [[SPK_A] + I(" guten tag"), [TRANS] + I(" Good day") + [SPK_A], I(" wie geht"), [TRANS] + I(" how")]
    src, tr, st, ds, dt = assemble(tok.token_bytes, rounds, keep_tr=False)
    assert src == "guten tag wie geht" and tr == "" and all(d == "" for d in dt)
    assert st["tr_dropped"] == len(I(" Good day")) + len(I(" how")) and st["n_tr"] == 0


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_equals_reference_decode(tok, seed):
    """무작위 태그 삽입·무작위 라운드 분할: 각 스트림 == 정규화된 tok.decode(구간 id), U+FFFD 는 참조에 있을 때만."""
    rnd = random.Random(seed)
    texts = ["안녕하세요 반갑습니다", "오늘은  날씨가 좋네요\n", " Grüße aus München", "hello   world ", "日本語のテキスト 😀",
             "\t탭과\n줄바꿈 ", "숫자 일 이 삼", " Straße über Äpfel"]
    ids = []
    for _ in range(rnd.randint(3, 12)):
        if rnd.random() < 0.6: ids.append(rnd.choice([SPK_A, SPK_B, TRANS]))
        ids += tok.ids(rnd.choice(texts))
        if rnd.random() < 0.2: ids.append(rnd.choice([SEM_END, 151800, NEXT]))
    rounds, i = [], 0
    while i < len(ids):
        n = rnd.randint(0, 6); rounds.append(ids[i:i + n]); i += n
    src, tr, st, ds, dt = assemble(tok.token_bytes, rounds)
    s_ids, t_ids = split_runs(ids)
    assert src == norm(tok.decode(s_ids)) and tr == norm(tok.decode(t_ids))
    assert "\ufffd" not in src + tr and st["replaced_bytes"] == 0
    for d in ds + dt: assert "\ufffd" not in d


# ---------------------------------------------------------------- 가짜 바이트 표 (토크나이저 없이)
def _byte_table(chunks):
    tbl = {1000 + i: c for i, c in enumerate(chunks)}
    return tbl, (lambda i: tbl.get(i, b""))


@pytest.mark.parametrize("seed", range(30))
def test_random_byte_splits_never_show_broken_chars(seed):
    rnd = random.Random(seed)
    text = rnd.choice(["  안녕하세요   반갑습니다 오늘은 날씨가 좋네요  ", "Grüße, Straße! ", "  😀 emoji 👍🏽 ", "日本語 と 한국어\n\n섞기"])
    b = text.encode()
    cuts = sorted(rnd.sample(range(1, len(b)), min(len(b) - 1, rnd.randint(1, len(b) - 1))))
    chunks = [b[i:j] for i, j in zip([0] + cuts, cuts + [len(b)])]
    tbl, tb = _byte_table(chunks)
    src, tr, st, ds, _ = assemble(tb, [[i] for i in tbl])
    assert src == norm(text) and all("\ufffd" not in d for d in ds) and st["replaced_bytes"] == 0


def test_whitespace_rules():
    tbl, tb = _byte_table([b"  hello", b"  ", b" \n world ", b"  ", b"!", b" "])
    src, _, _, ds, _ = assemble(tb, [[1000], [1001], [1002], [1003], [1004], [1005]])
    assert ds[:6] == ["hello", "", " world", "", " !", ""] and src == "hello world !"


def test_incomplete_bytes_at_finish_are_replaced_and_counted():
    tbl, tb = _byte_table([b"abc", "가".encode()[:2]])
    ta = TextAssembler(tb, (SPK_A, SPK_B), TRANS)
    assert ta.push([1000, 1001]) == ("abc", "")
    src, tr, st = ta.finish()                                        # flush() 없이 finish: 마지막 증분은 stats 에
    assert src == "abc\ufffd" and st["final_src"] == "\ufffd" and st["replaced_bytes"] == 2


def test_invalid_bytes_mid_stream_are_replaced_like_tok_decode():
    raw = [b"ok ", b"\x80", b"\xe3\x81", b" fine", b"\xf0\x9f", b"\x98\x80"]
    tbl, tb = _byte_table(raw)
    src, _, st, ds, _ = assemble(tb, [[i] for i in tbl])
    assert src == norm(b"".join(raw).decode("utf-8", "replace")) and st["replaced_bytes"] == 3


def test_char_split_across_trans_run():
    """원문 한 글자의 바이트가 번역 구간을 사이에 두고 나뉘어도 원문 버퍼에서 이어 붙는다(구간 id 연결 디코드와 같다)."""
    an = "안".encode()
    tbl, tb = _byte_table([b" x" + an[:2], b" Hi", an[2:] + b" y"])
    src, tr, st, ds, dt = assemble(tb, [[SPK_A, 1000], [TRANS, 1001], [SPK_A, 1002]])
    assert src == "x안 y" and tr == "Hi" and ds == ["x", "", "안 y", ""] and st["replaced_bytes"] == 0
