"""engine/tokens.py + engine/checkpoint.py 테스트.

prefix 표는 decode-contract §b.2 를 글자 그대로 옮겨 적었다 (코드의 expected_head 와 독립).
실제 토크나이저: ~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/tokenizer.json (없으면 건너뜀). 가중치는 읽지 않는다."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import checkpoint as ckm  # noqa: E402
from engine.checkpoint import SF_DATALESS, ModelNotFound, find_model, load_config  # noqa: E402
from engine.tokens import DEFAULTS, DELAYS, LANGS, PAIRS, SpecialIds, TokenTable, check_mode  # noqa: E402

CK = os.path.expanduser("~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final")


def _ok(p):
    try:
        return not (os.stat(p).st_flags & SF_DATALESS)
    except FileNotFoundError:
        return False


HAVE_CK = all(_ok(os.path.join(CK, n)) for n in ("", "config.json", "tokenizer.json"))
need_ck = pytest.mark.skipif(not HAVE_CK, reason="s4 체크포인트 토크나이저 없음")

HDR = [151644, 8948, 198, 151645, 198, 151644, 77091, 198]
TABLE_B2 = {   # decode-contract §b.2 (언어 줄 id)
    ("English", None): [11528, 220, 151707, 22574, 151704],
    ("Korean", None): [11528, 220, 151707, 42, 45195, 151704],
    ("German", None): [11528, 220, 151707, 32079, 151704],
    ("English", "German"): [11528, 220, 151707, 22574, 198, 14045, 220, 151707, 32079, 151704],
    ("English", "Korean"): [11528, 220, 151707, 22574, 198, 14045, 220, 151707, 42, 45195, 151704],
    ("German", "English"): [11528, 220, 151707, 32079, 198, 14045, 220, 151707, 22574, 151704],
    ("Korean", "English"): [11528, 220, 151707, 42, 45195, 198, 14045, 220, 151707, 22574, 151704],
}
DELAY_IDS = {4: 151712, 5: 151713, 6: 151714, 7: 151715, 8: 151716, 9: 151724, 10: 151725}
OFFLINE_BLOCKED = [151644, 151669, 151670, 151676, 151704, 151705, 151706] + list(range(151709, 151726))


@pytest.fixture(scope="module")
def tt():
    from tokenizers import Tokenizer
    with open(os.path.join(CK, "config.json")) as f:
        cfg = json.load(f)
    return TokenTable(Tokenizer.from_file(os.path.join(CK, "tokenizer.json")), cfg, 151936)


def test_constants():
    assert DELAYS == [4, 5, 6, 7, 8, 9, 10]
    assert set(LANGS) == {"English", "Korean", "German"} and LANGS["Korean"] == "한국어"
    assert set(PAIRS) == {("English", "German"), ("English", "Korean"), ("German", "English"), ("Korean", "English")}
    assert DEFAULTS["asr_delay"] == 6
    assert DEFAULTS["translate_delay"] == {"English>German": 10, "English>Korean": 10, "German>English": 10, "Korean>English": 8}


def test_check_mode_rejects():
    for bad in [("German", "Korean", 6), ("Korean", "German", 6), ("French", None, 6), ("English", None, 3), ("English", None, 11)]:
        with pytest.raises(ValueError):
            check_mode(*bad)
    check_mode("Korean", "English", 8)


@need_ck
def test_prefix_all_modes_and_delays(tt):
    assert len(TABLE_B2) == 7
    for (src, tgt), head in TABLE_B2.items():
        for d in DELAYS:
            assert tt.prefix_ids(src, tgt, d) == HDR + head + [DELAY_IDS[d]], (src, tgt, d)
    # §b.3 예: en→de δ6 = 19 id, 길이 14–20
    assert tt.prefix_ids("English", "German", 6) == [151644, 8948, 198, 151645, 198, 151644, 77091, 198, 11528, 220, 151707,
                                                     22574, 198, 14045, 220, 151707, 32079, 151704, 151714]
    assert sorted({len(tt.prefix_ids(s, t, 6)) for s, t in TABLE_B2}) == [14, 15, 19, 20]
    with pytest.raises(ValueError):
        tt.prefix_ids("German", "Korean", 6)


@need_ck
def test_special_ids(tt):
    sp = tt.sp
    assert (sp.next, sp.empty, sp.spk_a, sp.spk_b, sp.trans, sp.sem_end) == (151705, 151706, 151707, 151708, 151726, 151723)
    assert (sp.im_start, sp.im_end, sp.audio_start, sp.audio_end, sp.audio_pad, sp.asr_text) == (151644, 151645, 151669, 151670, 151676, 151704)
    assert sp.delay[9] == 151724 and sp.delay[10] == 151725 and sp.delay[1] == 151709


@need_ck
def test_offline_header_and_blocked(tt):
    ids, a0 = tt.offline_header("Korean", "English", 3)
    assert a0 == 9
    assert ids == [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669, 151676, 151676, 151676, 151670,
                   151645, 198, 151644, 77091, 198] + TABLE_B2[("Korean", "English")]
    ids, _ = tt.offline_header("English", None, 0)
    assert 151712 not in ids and ids[-1] == 151704
    assert tt.offline_blocked == OFFLINE_BLOCKED
    assert 151645 not in tt.offline_blocked and not {151707, 151708, 151726} & set(tt.offline_blocked)


@need_ck
def test_token_bytes_edges(tt):
    tok = tt.tok
    for i in list(range(151643, 151727)) + [151727, 151800, 151935, -1, 10 ** 6]:
        assert tt.token_bytes(i) == b"", i                       # 추가 토큰(<asr_text> 같은 비특수 포함)·표 밖 id
    assert tok.decode([151704], skip_special_tokens=True) == "<asr_text>"   # 그래서 special 플래그에 기대면 안 된다
    assert tt.token_bytes(220) == b" " and tt.token_bytes(198) == b"\n"
    for text in ("뷁 똠방각하", "Grüße 쀍", "안녕하세요 반갑습니다"):
        ids = tok.encode(text, add_special_tokens=False).ids
        parts = [tt.token_bytes(i) for i in ids]
        assert b"".join(parts).decode() == text
        if text != "안녕하세요 반갑습니다":                           # 글자 중간에서 쪼개지는 토큰 (표시 쪽 holdback 필요)
            assert any(not _utf8_complete(p) for p in parts), text
    import random
    r = random.Random(1)
    for _ in range(300):
        seq = [r.randrange(0, 151643) for _ in range(r.randrange(1, 10))]
        assert b"".join(tt.token_bytes(i) for i in seq).decode("utf-8", "replace") == tok.decode(seq)
    assert tt.decode([151707, 220, 1958, 151726]) == tok.decode([151707, 220, 1958, 151726], skip_special_tokens=True).strip()


def _utf8_complete(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


@need_ck
def test_load_config_real():
    cfg = load_config(CK)
    assert cfg["runaway_cap"] == 32 and cfg["max_flush_rounds"] == 8


@need_ck
def test_special_ids_mismatch_raises():
    from tokenizers import Tokenizer
    with open(os.path.join(CK, "config.json")) as f:
        cfg = json.load(f)
    cfg["sp_ids"] = dict(cfg["sp_ids"], **{"<TRANS>": 151727})
    with pytest.raises(RuntimeError):
        SpecialIds.from_config(cfg, Tokenizer.from_file(os.path.join(CK, "tokenizer.json")))


def _fake_ck(root, sub="final", files=("config.json", "tokenizer.json", "model.safetensors")):
    d = os.path.join(root, "run")
    os.makedirs(os.path.join(d, sub) if sub else d)
    for n in files:
        open(os.path.join(d, sub, n) if sub else os.path.join(d, n), "w").close()
    return d


def test_find_model_final_subdir(tmp_path, monkeypatch):
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    d = _fake_ck(str(tmp_path))
    assert find_model(d) == os.path.join(d, "final")
    monkeypatch.setenv("VAPKT_APP_MODEL", d)
    assert find_model() == os.path.join(d, "final")
    d2 = _fake_ck(str(tmp_path / "b"), sub="")
    assert find_model(d2) == d2


def test_find_model_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    with pytest.raises(ModelNotFound):
        find_model(str(tmp_path / "nope"))
    d = _fake_ck(str(tmp_path), files=("config.json", "model.safetensors"))
    with pytest.raises(ModelNotFound, match="tokenizer.json"):
        find_model(d)
    d = _fake_ck(str(tmp_path / "c"), files=("config.json", "tokenizer.json"))
    with pytest.raises(ModelNotFound, match="가중치"):
        find_model(d)


def test_find_model_dataless(tmp_path, monkeypatch):
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    d = _fake_ck(str(tmp_path))
    bad = os.path.join(d, "final", "model.safetensors")
    real_stat = os.stat

    class St:
        def __init__(self, st, flags):
            self.st, self.st_flags = st, flags

    def fake_stat(p, *a, **k):
        st = real_stat(p, *a, **k)
        return St(st, SF_DATALESS) if os.path.abspath(p) == bad else St(st, 0)

    monkeypatch.setattr(ckm.os, "stat", fake_stat)
    with pytest.raises(ModelNotFound, match="dataless"):
        find_model(d)


def test_find_model_candidates_skip_dataless(tmp_path, monkeypatch):
    """기본 후보 중 dataless 인 것은 건너뛰고, 아무것도 없으면 dataless 오류를 알린다."""
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    a = _fake_ck(str(tmp_path / "a"))
    b = _fake_ck(str(tmp_path / "b"))
    real_stat = os.stat

    class St:
        def __init__(self, flags):
            self.st_flags = flags

    def fake_stat(p, *x, **k):
        real_stat(p, *x, **k)
        return St(SF_DATALESS if os.path.abspath(p).startswith(a) else 0)

    monkeypatch.setattr(ckm.os, "stat", fake_stat)
    monkeypatch.setattr(ckm, "CANDIDATES", (a, b))
    assert find_model() == os.path.join(b, "final")
    monkeypatch.setattr(ckm, "CANDIDATES", (a, str(tmp_path / "none")))
    with pytest.raises(ModelNotFound, match="dataless"):
        find_model()


# ───────────────────────── 체크포인트 오류 경로 (F18): 날것의 파이썬 예외 대신 경로를 담은 한국어 ModelNotFound
def test_find_model_given_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    d = _fake_ck(str(tmp_path))
    f = os.path.join(d, "final", "model.safetensors")
    with pytest.raises(ModelNotFound, match="폴더가 아니라 파일"):
        find_model(f)
    with pytest.raises(ModelNotFound, match="찾을 수 없습니다"):              # 경로 중간이 파일 (NotADirectoryError)
        find_model(os.path.join(f, "final"))


def test_find_model_permission_error(tmp_path, monkeypatch):
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    a = _fake_ck(str(tmp_path / "a"))
    b = _fake_ck(str(tmp_path / "b"))
    real_stat = os.stat

    def fake_stat(p, *x, **k):
        if os.path.abspath(p).startswith(a):
            raise PermissionError(1, "Operation not permitted", p)
        return real_stat(p, *x, **k)

    monkeypatch.setattr(ckm.os, "stat", fake_stat)
    with pytest.raises(ModelNotFound, match="접근할 수 없습니다"):
        find_model(a)
    monkeypatch.setattr(ckm, "CANDIDATES", (a, b))
    assert find_model() == os.path.join(b, "final")                             # 기본 후보는 다음으로 넘어간다


def test_find_model_dataless_ancestor(tmp_path, monkeypatch):
    """상위 폴더가 dataless 면 그 안을 찾지 않고 그 폴더를 알린다."""
    monkeypatch.delenv("VAPKT_APP_MODEL", raising=False)
    d = _fake_ck(str(tmp_path / "x"))
    anc = os.path.dirname(d)
    real_stat = os.stat
    seen = []

    class St:
        def __init__(self, st, flags):
            self.st_mode, self.st_flags = st.st_mode, flags

    def fake_stat(p, *x, **k):
        p = os.path.abspath(p)
        seen.append(p)
        return St(real_stat(p, *x, **k), SF_DATALESS if p == anc else 0)

    monkeypatch.setattr(ckm.os, "stat", fake_stat)
    with pytest.raises(ModelNotFound, match="dataless") as e:
        find_model(d)
    assert anc in str(e.value) and not any(s.startswith(d) for s in seen)


def test_load_config_broken_or_foreign(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{")
    with pytest.raises(ModelNotFound, match="config.json 을 읽을 수 없습니다") as e:
        load_config(str(tmp_path))
    assert str(p) in str(e.value)
    p.write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(ModelNotFound, match="없는 키: thinker, thinker.text_config, sp_ids"):
        load_config(str(tmp_path))
    p.write_text(json.dumps({"thinker": {"text_config": {}}}))
    with pytest.raises(ModelNotFound, match="없는 키: sp_ids"):
        load_config(str(tmp_path))


def test_split_weights_reports_counts():
    from engine.checkpoint import split_weights
    with pytest.raises(RuntimeError, match=r"q06-s4 가중치가 아닙니다 \(/x/y\).*전체 1, thinker 1, adapter 0"):
        split_weights({"thinker.a": 0}, "/x/y")
