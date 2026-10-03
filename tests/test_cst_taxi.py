"""TAXI 파서(vapasr/cst/taxi.py) — 작은 가짜 세션으로 Partitur 파싱·표기 정리·usable 판정·스키마 왕복."""
import struct
import wave

from vapasr.cst.schema import Session
from vapasr.cst.taxi import clean_ort, iter_sessions, parse_par, stats

PAR_DP = """LHD: Partitur 1.3
SAM: 8000
SPN: DP
LBD:
ORT:\t0\t[fil]
ORT:\t1\tes
ORT:\t2\tist
ORT:\t3\t*leider
ORT:\t4\tmöglich
ORT:\t5\t**
ORT:\t6\thal~
KAN:\t0\tQE:
NOI:\t-1;0\t<Noise>
TLN:\t0,1,2,3,4,5,6\tDE>EN\tit is unfortunately possible
"""
PAR_CL = """LHD: Partitur 1.3
SPN: CL
ORT:\t0\thello
ORT:\t1\tthere
TLN:\t0,1\tEN>DE\thallo da
"""


def _wav(path, n=800, sr=8000):
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(struct.pack(f"<{n}h", *([0] * n)))


def _session(root):
    d = root / "SES0001"; d.mkdir()
    for name in ("0001PD00", "0001DP01", "0001CL02", "0001DP03", "0001HP04"):
        _wav(d / f"{name}.wav")
    (d / "0001DP01.par").write_text(PAR_DP, encoding="utf-8")
    (d / "0001CL02.par").write_text(PAR_CL, encoding="utf-8")   # 0001DP03 은 .par 없음 = garbage
    return d


def test_parse_par_and_clean():
    p = parse_par(PAR_DP)
    assert [w for _, w in p["ort"]][:3] == ["[fil]", "es", "ist"] and p["tln"] == [("0,1,2,3,4,5,6", "DE>EN", "it is unfortunately possible")]
    assert p["noise"] == [{"span": "-1;0", "label": "<Noise>"}] and p["header"]["SPN"] == "DP"
    assert clean_ort([w for _, w in p["ort"]]) == "es ist leider möglich hal"


def test_iter_sessions_roles_usable_and_roundtrip(tmp_path):
    _session(tmp_path)
    (s,) = list(iter_sessions(str(tmp_path)))
    assert s.session_id == "taxi/SES0001" and not s.redistributable and s.timing == "turn_order_only"
    assert [t.turn_id for t in s.turns] == ["0001DP01", "0001CL02", "0001DP03"]            # PD·HP 제외, index 순
    dp, cl, bad = s.turns
    assert dp.lang == "German" and dp.translation.lang == "English" and dp.usable and dp.duration_s == 0.1
    assert cl.lang == "English" and cl.translation.text == "hallo da" and cl.transcript == "hello there"
    assert not bad.usable and bad.translation is None
    assert Session.from_json(s.to_json()) == s
    st = stats([s])
    assert st["usable_turns"] == 2 and st["usable_DP"] == 1 and st["usable_CL"] == 1 and st["sessions_without_usable"] == 0
