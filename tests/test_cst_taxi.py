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


def test_timeline_place_render_overlap():
    import numpy as np
    from types import SimpleNamespace as NS
    from vapasr.cst.timeline import overlap_ratio, place_turns, render
    turns = [NS(turn_id="a", speaker="DP", duration_s=1.0), NS(turn_id="b", speaker="CL", duration_s=0.5), NS(turn_id="c", speaker="CL", duration_s=0.5)]
    p = place_turns(turns, gap="natural", seed=1)
    assert p[0].start_s == 0.5 and p[0].gap_before_s == 0.0
    assert all(q.start_s >= prev.end_s for prev, q in zip(p, p[1:]))                       # 겹침 없음
    assert 0.3 <= p[2].gap_before_s <= 0.8 and 0.05 <= p[1].gap_before_s <= 1.0            # 같은 화자 쉼 · 교대 간격
    assert place_turns(turns, seed=1) == p                                                  # 재현
    sr = 100; audio = {t.turn_id: np.full(int(t.duration_s * sr), 0.1, np.float32) for t in turns}
    mono, st = render(p, audio, ["DP", "CL"], sr)
    assert st.shape[1] == 2 and abs(st[:, 0].sum() - 10.0) < 1e-3 and abs(st[:, 1].sum() - 10.0) < 1e-3 and abs(mono.sum() - 20.0) < 1e-3
    assert overlap_ratio(p) == 0.0
    p2 = place_turns(turns, seed=1); p2[1].start_s = p2[0].end_s - 0.5; p2[1].end_s = p2[1].start_s + 0.5   # 0.5 s 겹침
    speech = (p2[0].end_s - p2[0].start_s) + (p2[2].end_s - p2[2].start_s)                  # b 는 a 안에 들어감 → 발화 합집합 = a + c
    assert abs(overlap_ratio(p2) - 0.5 / speech) < 1e-6


def test_trim_silence():
    import numpy as np
    from vapasr.cst.timeline import trim_silence
    sr = 1000; x = np.zeros(3000, np.float32); x[1000:2000] = 0.5                          # 1 s 무음 + 1 s 소리 + 1 s 무음
    i, j = trim_silence(x, sr, margin_s=0.1)
    assert abs(i - 900) <= 20 and abs(j - 2100) <= 20
    assert trim_silence(np.zeros(500, np.float32), sr) == (0, 500)
