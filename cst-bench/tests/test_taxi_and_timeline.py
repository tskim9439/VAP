"""Unit tests with a tiny synthetic TAXI-like session (no corpus data required)."""
import json
import struct
import wave
from types import SimpleNamespace as NS

import numpy as np

from cstbench.cli import main
from cstbench.corpora.taxi import clean_ort, iter_sessions, parse_par, stats
from cstbench.schema import Session
from cstbench.timeline import overlap_ratio, place_turns, render, session_seed, trim_silence

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
NATURAL = {"type": "normal", "mean": 0.2, "std": 0.25, "min": 0.05, "max": 1.0}


def _wav(path, n=8000, sr=8000, amp=0):
    x = [0] * (n // 4) + [amp] * (n // 2) + [0] * (n - n // 4 - n // 2)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(struct.pack(f"<{n}h", *x))


def _corpus(root):
    d = root / "SES0001"
    d.mkdir(parents=True)
    for name in ("0001PD00", "0001DP01", "0001CL02", "0001DP03", "0001HP04"):
        _wav(d / f"{name}.wav", amp=3000)
    (d / "0001DP01.par").write_text(PAR_DP, encoding="utf-8")
    (d / "0001CL02.par").write_text(PAR_CL, encoding="utf-8")      # 0001DP03 has no .par -> garbage
    return root


def test_parse_par_and_clean():
    p = parse_par(PAR_DP)
    assert [w for _, w in p["ort"]][:3] == ["[fil]", "es", "ist"]
    assert p["tln"] == [("0,1,2,3,4,5,6", "DE>EN", "it is unfortunately possible")]
    assert p["noise"] == [{"span": "-1;0", "label": "<Noise>"}] and p["header"]["SPN"] == "DP"
    assert clean_ort([w for _, w in p["ort"]]) == "es ist leider möglich hal"


def test_taxi_sessions_and_roundtrip(tmp_path):
    (s,) = list(iter_sessions(str(_corpus(tmp_path))))
    assert s.session_id == "taxi/SES0001" and not s.redistributable and s.timing == "turn_order_only"
    assert [t.turn_id for t in s.turns] == ["0001DP01", "0001CL02", "0001DP03"]
    dp, cl, bad = s.turns
    assert dp.lang == "German" and dp.translation.lang == "English" and dp.usable and dp.audio == "SES0001/0001DP01.wav"
    assert cl.translation.text == "hallo da" and cl.transcript == "hello there"
    assert not bad.usable and bad.translation is None
    assert Session.from_json(s.to_json()) == s
    st = stats([s])
    assert (st["usable_turns"], st["usable_DP"], st["usable_CL"]) == (2, 1, 1)


def test_place_render_overlap_and_seed():
    turns = [NS(turn_id="a", speaker="DP", duration_s=1.0), NS(turn_id="b", speaker="CL", duration_s=0.5),
             NS(turn_id="c", speaker="CL", duration_s=0.5)]
    p = place_turns(turns, NATURAL, seed=1)
    assert p[0].start_s == 0.5 and p[0].gap_before_s == 0.0
    assert all(q.start_s >= prev.end_s for prev, q in zip(p, p[1:]))
    assert 0.3 <= p[2].gap_before_s <= 0.8 and 0.05 <= p[1].gap_before_s <= 1.0
    assert place_turns(turns, NATURAL, seed=1) == p
    assert session_seed("taxi/SES0037", 3) == session_seed("taxi/SES0037", 0) + 3
    sr = 100
    audio = {t.turn_id: np.full(int(t.duration_s * sr), 0.1, np.float32) for t in turns}
    mix, st = render(p, audio, ["DP", "CL"], sr)
    assert st.shape[1] == 2 and abs(st[:, 0].sum() - 10.0) < 1e-3 and abs(mix.sum() - 20.0) < 1e-3
    assert overlap_ratio(p) == 0.0
    p[1].start_s = p[0].end_s - 0.5
    p[1].end_s = p[1].start_s + 0.5
    speech = (p[0].end_s - p[0].start_s) + (p[2].end_s - p[2].start_s)
    assert abs(overlap_ratio(p) - 0.5 / speech) < 1e-6


def test_trim_silence():
    sr = 1000
    x = np.zeros(3000, np.float32)
    x[1000:2000] = 0.5
    i, j = trim_silence(x, sr, margin_s=0.1)
    assert abs(i - 900) <= 20 and abs(j - 2100) <= 20
    assert trim_silence(np.zeros(500, np.float32), sr) == (0, 500)


def test_cli_build_and_verify(tmp_path):
    root = _corpus(tmp_path / "taxi")
    out = tmp_path / "out"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"name": "t", "sample_rate": 16000, "gap": NATURAL, "same_speaker_pause": [0.3, 0.8],
                               "lead_s": 0.5, "tail_s": 0.5, "trim": {"enabled": True, "frame_s": 0.02, "rel": 0.1, "margin_s": 0.1},
                               "seed": 0}))
    main(["build-taxi", "--root", str(root), "--out", str(out), "--config", str(cfg)])
    tl = json.loads((out / "t" / "SES0001" / "timeline.json").read_text())
    assert [t["turn_id"] for t in tl["turns"]] == ["0001DP01", "0001CL02"] and tl["skipped"] == ["0001DP03"]
    assert tl["turns"][0]["trimmed_s"][0] > 0.1                                   # leading silence removed
    info = json.loads((out / "t" / "build_info.json").read_text())
    main(["render", "--manifest", str(out / "manifest.jsonl"), "--audio-root", str(out / "audio16k"), "--config", str(cfg), "--out", str(out / "t2")])
    info2 = json.loads((out / "t2" / "build_info.json").read_text())
    assert info["checksums"] == info2["checksums"]                                # deterministic rebuild
    main(["verify", "--out", str(out / "t2"), "--reference", str(out / "t" / "build_info.json")])
