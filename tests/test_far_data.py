"""데이터 확장 D-1(2026-10-26): 원거리 버전 mono(dialogue_mix.far_mono — 장치 녹음을 offset_s 만큼 당겨 대화 시간축에), 화자별 신호(mix_stems — 합이 mono),
원거리 빌더의 포락선 상호상관 지연(p2_build_far.xcorr_lag). CPU."""
import importlib.util, pathlib

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
from vapasr.data.dialogue import Dialogue, ChannelRef, Utterance  # noqa: E402
from vapasr.data.dialogue_mix import far_mono, mix_dialogue, mix_stems, _rms_over  # noqa: E402

SR = 16000


def _dlg(tmp_path, chans, dur=4.0, meta=None):
    utts = [Utterance(speaker=s, start=0.5 + i, end=1.4 + i, raw="x", utt_id=f"u{i}") for i, s in enumerate(chans)]
    refs = {}
    for i, (s, x) in enumerate(chans.items()):
        p = tmp_path / f"{s}.wav"; sf.write(p, x, SR, subtype="FLOAT"); refs[s] = ChannelRef(path=str(p))
    return Dialogue(conv_id="t:1", corpus="T", lang="English", split="train", duration_s=dur, speakers=sorted(chans), utterances=utts, channels=refs, meta=meta or {})


def test_far_mono_offset_and_level(tmp_path):
    rng = np.random.default_rng(0); x = (rng.standard_normal(5 * SR) * 0.05).astype(np.float32); x[SR: SR + 100] = 0.9      # 장치 1.0 s 에 표지
    p = tmp_path / "dev.wav"; sf.write(p, np.stack([x * 0.1, x], 1), SR, subtype="FLOAT")                                                       # 2 채널, 1 번 채널을 쓴다
    d = _dlg(tmp_path, {"A": np.zeros(4 * SR, np.float32)}, meta=dict(mono_src=dict(path=str(p), channel=1, offset_s=0.25)))
    y, m = far_mono(d); assert y.shape == (4 * SR,) and m["src"]["channel"] == 1
    assert int(np.argmax(np.abs(y))) == SR - SR // 4                                                                           # 대화 0.75 s = 장치 1.0 s
    g = m["gain"] * m["peak_scale"]; assert np.allclose(y[:1000], x[SR // 4: SR // 4 + 1000] * g, atol=1e-6)
    assert np.allclose(mix_dialogue(d)[0], y)                                                                                   # mix_dialogue 가 mono_src 를 따른다
    d2 = _dlg(tmp_path, {"A": np.zeros(4 * SR, np.float32)}, meta=dict(mono_src=dict(path=str(p), channel=1, offset_s=-0.5)))
    y2, _ = far_mono(d2); assert not np.any(y2[: SR // 2]) and np.any(y2[SR // 2: SR])                                        # 장치가 늦게 시작 → 앞은 0


def test_mix_stems_sum_is_mono(tmp_path):
    rng = np.random.default_rng(1)
    chans = {"A": (rng.standard_normal(4 * SR) * 0.02).astype(np.float32), "B": (rng.standard_normal(4 * SR) * 0.3).astype(np.float32)}
    d = _dlg(tmp_path, chans); st, meta = mix_stems(d); mono, _ = mix_dialogue(d)
    assert st.shape == (2, 4 * SR) and meta["speakers"] == ["A", "B"] and np.allclose(st.sum(0), mono, atol=1e-6)
    for i, s in enumerate(meta["speakers"]):                                                                                    # 화자마다 자기 발화 구간 RMS 가 -23 dBFS(피크 배율 전)
        spans = [(u.start, u.end) for u in d.utterances if u.speaker == s]
        assert abs(20 * np.log10(_rms_over(st[i] / meta["peak_scale"], spans, SR)) + 23.0) < 0.05
    with pytest.raises(ValueError): mix_stems(_dlg(tmp_path, {"A": chans["A"]}, meta=dict(mono_src=dict(path="x"))))


def test_xcorr_lag_finds_shift():
    spec = importlib.util.spec_from_file_location("_far", pathlib.Path(__file__).parents[1] / "experiments" / "p2_build_far.py"); far = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(far)
    rng = np.random.default_rng(2); burst = np.zeros(70 * SR, np.float32)
    for t in rng.uniform(1, 68, 60): burst[int(t * SR): int((t + rng.uniform(0.2, 1.0)) * SR)] = rng.standard_normal(1) * 0.5 + 0.6
    sig = burst * rng.standard_normal(len(burst)).astype(np.float32)
    ml = 200; ref = far.env(sig[2 * SR: 62 * SR])
    for shift in (0, 37, -55):                                                                                                  # 장치가 shift 프레임 늦다
        dev = far.env(np.roll(sig, shift * far.HOP)[2 * SR - ml * far.HOP: 62 * SR + ml * far.HOP])
        L, c = far.xcorr_lag(ref, dev, ml); assert L == shift and c > 0.8, (shift, L, c)
