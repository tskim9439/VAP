"""데이터 확장 D-1(2026-10-26): 원거리 버전 mono(dialogue_mix.far_mono — 장치 녹음을 offset_s 만큼 당겨 대화 시간축에), 화자별 신호(mix_stems — 합이 mono),
원거리 빌더의 포락선 상호상관 지연(p2_build_far.xcorr_lag)·학습용 키(--key chime6tr), 자기 채널에서 들리지 않는 발화 표시(p2_flag_channel_level). CPU."""
import importlib.util, json, os, pathlib, subprocess, sys

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


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", pathlib.Path(__file__).parents[1] / "experiments" / f"{name}.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def test_flag_channel_level_marks_dead_channel_utterances(tmp_path):
    rng = np.random.default_rng(3); floor = lambda: (rng.standard_normal(6 * SR) * 1e-4).astype(np.float32)   # 채널 바닥 −80 dB
    a, b = floor(), floor()
    for i in (0, 2): a[int((0.5 + i) * SR): int((1.4 + i) * SR)] += (rng.standard_normal(int(0.9 * SR)) * 0.05).astype(np.float32)   # A 발화는 들린다
    b[int(1.5 * SR): int(2.4 * SR)] += (rng.standard_normal(int(0.9 * SR)) * 0.05).astype(np.float32)                                # B 는 첫 발화만, 둘째(3.5 s)는 죽은 채널
    d = _dlg(tmp_path, {"A": a, "B": b, "A2": a}, dur=6.0)
    d.utterances = [Utterance(speaker=s, start=st, end=st + 0.9, raw="x", utt_id=f"u{i}", flags=fl) for i, (s, st, fl) in
                    enumerate([("A", 0.5, None), ("B", 1.5, None), ("A", 2.5, ["empty"]), ("B", 3.5, None)])]
    st = _load("p2_flag_channel_level").flag_dialogue(d, 6.0)
    assert [u.flags for u in d.utterances] == [None, None, ["empty"], ["own_ch_inaudible"]]                     # 들리는 발화는 그대로, 원래 표시는 유지
    assert {c["speaker"]: c["flagged"] for c in st} == {"A": 0, "B": 1}                                           # 발화 없는 채널(A2)은 통계에 없다
    _load("p2_flag_channel_level").flag_dialogue(d, 6.0); assert d.utterances[3].flags == ["own_ch_inaudible"]   # 두 번 돌려도 한 번만


def test_build_far_key_reads_and_writes_training_key(tmp_path):
    """--key chime6tr: 입력 <data>/chime6tr.refined, 출력 chime6tr-far(conv_id 접두사), 장치는 --corpus chime6 규칙, far_of 는 원본 id. chime6.refined 는 만들지 않는다."""
    from vapasr.data.dialogue_dataset import mono_cache_path
    rng = np.random.default_rng(4); dur = 80.0; sig = np.zeros(int(dur * SR), np.float32)
    for t in rng.uniform(1, 78, 70): sig[int(t * SR): int((t + rng.uniform(0.2, 1.0)) * SR)] = rng.standard_normal(1) * 0.5 + 0.6
    sig *= rng.standard_normal(len(sig)).astype(np.float32)
    data, dev = tmp_path / "phase2", tmp_path / "dev"; data.mkdir(); dev.mkdir()
    d = _dlg(tmp_path, {"P01": sig}, dur=dur); d.conv_id, d.corpus = "chime6:S03", "CHiME6"
    d.utterances = [Utterance(speaker="P01", start=float(t), end=float(t) + 4.0, raw="x", utt_id=f"u{t}") for t in range(2, 76, 5)]
    (data / "chime6tr.refined.dialogues.jsonl").write_text(d.to_json() + "\n"); (data / "split.json").write_text(json.dumps({"corpora": {}}))
    mf = mono_cache_path(str(data / "mono"), d); os.makedirs(os.path.dirname(mf)); np.save(mf, sig.astype(np.float16))
    sf.write(dev / "S03_U01.CH1.wav", np.concatenate([np.zeros(int(0.3 * SR), np.float32), sig]) * 0.2, SR, subtype="FLOAT")    # 장치가 0.3 s 늦다
    r = subprocess.run([sys.executable, "experiments/p2_build_far.py", "--corpus", "chime6", "--key", "chime6tr", "--data", str(data), "--split", str(data / "split.json"),
                        "--chime6-root", str(dev), "--procs", "1"], cwd=pathlib.Path(__file__).parents[1], capture_output=True, text=True)
    assert r.returncode == 0 and "FAR_OK" in r.stdout, r.stdout + r.stderr
    rows = [json.loads(x) for x in (data / "chime6tr-far.refined.dialogues.jsonl").read_text().splitlines()]
    assert [x["conv_id"] for x in rows] == ["chime6tr-far:S03@U01.CH1"] and rows[0]["corpus"] == "CHiME6-far" and rows[0]["meta"]["far_of"] == "chime6:S03"
    assert abs(rows[0]["meta"]["mono_src"]["offset_s"] - 0.3) <= 0.02 and (data / "chime6tr-far.build.json").exists()
    assert not (data / "chime6.refined.dialogues.jsonl").exists() and not (data / "chime6-far.refined.dialogues.jsonl").exists()
