"""vapasr/data/noise_aug — 합성 뱅크로 SNR·잔향 정렬(지연 없음)·길이·끔·결정성 검사(CPU)."""
import json, math, random
import numpy as np, pytest
sf = pytest.importorskip("soundfile"); pytest.importorskip("scipy")
from vapasr.data.noise_aug import NoiseAugment, speech_power, from_args

SR = 16000


def _bank(tmp_path, rirs):
    rng = np.random.default_rng(0); (tmp_path / "audio").mkdir()
    rows = []
    for name, cat, sec in (("n0", "noise", 3.0), ("short", "noise", 0.7), ("m0", "music", 2.0)):
        x = (0.1 * rng.standard_normal(int(sec * SR))).astype(np.float32); p = tmp_path / "audio" / f"{name}.wav"; sf.write(p, x, SR, subtype="PCM_16")
        rows.append(dict(path=str(p), source="t", category=cat, n=len(x), dur=sec))
    (tmp_path / "noise.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    L = max(len(r) for r in rirs); bank = np.zeros((len(rirs), L), np.float16)
    for i, r in enumerate(rirs): bank[i, : len(r)] = r
    np.save(tmp_path / "rir.npy", bank); (tmp_path / "rir.jsonl").write_text("".join(json.dumps(dict(i=i, source="t", n=len(r))) + "\n" for i, r in enumerate(rirs)))
    (tmp_path / "summary.json").write_text("{}")
    return str(tmp_path)


def _speech(n=2 * SR):
    t = np.arange(n) / SR; x = 0.05 * np.sin(2 * math.pi * 220 * t).astype(np.float32); x[: n // 4] = 0; return x   # 앞 1/4 무음


def test_snr_uses_active_frames_and_length_is_kept(tmp_path):
    aug = NoiseAugment(_bank(tmp_path, [np.array([1.0])]), p_noise=1.0, p_rir=0.0)
    x = _speech(); r = aug.noise[0]
    for snr in (0.0, 10.0, 20.0):
        y = aug.add_noise(x, random.Random(1), r=r, snr_db=snr); nz = y - x
        assert y.shape == x.shape and y.dtype == np.float32
        assert abs(10 * math.log10(speech_power(x) / float(np.mean(nz.astype(np.float64) ** 2))) - snr) < 0.05      # 활성 프레임 기준 SNR
    short = next(q for q in aug.noise if q["dur"] < 1)                                           # 발화보다 짧은 잡음은 이어 붙인다
    assert aug.add_noise(x, random.Random(2), r=short, snr_db=5).shape == x.shape


def test_reverb_is_aligned_to_direct_path_no_delay(tmp_path):
    h = np.zeros(400, np.float32); h[120] = 1.0; h[200] = 0.4; h[300] = -0.2                        # 직접음이 120 샘플 뒤에 있는 RIR
    aug = NoiseAugment(_bank(tmp_path, [h]), p_noise=0.0, p_rir=1.0)
    x = np.zeros(8000, np.float32); x[1000] = 1.0
    y = aug.reverb(x, random.Random(0), i=0)
    assert y.shape == x.shape and int(np.argmax(np.abs(y))) == 1000                                 # 임펄스 위치 그대로(지연 없음)
    assert abs(math.sqrt(float(np.mean(y.astype(np.float64) ** 2))) - math.sqrt(float(np.mean(x.astype(np.float64) ** 2)))) < 1e-4   # RMS 유지


def test_call_off_deterministic_and_settings(tmp_path):
    b = _bank(tmp_path, [np.array([1.0, 0.3], np.float32)])
    off = NoiseAugment(b, p_noise=0.0, p_rir=0.0); x = _speech()
    rng = random.Random(3); assert np.array_equal(off(x, rng), x) and rng.random() == random.Random(3).random()   # 끄면 난수를 쓰지 않는다
    on = NoiseAugment(b, p_noise=1.0, p_rir=1.0)
    assert np.array_equal(on(x, random.Random(7)), on(x, random.Random(7))) and not np.array_equal(on(x, random.Random(7)), x)
    assert np.abs(on(x * 19.0, random.Random(9))).max() <= 1.0                                     # 클립 방지
    st = on.settings(); assert st["n_noise"] == 2 and st["n_music"] == 1 and st["n_rir"] == 1 and st["bank_summary_sha256"]
    assert from_args(None, 0.4, "5,30", 0.2) is None
    with pytest.raises(ValueError): from_args(b, 0.4, "30,5", 0.2)
