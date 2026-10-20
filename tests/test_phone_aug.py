import random

import numpy as np
import pytest

from vapasr.data.phone_aug import AugChain, PhoneAugment, mulaw_roundtrip


def _signal(n=32000, sr=16000, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n) / sr
    x = 0.3 * np.sin(2 * np.pi * 700 * t) + 0.2 * np.sin(2 * np.pi * 6000 * t) + 0.05 * rng.standard_normal(n)
    x[12000:12040] += 0.6                                                    # 시각 기준이 되는 짧은 펄스
    return x.astype(np.float32)


def test_off_and_shape():
    x = _signal(); r = random.Random(0)
    assert np.array_equal(PhoneAugment(p=0.0)(x, r), x)
    y = PhoneAugment(p=1.0)(x, r)
    assert y.shape == x.shape and y.dtype == np.float32 and np.max(np.abs(y)) <= 1.0


def test_band_and_no_delay():
    x = _signal(); y = PhoneAugment(p=1.0, mulaw_p=0.0)(x, random.Random(1))
    f = np.fft.rfftfreq(len(x), 1 / 16000); X, Y = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y))
    hi = f > 4200
    assert Y[hi].sum() < 0.02 * X[hi].sum()                                   # 4 kHz 위는 사라진다(6 kHz 성분 포함)
    lo = (f > 600) & (f < 800)
    assert Y[lo].sum() > 0.5 * X[lo].sum()                                    # 전화 대역 안 성분은 남는다
    c = np.correlate(y[11000:13000], x[11000:13000], mode="full"); lag = int(np.argmax(c)) - (2000 - 1)
    assert abs(lag) <= 1                                                     # 영위상 — 단어 시각이 밀리지 않는다


def test_rms_and_mulaw():
    x = _signal(); y = PhoneAugment(p=1.0, mulaw_p=1.0)(x, random.Random(2))
    assert abs(np.sqrt(np.mean(y ** 2)) - np.sqrt(np.mean(x ** 2))) < 1e-3
    z = mulaw_roundtrip(np.linspace(-1, 1, 1001).astype(np.float32))
    assert np.max(np.abs(z - np.linspace(-1, 1, 1001))) < 0.03 and z.dtype == np.float32


def test_info_records_draws_without_changing_output():
    """info(화자별 전화 채널 meta.chan)는 출력·난수 소비를 바꾸지 않고, 건 경우에만 뽑은 대역·μ-law 결정을 담는다."""
    x = _signal()
    for p, seed in ((1.0, 5), (1.0, 6), (0.5, 7), (0.5, 9), (0.0, 8)):
        r1, r2, info = random.Random(seed), random.Random(seed), {}
        y1, y2 = PhoneAugment(p=p)(x, r1), PhoneAugment(p=p)(x, r2, info=info)
        assert np.array_equal(y1, y2) and r1.random() == r2.random()
        r3 = random.Random(seed)
        if r3.random() < p: assert info == dict(low=round(r3.uniform(200.0, 400.0), 1), high=round(r3.uniform(3000.0, 3600.0), 1), mulaw=r3.random() < 0.5)
        else: assert info == {}


def test_default_call_matches_pre_info_golden():
    """info 를 더하기 전 코드(dfe1a8d)로 잰 출력 해시·호출 뒤 rng 다음 값 — 기본 호출이 그대로다(잰 환경에서만)."""
    import hashlib, platform, sys, scipy
    if (sys.platform, platform.machine(), np.__version__, scipy.__version__) != ("darwin", "arm64", "1.24.3", "1.10.1"): pytest.skip("골든은 darwin-arm64 numpy 1.24.3 scipy 1.10.1")
    x = (0.2 * np.random.default_rng(3).standard_normal(48000)).astype(np.float32)
    for p, seed, h, nxt in ((1.0, 5, "585f11dff45ac13e", 0.7398985747399307), (1.0, 6, "167e059f646b802a", 0.00045171488507100843),
                            (0.5, 7, "c34d19b099c99386", 0.5358820043066892), (0.0, 8, "c1dc7ed39a6deddf", 0.9622950358343828)):
        r = random.Random(seed); y = PhoneAugment(p=p, mulaw_p=0.5)(x, r)
        assert hashlib.sha256(np.asarray(y, np.float32).tobytes()).hexdigest()[:16] == h and r.random() == nxt


def test_chain_skips_none_and_settings():
    ch = AugChain(None, PhoneAugment(p=1.0))
    assert len(ch.augs) == 1 and ch.settings()["chain"][0]["kind"] == "phone"
    x = _signal(); assert ch(x, random.Random(3)).shape == x.shape
