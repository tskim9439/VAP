import random

import numpy as np

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


def test_chain_skips_none_and_settings():
    ch = AugChain(None, PhoneAugment(p=1.0))
    assert len(ch.augs) == 1 and ch.settings()["chain"][0]["kind"] == "phone"
    x = _signal(); assert ch(x, random.Random(3)).shape == x.shape
