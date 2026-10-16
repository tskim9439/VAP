"""engine/frontend.py + engine/encoder.py 단위 테스트 — 작은 랜덤 설정(2 레이어, D 64), 가중치 파일 불필요.

기준: numpy float64 로 짠 NeMo 오프라인 masked forward (chunked_limited 마스크, rel-pos, causal conv) —
스트리밍 결과가 오프라인[:K] 와 같아야 한다 (encoder-spec §6-§7).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mx = pytest.importorskip("mlx.core")

from engine.frontend import HOP, MelStream, final_frames, logmel_full  # noqa: E402
from engine.encoder import EncoderModel, k_frames, offline_features  # noqa: E402

D, H, NL, SUB_CH, FF, KC = 64, 4, 2, 8, 256, 9
ATOL = 2e-4


# ------------------------------------------------------------------ 고정물
def tiny_cfg(R=3):
    return {"preprocessor": {"sample_rate": 16000, "normalize": "NA", "window_size": 0.025, "window_stride": 0.01,
                             "window": "hann", "features": 128, "n_fft": 512, "log": True, "frame_splicing": 1,
                             "dither": 1e-05, "pad_to": 0, "pad_value": 0.0},
            "encoder": {"feat_in": 128, "feat_out": -1, "n_layers": NL, "d_model": D, "use_bias": False,
                        "subsampling": "dw_striding", "subsampling_factor": 8, "subsampling_conv_channels": SUB_CH,
                        "causal_downsampling": True, "ff_expansion_factor": FF // D, "self_attention_model": "rel_pos",
                        "n_heads": H, "att_context_size": [[56, 3], [56, 0], [56, 6], [56, 13]],
                        "att_context_style": "chunked_limited", "xscaling": False, "untie_biases": True,
                        "conv_kernel_size": KC, "conv_norm_type": "layer_norm", "conv_context_size": "causal"}}


def tiny_weights(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda *s: (rng.standard_normal(s) / np.sqrt(np.prod(s[1:]))).astype(np.float32)   # noqa: E731
    sd = {}
    p = "enc.pre_encode."
    for i in (0, 2, 5):
        sd[f"{p}conv.{i}.weight"] = w(SUB_CH, 1, 3, 3) * 3
        sd[f"{p}conv.{i}.bias"] = (0.1 * rng.standard_normal(SUB_CH)).astype(np.float32)
    for i in (3, 6):
        sd[f"{p}conv.{i}.weight"] = w(SUB_CH, SUB_CH, 1, 1)
        sd[f"{p}conv.{i}.bias"] = (0.1 * rng.standard_normal(SUB_CH)).astype(np.float32)
    sd[p + "out.weight"] = w(D, SUB_CH * 17)
    sd[p + "out.bias"] = (0.1 * rng.standard_normal(D)).astype(np.float32)
    for l in range(NL):
        q = f"enc.layers.{l}."
        for n in ("norm_feed_forward1", "norm_self_att", "norm_conv", "conv.batch_norm", "norm_feed_forward2", "norm_out"):
            sd[q + n + ".weight"] = (1 + 0.2 * rng.standard_normal(D)).astype(np.float32)
            sd[q + n + ".bias"] = (0.2 * rng.standard_normal(D)).astype(np.float32)
        for n in ("feed_forward1", "feed_forward2"):
            sd[q + n + ".linear1.weight"] = w(FF, D)
            sd[q + n + ".linear2.weight"] = w(D, FF)
        for n in ("q", "k", "v", "out", "pos"):
            sd[q + f"self_attn.linear_{n}.weight"] = w(D, D) * (2 if n in "qk" else 1)
        sd[q + "self_attn.pos_bias_u"] = (0.3 * rng.standard_normal((H, D // H))).astype(np.float32)
        sd[q + "self_attn.pos_bias_v"] = (0.3 * rng.standard_normal((H, D // H))).astype(np.float32)
        sd[q + "conv.pointwise_conv1.weight"] = w(2 * D, D, 1)
        sd[q + "conv.depthwise_conv.weight"] = w(D, 1, KC) * 2
        sd[q + "conv.pointwise_conv2.weight"] = w(D, D, 1)
    sd["pre.featurizer.window"] = np.hanning(400).astype(np.float32)          # 대칭 hann, w[0]=w[399]=0
    sd["pre.featurizer.fb"] = (np.abs(rng.standard_normal((1, 128, 257))) * 0.05).astype(np.float32)
    return sd


def audio(n, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    x = 0.2 * np.sin(2 * np.pi * (150 + 40 * np.sin(2 * np.pi * 0.9 * t)) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = x + 0.01 * rng.standard_normal(n)
    if n > 12000:
        x[8000:12000] = 0.0                                                     # 디지털 무음 구간
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def sd():
    return tiny_weights()


@pytest.fixture(scope="module")
def model(sd):
    return EncoderModel(sd, tiny_cfg(), right_context=3)


@pytest.fixture(scope="module")
def model0(sd):
    return EncoderModel(sd, tiny_cfg(), right_context=0)


# ------------------------------------------------------------------ numpy 오프라인 기준 (float64)
def _ln(x, w, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _conv(h, w, b, depthwise):
    hp = np.pad(h, ((2, 1), (2, 1), (0, 0)))
    To, Fo = (hp.shape[0] - 3) // 2 + 1, (hp.shape[1] - 3) // 2 + 1
    out = np.zeros((To, Fo, w.shape[0]))
    for dt in range(3):
        for df in range(3):
            patch = hp[dt: dt + 2 * To: 2, df: df + 2 * Fo: 2, :]
            out += patch * w[:, 0, dt, df] if depthwise else patch @ w[:, :, dt, df].T
    return out + b


def ref_offline(sd, mel, C=56, R=3):
    g = lambda k: np.asarray(sd[k], np.float64)   # noqa: E731
    p = "enc.pre_encode."
    h = np.maximum(_conv(np.asarray(mel, np.float64)[:, :, None], g(p + "conv.0.weight"), g(p + "conv.0.bias"), False), 0)
    for a, b in ((2, 3), (5, 6)):
        h = _conv(h, g(f"{p}conv.{a}.weight"), g(f"{p}conv.{a}.bias"), True)
        h = np.maximum(h @ g(f"{p}conv.{b}.weight")[:, :, 0, 0].T + g(f"{p}conv.{b}.bias"), 0)
    Tn = h.shape[0]
    x = h.transpose(0, 2, 1).reshape(Tn, -1) @ g(p + "out.weight").T + g(p + "out.bias")   # 열 = c*17+f
    ci = np.arange(Tn) // (R + 1)
    dc = ci[:, None] - ci[None, :]
    allowed = (dc >= 0) & (dc <= C // (R + 1))
    rels = np.arange(Tn - 1, -Tn, -1)
    div = np.exp(np.arange(0, D, 2) * -(np.log(10000.0) / D))
    pe = np.zeros((len(rels), D))
    pe[:, 0::2], pe[:, 1::2] = np.sin(rels[:, None] * div), np.cos(rels[:, None] * div)
    i, j = np.arange(Tn)[:, None], np.arange(Tn)[None]
    row = (Tn - 1) - (i - j)                                                    # pe 행 ↔ rel = i - j
    DK = D // H
    for l in range(NL):
        q_ = f"enc.layers.{l}."
        L = lambda n: (g(q_ + n + ".weight"), g(q_ + n + ".bias"))   # noqa: E731
        ff = lambda z, n: (lambda u: u * _sig(u))(z @ g(q_ + n + ".linear1.weight").T) @ g(q_ + n + ".linear2.weight").T  # noqa: E731
        r = x + 0.5 * ff(_ln(x, *L("norm_feed_forward1")), "feed_forward1")
        a = _ln(r, *L("norm_self_att"))
        q = (a @ g(q_ + "self_attn.linear_q.weight").T).reshape(Tn, H, DK)
        k = (a @ g(q_ + "self_attn.linear_k.weight").T).reshape(Tn, H, DK)
        v = (a @ g(q_ + "self_attn.linear_v.weight").T).reshape(Tn, H, DK)
        P = (pe @ g(q_ + "self_attn.linear_pos.weight").T).reshape(-1, H, DK)
        u, vb = g(q_ + "self_attn.pos_bias_u"), g(q_ + "self_attn.pos_bias_v")
        ac = np.einsum("ihd,jhd->hij", q + u, k)
        bd = np.einsum("ihd,ijhd->hij", q + vb, P[row])
        s = (ac + bd) / np.sqrt(DK)
        s = np.where(allowed[None], s, -np.inf)
        pr = np.exp(s - s.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        o = np.einsum("hij,jhd->ihd", pr, v).reshape(Tn, D)
        r = r + o @ g(q_ + "self_attn.linear_out.weight").T
        gl = _ln(r, *L("norm_conv")) @ g(q_ + "conv.pointwise_conv1.weight")[:, :, 0].T
        gl = gl[:, :D] * _sig(gl[:, D:])
        gp = np.concatenate([np.zeros((KC - 1, D)), gl])
        dw = g(q_ + "conv.depthwise_conv.weight")[:, 0, :]
        y = sum(gp[t: t + Tn] * dw[:, t] for t in range(KC))
        z = _ln(y, *L("conv.batch_norm"))
        r = r + (z * _sig(z)) @ g(q_ + "conv.pointwise_conv2.weight")[:, :, 0].T
        r = r + 0.5 * ff(_ln(r, *L("norm_feed_forward2")), "feed_forward2")
        x = _ln(r, *L("norm_out"))
    return x


def stream_all(model, x, tail, packets=None, seed=0):
    """패킷 단위 feed + finish → (프레임 np, K, feed 반환 크기들)."""
    st = model.new_stream()
    rng = np.random.default_rng(seed)
    outs, sizes, pos = [], [], 0
    while pos < len(x):
        b = int(rng.choice(packets)) if packets else len(x)
        o = st.feed(x[pos: pos + b])
        sizes.append(o.shape[0])
        outs.append(np.array(o))
        pos += b
    o, K = st.finish(tail)
    outs.append(np.array(o))
    return np.concatenate(outs) if outs else np.zeros((0, D)), K, sizes, st


# ------------------------------------------------------------------ frontend
def test_mel_stream_any_packets_matches_full(sd):
    x = audio(23456)
    full = logmel_full(x, sd["pre.featurizer.window"], sd["pre.featurizer.fb"])
    assert full.shape == (len(x) // HOP, 128)
    ms = MelStream(sd["pre.featurizer.window"], sd["pre.featurizer.fb"])
    rng = np.random.default_rng(3)
    outs, pos = [], 0
    while pos < len(x):
        b = int(rng.choice([1, 7, 160, 333, 5120, int(rng.integers(1, 3000))]))
        outs.append(ms.push_frames(x[pos: pos + b]))
        pos = min(len(x), pos + b)
        assert ms.t == final_frames(pos)                                           # 보수 규칙
    ms.finish()
    outs.append(ms.take(ms.ready()))
    inc = np.concatenate(outs)
    assert inc.shape == full.shape
    np.testing.assert_allclose(inc, full, atol=1e-5, rtol=0)


def test_mel_final_frame_rule():
    assert [final_frames(n) for n in (0, 255, 256, 415, 416, 4095, 4096)] == [0, 0, 1, 1, 2, 24, 25]
    for n in range(256, 50000, 37):
        F = final_frames(n)
        assert n >= 160 * (F - 1) + 256 and n < 160 * F + 256 or F == n // 160


# ------------------------------------------------------------------ 스트리밍 = 오프라인
@pytest.mark.parametrize("n_real,tail", [(160 * (25 + 32 * 5 + r) + 37, 0) for r in (0, 3, 7, 8, 9, 16, 31)]
                         + [(40000, 16000), (30080 - 16000, 16000), (2000, 3200), (3000, 0)])
def test_stream_matches_offline_r3(model, sd, n_real, tail):
    x = audio(n_real)
    y, K, _, st = stream_all(model, x, tail, packets=[1, 160, 333, 5120])
    xt = np.concatenate([x, np.zeros(tail, np.float32)])
    mel = logmel_full(xt, sd["pre.featurizer.window"], sd["pre.featurizer.fb"])
    ref = ref_offline(sd, mel, R=3)
    assert K == k_frames(len(xt)) and K <= len(y) == st.frames_out <= len(ref)   # feed 는 K 이상 프레임을 낼 수 있다 (꼬리 0)
    np.testing.assert_allclose(y, ref[:len(y)], atol=ATOL, rtol=0)


@pytest.mark.parametrize("n_real,tail", [(30000, 0), (20000, 16000), (500, 1000)])
def test_stream_matches_offline_r0(model0, sd, n_real, tail):
    x = audio(n_real, seed=5)
    y, K, _, _ = stream_all(model0, x, tail, packets=[1, 160, 333, 5120])
    xt = np.concatenate([x, np.zeros(tail, np.float32)])
    ref = ref_offline(sd, logmel_full(xt, sd["pre.featurizer.window"], sd["pre.featurizer.fb"]), R=0)
    assert K <= len(y) <= len(ref)
    np.testing.assert_allclose(y, ref[:len(y)], atol=ATOL, rtol=0)


# ------------------------------------------------------------------ 패킷 불변성 / 상태 격리
def test_packet_sizes_bit_identical(model):
    x = audio(70001, seed=2)
    one, K1, _, _ = stream_all(model, x, 16000)
    for packets, seed in (([1, 160, 333, 5120], 0), ([5120], 1), ([1, 2, 3], 2), ([4096, 512], 3), ([511, 160, 7919], 4)):
        if packets == [1, 2, 3]:
            xs = x[:9000]
            a, Ka, _, _ = stream_all(model, xs, 16000)
            b, Kb, sizes, _ = stream_all(model, xs, 16000, packets, seed)
        else:
            a, Ka = one, K1
            b, Kb, sizes, _ = stream_all(model, x, 16000, packets, seed)
        assert Ka == Kb and all(s % 4 == 0 for s in sizes)
        assert np.array_equal(a, b), packets


def test_chunk_emission_threshold(model, model0):
    x = audio(20000)
    for m, c_max in ((model, 3), (model0, 10)):
        st = m.new_stream()
        fed = 0
        for c in range(c_max):
            need = st.chunk_samples(c)
            o = st.feed(x[fed: need - 1]); fed = need - 1
            assert st.frames_out == m.Tm * c, (c, o.shape)
            o = st.feed(x[fed: need]); fed = need
            assert o.shape[0] == m.Tm and st.frames_out == m.Tm * (c + 1)
    assert model.new_stream().chunk_samples(2) == 4096 + 5120 * 2
    assert model0.new_stream().chunk_samples(3) == 256 + 1280 * 3


def test_fresh_streams_do_not_leak(model):
    xa, xb = audio(30000, seed=7), audio(25000, seed=8)
    a1, _, _, _ = stream_all(model, xa, 16000)
    b1, _, _, _ = stream_all(model, xb, 16000)
    a2, _, _, _ = stream_all(model, xa, 16000)
    assert np.array_equal(a1, a2)
    # 두 스트림을 번갈아 먹여도 같다
    s1, s2 = model.new_stream(), model.new_stream()
    o1, o2 = [], []
    for i in range(0, 30000, 1000):
        o1.append(np.array(s1.feed(xa[i:i + 1000])))
        o2.append(np.array(s2.feed(xb[i:i + 1000])))
    o1.append(np.array(s1.finish(16000)[0])); o2.append(np.array(s2.finish(16000)[0]))
    assert np.array_equal(np.concatenate(o1), a1) and np.array_equal(np.concatenate(o2), b1)


# ------------------------------------------------------------------ finish(): K / 자르기 / 가장자리
def test_k_formula_eval_expression():
    assert k_frames(69760) == 55 and round(69760 / 1280) == 54
    assert k_frames(73600) == 57 and k_frames(30080) == 24


def test_finish_n_total_69760(model, sd):
    x = audio(69760 - 16000, seed=9)
    y, K, _, st = stream_all(model, x, 16000, packets=[512])
    assert K == 55 and len(y) == 55 and st.frames_out == 55
    ref = ref_offline(sd, logmel_full(np.pad(x, (0, 16000)), sd["pre.featurizer.window"], sd["pre.featurizer.fb"]))
    np.testing.assert_allclose(y, ref[:55], atol=ATOL, rtol=0)


@pytest.mark.parametrize("c,r", [(2, 0), (5, 200), (13, 383), (13, 400)])
def test_finish_edge_no_tail(model, c, r):
    """n_total = 4096+5120c+r, 꼬리 0: feed 가 청크 c 를 이미 냈다 → 프레임 4c+3 은 r<384 이면 K 이상 (호출자가 거른다)."""
    n = 4096 + 5120 * c + r
    st = model.new_stream()
    o = st.feed(audio(n, seed=c))
    assert o.shape[0] == 4 * (c + 1)
    K_now = st.k_received
    rest, K = st.finish(0)
    assert K == K_now == k_frames(n)
    assert K == (4 * c + 3 if r < 384 else 4 * c + 4)                          # r=400: 4c+3.5125 → 4c+4
    assert rest.shape[0] == max(0, K - 4 * (c + 1))


@pytest.mark.parametrize("c,r", [(3, 0), (6, 200), (13, 383)])
def test_finish_tail_completes_chunk_truncated(model, sd, c, r):
    """0 꼬리가 청크 c 를 완성: finish 는 프레임 ≥ K 를 내지 않는다 (합계 == K == 4c+3)."""
    n_total = 4096 + 5120 * c + r
    tail = 1200
    x = audio(n_total - tail, seed=c + 20)
    st = model.new_stream()
    o = st.feed(x)
    assert o.shape[0] == 4 * c
    rest, K = st.finish(tail)
    assert K == 4 * c + 3 and rest.shape[0] == 3 and st.frames_out == K
    ref = ref_offline(sd, logmel_full(np.pad(x, (0, tail)), sd["pre.featurizer.window"], sd["pre.featurizer.fb"]))
    np.testing.assert_allclose(np.concatenate([np.array(o), np.array(rest)]), ref[:K], atol=ATOL, rtol=0)


@pytest.mark.parametrize("n_real,tail", [(0, 0), (0, 600), (0, 1000), (100, 900), (0, 3200), (4000, 96), (4000, 97)])
def test_finish_tiny_segments(model, sd, n_real, tail):
    x = audio(n_real) if n_real else np.zeros(0, np.float32)
    st = model.new_stream()
    if n_real:
        st.feed(x)
    rest, K = st.finish(tail)
    n = n_real + tail
    assert K == k_frames(n) and st.samples_fed == n
    assert st.frames_out == K and rest.shape == (K, D)                       # S ≥ K 이므로 0 채움이 필요 없다
    if K:
        ref = ref_offline(sd, logmel_full(np.pad(x, (0, tail)), sd["pre.featurizer.window"], sd["pre.featurizer.fb"]))
        np.testing.assert_allclose(np.array(rest), ref[:K], atol=ATOL, rtol=0)


@pytest.mark.parametrize("n_real,tail", [(4096 + 5120 * 6 - 385, 385), (4096 + 5120 * 6 + 200 - 1200, 1200), (30000, 16000)])
def test_live_tail_equals_finish_tail(model, n_real, tail):
    """0 꼬리를 feed 로 넣고 finish(0) 해도 [:K] 는 finish(tail) 과 비트 단위로 같다. feed 는 K 이상 프레임을 낼 수 있다."""
    x = audio(n_real, seed=11)
    a, K, _, _ = stream_all(model, x, tail, packets=[333])
    b, Kb, _, st = stream_all(model, np.concatenate([x, np.zeros(tail, np.float32)]), 0, packets=[333])
    assert K == Kb == len(a) and len(b) >= K and st.frames_out == len(b)
    assert np.array_equal(a, b[:K])


def test_finish_twice_and_feed_after_finish_fail(model):
    st = model.new_stream()
    st.finish(1000)
    with pytest.raises(AssertionError):
        st.feed(np.zeros(10, np.float32))
    with pytest.raises(AssertionError):
        st.finish(0)


def test_offline_features_matches_stream(model):
    x = audio(33333, seed=4)
    f = np.array(offline_features(model, x, 16000))
    y, K, _, _ = stream_all(model, x, 16000, packets=[333])
    assert f.shape == (K, D) and np.array_equal(f, y)


# ------------------------------------------------------------------ 정밀도 / 컴파일 경로
def test_eager_probe_path_matches_compiled(sd):
    m = EncoderModel(sd, tiny_cfg(), right_context=3)
    x = audio(20000, seed=6)
    a = np.array(offline_features(m, x, 16000))
    seen = []
    m.probe = lambda name, li, arr: seen.append((name, li))
    b = np.array(offline_features(m, x, 16000))
    assert ("qkv", 0) in seen and ("resid", NL - 1) in seen
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)


@pytest.mark.parametrize("dt", ["float16", "bfloat16"])
def test_low_precision_close(sd, dt):
    x = audio(30000, seed=10)
    a = np.array(offline_features(EncoderModel(sd, tiny_cfg(), dtype=mx.float32), x, 16000))
    b = np.array(offline_features(EncoderModel(sd, tiny_cfg(), dtype=getattr(mx, dt)), x, 16000))
    assert np.isfinite(b).all()
    assert np.abs(a - b).max() < (0.02 if dt == "float16" else 0.1)


def test_warmup_and_timing_fields(model):
    model.warmup()
    st = model.new_stream()
    st.feed(audio(4096))
    assert st.frames_out == 4 and st.last_step_ms > 0 and len(st.step_ms) == 1
    st.feed(np.zeros(100, np.float32))
    assert st.last_step_ms == 0.0 and st.step_ms == []


def test_pre_endpoint_frames_below_k_with_tail_floor(model):
    """꼬리 ≥ 200 ms 면 feed() 가 끝 전에 낸 프레임은 모두 k < K (decode-contract §d.5, audio-ui-spec §5.2)."""
    st = model.new_stream()
    th = np.array([st.chunk_samples(c) for c in range(200)])
    for n in range(0, 480000, 7):
        emitted = 4 * int((th <= n).sum())
        assert emitted <= k_frames(n + 3200), n


def test_unsupported_cfg_rejected(sd):
    cfg = tiny_cfg()
    cfg["encoder"]["xscaling"] = True
    with pytest.raises(ValueError):
        EncoderModel(sd, cfg)
    cfg = tiny_cfg()
    cfg["preprocessor"]["normalize"] = "per_feature"
    with pytest.raises(ValueError):
        EncoderModel(sd, cfg)
    with pytest.raises(ValueError):
        EncoderModel(sd, tiny_cfg(), right_context=2)


def test_consume_builds_identical_model_and_empties_dict():
    """consume=True (Engine 적재): 원본 dict 를 비우고 텐서를 하나씩 꺼내 써도 같은 모델 (적재 최대 메모리↓, F16)."""
    sd = tiny_weights()
    x = audio(30000)
    a = np.array(offline_features(EncoderModel(dict(sd), tiny_cfg(), dtype=mx.float16), x, 16000))
    src = dict(sd)
    b = np.array(offline_features(EncoderModel(src, tiny_cfg(), dtype=mx.float16, consume=True), x, 16000))
    assert src == {} and np.array_equal(a, b)
