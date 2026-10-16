"""engine/thinker.py + engine/decoder.py 테스트 — 작은 무작위 thinker (2 층, D 64, 어휘 151 936 그대로: 특수 id 불변).

- 층 수학: numpy float64 로 짠 Qwen3 (q/k norm → RoPE, GQA, SwiGLU, tied head) 와 MLX fp32 logits 비교.
- 방출 일정: 결정 행의 '절대 위치'마다 NEXT logit 에 ±1e4 를 더한다 (thinker._row_add). 순차·투기·참조는 같은 결정을
  같은 위치에서 하므로 일정이 같고, 어떤 토큰인지는 모델 argmax(KV 문맥 전체)에 달려 되감기·마스크 오류가 드러난다.
- 참조 = decode-contract §c.2 를 그대로 옮긴 stream_decode 식 루프 (NEXT 를 따로 q=1 로 먹임). 디코더 코드와 독립."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mx = pytest.importorskip("mlx.core")

from engine.decoder import MIN_TAIL, SegmentDecoder, k_of  # noqa: E402
from engine.thinker import Adapter, MlxThinker, rope_table  # noqa: E402
from engine.tokens import BLOCKED_EXPECTED, SpecialIds  # noqa: E402

V, D, H, KVH, HD, FF, NL, DIN = 151936, 64, 4, 2, 16, 128, 2, 32
NEXT, EMPTY = 151705, 151706
CFG = dict(num_hidden_layers=NL, hidden_size=D, num_attention_heads=H, num_key_value_heads=KVH, head_dim=HD,
           intermediate_size=FF, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=V)
SP = SpecialIds(next=NEXT, empty=EMPTY, spk_a=151707, spk_b=151708, trans=151726, sem_end=151723, im_start=151644,
                im_end=151645, audio_start=151669, audio_end=151670, audio_pad=151676, asr_text=151704, endoftext=151643,
                delay={d: i for d, i in zip(range(1, 11), [151709, 151710, 151711, 151712, 151713, 151714, 151715, 151716, 151724, 151725])})
PREFIX = [151644, 8948, 198, 151645, 198, 151644, 77091, 198, 11528, 220, 151707, 22574, 198, 14045, 220, 151707, 32079, 151704, 151714]


def tiny_weights(seed=0):
    r = np.random.default_rng(seed)
    n = lambda *s, sc=0.2: (r.standard_normal(s) * sc).astype(np.float32)
    w = {"model.embed_tokens.weight": n(V, D, sc=0.5), "model.norm.weight": 1 + n(D, sc=0.1)}
    for i in range(NL):
        p = f"model.layers.{i}."
        w.update({p + "self_attn.q_proj.weight": n(H * HD, D), p + "self_attn.k_proj.weight": n(KVH * HD, D),
                  p + "self_attn.v_proj.weight": n(KVH * HD, D), p + "self_attn.o_proj.weight": n(D, H * HD),
                  p + "self_attn.q_norm.weight": 1 + n(HD, sc=0.1), p + "self_attn.k_norm.weight": 1 + n(HD, sc=0.1),
                  p + "mlp.gate_proj.weight": n(FF, D), p + "mlp.up_proj.weight": n(FF, D), p + "mlp.down_proj.weight": n(D, FF),
                  p + "input_layernorm.weight": 1 + n(D, sc=0.1), p + "post_attention_layernorm.weight": 1 + n(D, sc=0.1)})
    a = {"net.0.weight": 1 + n(DIN, sc=0.1), "net.0.bias": n(DIN, sc=0.1), "net.1.weight": n(48, DIN), "net.1.bias": n(48),
         "net.3.weight": n(D, 48), "net.3.bias": n(D)}
    return w, a


W_NP, A_NP = tiny_weights()


class SchedThinker(MlxThinker):
    """결정 행 위치 p 에서 NEXT logit 에 bias[p] 를 더한다 (+1e4 = 이 결정은 NEXT, −1e4 = 텍스트)."""

    def __init__(self, bias: np.ndarray, **kw):
        super().__init__({k: mx.array(v) for k, v in W_NP.items()}, CFG, BLOCKED_EXPECTED, **kw)
        self.bias = mx.array(bias.astype(np.float32))
        oh = np.zeros(V, np.float32)
        oh[NEXT] = 1.0
        self.oh = mx.array(oh)

    def _row_add(self, start, sel, block):
        b = self.bias[mx.array([start + s for s in sel], dtype=mx.int32)]
        return block[None, :] + b[:, None] * self.oh[None, :]


def make(p_text, seed=1, max_len=4096, dtype=mx.float32, **kw):
    r = np.random.default_rng(seed)
    bias = np.where(r.random(max_len) < p_text, -1e4, 1e4)
    return SchedThinker(bias, dtype=dtype, max_len=max_len, **kw)


def frames(K, seed=3):
    return mx.array((np.random.default_rng(seed).standard_normal((K, DIN)) * 0.7).astype(np.float32))


ADAPTER = Adapter({k: mx.array(v) for k, v in A_NP.items()})


def seg(th, K, speculative, cap=32, max_flush=8, mtpc=6.0, kv=None, known=True):
    return SegmentDecoder(th, ADAPTER, PREFIX, SP, cap=cap, max_flush=max_flush, kv_capacity=kv, speculative=speculative,
                          max_total_per_chunk=mtpc, known_K=K if known else None)


def run(th, fr, K, speculative, chunks=None, **kw):
    s = seg(th, K, speculative, **kw)
    rounds = []
    if chunks is None:
        rounds += s.feed_frames(fr)
    else:
        i = 0
        for c in chunks:
            rounds += s.feed_frames(fr[i:i + c])
            i += c
    rounds += s.finish_frames(K)
    return s, rounds


def ref_decode(th, fr, K, cap=32, max_flush=8, mtpc=6.0):
    """decode-contract §c.2 / MV:218-246 를 그대로: NEXT 는 따로 q=1 forward (logits 버림)."""
    ce = ADAPTER(fr).astype(th.in_dtype)
    eN, eE = th.embed([NEXT])[0], th.embed([EMPTY])[0]
    th.forward(th.embed(PREFIX), 0)
    st = dict(pos=len(PREFIX), forced=0)
    out, max_total = [], int(mtpc * K) + 64

    def step(x):
        t = th.forward(x[None], st["pos"])[0]
        st["pos"] += 1
        return t

    def emit_round(k, tid):
        n = 0
        while True:
            if tid == NEXT or n >= cap or len(out) >= max_total:
                st["forced"] += int(tid != NEXT)
                step(eN)
                return n
            out.append((k, tid))
            n += 1
            tid = step(th.embed([tid])[0])

    for k in range(K):
        emit_round(k, step(ce[k]))
    rounds = 0
    for r in range(max_flush):
        rounds += 1
        if emit_round(K + r, step(eE)) == 0:
            break
    return out, st["forced"], rounds


def summary(s, rounds):
    return dict(out=s.out, forced=s.forced, flush=s.flush_rounds, pos=s.pos, kv_full=s.kv_full,
                rounds=[(r.k, r.ids, r.forced, r.is_flush, r.kv_full) for r in rounds])


# ───────────────────────── 층 수학 (numpy float64 기준)
def np_forward(X):
    W = {k: v.astype(np.float64) for k, v in W_NP.items()}
    T = X.shape[0]
    rms = lambda x, w: x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
    inv = 1.0 / (1e6 ** (np.arange(0, HD, 2) / HD))
    ang = np.arange(T)[:, None] * inv[None]
    cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
    rot = lambda x: np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)
    h = X.astype(np.float64)
    for i in range(NL):
        p = f"model.layers.{i}."
        hn = rms(h, W[p + "input_layernorm.weight"])
        q = rms((hn @ W[p + "self_attn.q_proj.weight"].T).reshape(T, H, HD), W[p + "self_attn.q_norm.weight"])
        k = rms((hn @ W[p + "self_attn.k_proj.weight"].T).reshape(T, KVH, HD), W[p + "self_attn.k_norm.weight"])
        v = (hn @ W[p + "self_attn.v_proj.weight"].T).reshape(T, KVH, HD)
        q = q * cos[:, None] + rot(q) * sin[:, None]
        k = k * cos[:, None] + rot(k) * sin[:, None]
        k, v = np.repeat(k, H // KVH, axis=1), np.repeat(v, H // KVH, axis=1)     # q 헤드 h ↔ kv 헤드 h//2
        s = np.einsum("thd,shd->hts", q, k) * HD ** -0.5
        s = np.where(np.tril(np.ones((T, T), bool))[None], s, -np.inf)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", a, v).reshape(T, H * HD)
        h = h + o @ W[p + "self_attn.o_proj.weight"].T
        hn = rms(h, W[p + "post_attention_layernorm.weight"])
        g = hn @ W[p + "mlp.gate_proj.weight"].T
        h = h + ((g / (1 + np.exp(-g))) * (hn @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
    return rms(h, W["model.norm.weight"]) @ W["model.embed_tokens.weight"].T


def test_layer_math_vs_numpy():
    th = MlxThinker({k: mx.array(v) for k, v in W_NP.items()}, CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=64)
    X = (np.random.default_rng(7).standard_normal((24, D)) * 0.5).astype(np.float32)
    ref = np_forward(X)
    ok = np.ones(V, bool)
    ok[BLOCKED_EXPECTED] = False
    full = th.forward_logits(mx.array(X), 0, sel=list(range(24)))             # 한 번에
    assert np.isinf(full[:, BLOCKED_EXPECTED]).all() and np.isfinite(full[:, ok]).all()
    assert np.abs(full[:, ok] - ref[:, ok]).max() < 2e-4
    th2 = MlxThinker({k: mx.array(v) for k, v in W_NP.items()}, CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=64)
    th2.forward(mx.array(X[:10]), 0, sel=[])                                     # prefix 뒤 q=1/q=3 증분
    rows = [th2.forward_logits(mx.array(X[i:i + 1]), i)[0] for i in range(10, 16)]
    rows.append(th2.forward_logits(mx.array(X[16:24]), 16, sel=[1, 3, 7])[2])
    assert np.abs(np.stack(rows[:6])[:, ok] - ref[10:16][:, ok]).max() < 2e-4
    assert np.abs(rows[6][ok] - ref[23][ok]).max() < 2e-4
    ids = th2.forward(mx.array(X[16:24]), 16, sel=[1, 3, 7])                   # 같은 자리 다시 쓰기(되감기) 후 같은 argmax
    assert ids == [int(np.argmax(np.where(ok, ref[r], -np.inf))) for r in (17, 19, 23)]


def test_rope_table_matches_hf_formula():
    cos, sin = rope_table(2048, 128, 1e6)
    inv = 1.0 / (1e6 ** (np.arange(0, 128, 2) / 128))
    a = np.arange(2048)[:, None] * inv[None]
    assert np.abs(cos - np.cos(np.concatenate([a, a], -1))).max() < 2e-4     # fp32 각도 반올림 수준 (pos 2047·1 ≈ ulp 1.2e-4)
    assert cos.dtype == np.float32 and cos.shape == (2048, 128)
    assert cos[0].tolist() == [1.0] * 128 and sin[0].tolist() == [0.0] * 128


def test_fp16_rescore_and_finite_flag():
    th32 = MlxThinker({k: mx.array(v) for k, v in W_NP.items()}, CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=64)
    th16 = MlxThinker({k: mx.array(v) for k, v in W_NP.items()}, CFG, BLOCKED_EXPECTED, dtype=mx.float16, max_len=64)
    assert th16.rescore_k == 8 and th16.in_dtype == mx.float32 and th16.E32 is not None
    X = mx.array((np.random.default_rng(9).standard_normal((20, D)) * 0.5).astype(np.float32))
    a = th32.forward(X, 0, sel=list(range(20)))
    b, L = th16.forward_ids_logits(X, 0, sel=list(range(20)))
    assert np.mean(np.array(a) == np.array(b)) >= 0.9 and th16.nonfinite == 0
    assert all(np.isinf(L[:, BLOCKED_EXPECTED]).all(axis=1))
    bad = mx.array(np.full((2, D), np.inf, np.float32))
    th16.forward(bad, 30)
    assert th16.nonfinite == 1


# ───────────────────────── 디코드 루프
@pytest.mark.parametrize("K,p_text,seed", [(37, 0.35, 1), (41, 0.5, 2), (53, 0.2, 3), (8, 0.0, 4)])
def test_spec_equals_seq_equals_ref(K, p_text, seed):
    fr = frames(K, seed)
    o_ref, f_ref, r_ref = ref_decode(make(p_text, seed), fr, K)
    s1, rd1 = run(make(p_text, seed), fr, K, speculative=False)
    s2, rd2 = run(make(p_text, seed), fr, K, speculative=True)
    assert (s1.out, s1.forced, s1.flush_rounds) == (o_ref, f_ref, r_ref)
    assert summary(s1, rd1) == summary(s2, rd2)
    assert [r.k for r in rd1 if not r.is_flush] == list(range(K))             # 프레임마다 Round 하나
    assert s2.stats["forwards"] <= s1.stats["forwards"]
    if p_text == 0.0:
        assert s1.out == [] and s1.flush_rounds == 1


def test_cap_forced_cuts():
    K, fr = 29, frames(29, 5)
    o_ref, f_ref, r_ref = ref_decode(make(0.7, 5), fr, K, cap=2)
    s1, rd1 = run(make(0.7, 5), fr, K, speculative=False, cap=2)
    s2, rd2 = run(make(0.7, 5), fr, K, speculative=True, cap=2)
    assert (s1.out, s1.forced, s1.flush_rounds) == (o_ref, f_ref, r_ref)
    assert summary(s1, rd1) == summary(s2, rd2)
    assert f_ref > 3 and max(len(r.ids) for r in rd1) == 2 and any(r.forced for r in rd1)


def test_total_cap_path():
    K, fr = 50, frames(50, 6)
    o_ref, f_ref, r_ref = ref_decode(make(0.75, 6), fr, K, mtpc=0.3)
    s1, rd1 = run(make(0.75, 6), fr, K, speculative=False, mtpc=0.3)
    s2, rd2 = run(make(0.75, 6), fr, K, speculative=True, mtpc=0.3)
    assert len(o_ref) == int(0.3 * K) + 64                                     # 전체 cap 에 닿음
    assert (s1.out, s1.forced, s1.flush_rounds) == (o_ref, f_ref, r_ref) and r_ref == 1
    assert summary(s1, rd1) == summary(s2, rd2)


def test_flush_runs_to_max_when_always_text():
    K, fr = 5, frames(5, 7)
    o_ref, f_ref, r_ref = ref_decode(make(1.0, 7), fr, K, cap=2)
    s1, rd1 = run(make(1.0, 7), fr, K, speculative=False, cap=2)
    s2, rd2 = run(make(1.0, 7), fr, K, speculative=True, cap=2)
    assert (s1.out, s1.forced, s1.flush_rounds) == (o_ref, f_ref, r_ref) == (o_ref, 13, 8)
    assert summary(s1, rd1) == summary(s2, rd2)
    fl = [r for r in rd1 if r.is_flush]
    assert [r.k for r in fl] == list(range(K, K + 8)) and all(len(r.ids) == 2 and r.is_tail for r in fl)


def test_flush_stops_after_first_empty_round():
    for seed in range(8, 20):
        K, fr = 12, frames(12, seed)
        s, rd = run(make(0.5, seed), fr, K, speculative=True)
        fl = [r for r in rd if r.is_flush]
        assert 1 <= len(fl) == s.flush_rounds <= 8
        assert all(r.ids for r in fl[:-1]) and (not fl[-1].ids or len(fl) == 8)
        o_ref, f_ref, r_ref = ref_decode(make(0.5, seed), fr, K)
        assert (s.out, s.forced, s.flush_rounds) == (o_ref, f_ref, r_ref)


@pytest.mark.parametrize("cap_kv", [60, 75, 97, 140])
def test_kv_clamp_same_in_both_schedules(cap_kv):
    K, fr = 40, frames(40, 9)
    s1, rd1 = run(make(0.4, 9), fr, K, speculative=False, kv=cap_kv)
    s2, rd2 = run(make(0.4, 9), fr, K, speculative=True, kv=cap_kv)
    assert summary(s1, rd1) == summary(s2, rd2)
    assert s1.kv_full and s1.pos <= cap_kv and rd1[-1].kv_full and rd1[-1].forced
    assert not any(r.kv_full for r in rd1[:-1])


@pytest.mark.parametrize("speculative", [False, True])
def test_chunked_feeding_identical(speculative):
    K, fr = 45, frames(45, 11)
    base = summary(*run(make(0.4, 11), fr, K, speculative))
    r = np.random.default_rng(0)
    chunks, tot = [], 0
    while tot < K:
        c = int(r.integers(1, 10))
        chunks.append(min(c, K - tot))
        tot += chunks[-1]
    assert summary(*run(make(0.4, 11), fr, K, speculative, chunks=chunks)) == base


def test_streaming_cap_then_exact_cap_in_finish():
    """K 를 모를 때(스트리밍) 전체 cap 은 6·K_cap+64, finish 에서 정확한 6·K+64 로 바뀐다."""
    K, fr = 30, frames(30, 12)
    s = seg(make(0.3, 12), K, True, known=False)
    assert s.max_total == 6 * 388 + 64
    s.feed_frames(fr)
    s.finish_frames(K)
    assert s.max_total == 6 * K + 64 and s.K == K and s.done
    with pytest.raises(RuntimeError):
        s.feed_frames(fr[:1])


class FakeStream:
    """EncoderStream 흉내: 4096+5120c 샘플마다 4 프레임 (encoder-spec §6)."""

    def __init__(self, fr):
        self.fr, self.n, self.out = fr, 0, 0
        self.samples_fed = self.frames_out = 0

    def _ready(self):
        c = 0
        while 4096 + 5120 * c <= self.n:
            c += 1
        return 4 * c

    def feed(self, pcm):
        self.n += len(pcm)
        r = min(self._ready(), self.fr.shape[0])
        x = self.fr[self.out:r]
        self.out = max(self.out, r)
        return x

    def finish(self, zt):
        self.n += zt
        K = k_of(self.n)
        x = self.fr[self.out:K]
        self.out = K
        return x, K


def test_feed_pcm_path_matches_frames_path():
    n_real = 41000
    K = k_of(n_real + 16000)
    fr = frames(K + 4, 13)
    th = make(0.4, 13)
    s = SegmentDecoder(th, ADAPTER, PREFIX, SP, cap=32, max_flush=8, stream=FakeStream(fr), speculative=True)
    rounds = []
    for i in range(0, n_real, 1234):
        rounds += s.feed(np.zeros(min(1234, n_real - i), np.float32))
    pre = len(rounds)
    assert all(r.k < K and not r.is_tail for r in rounds)
    assert all(r.need_samples == 4096 + 5120 * (r.k // 4) for r in rounds)
    rounds += s.finish(16000)
    assert s.K == K and s.samples_fed == n_real + 16000 and s.stats["overrun"] == 0
    s2, rd2 = run(make(0.4, 13), fr[:K], K, True)
    assert s.out == s2.out and s.forced == s2.forced and s.flush_rounds == s2.flush_rounds
    tail = [r for r in rounds[pre:] if not r.is_flush]
    assert all(r.is_tail == (r.k * 1280 >= n_real) for r in tail) and any(r.is_tail for r in tail)
    assert all(r.need_samples <= n_real + 16000 for r in rounds)
    assert s.features().shape == (K, DIN)


def test_min_tail_floor():
    fr = frames(20, 14)
    s = SegmentDecoder(make(0.3, 14), ADAPTER, PREFIX, SP, cap=32, max_flush=8, stream=FakeStream(fr))
    s.feed(np.zeros(4096 + 5120 * 2, np.float32))
    s.finish(0)
    assert s.stats["tail_raised"] == MIN_TAIL and s.stats["overrun"] == 0


def test_final_redecode_mechanics():
    """final 모드: 별도 KV, 헤더 오디오 자리 교체, 차단 마스크, 상한 K+64 (참조 루프와 같은 id)."""
    class TT:
        def offline_header(self, src, tgt, K):
            return [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669] + [151676] * K + [151670, 151645, 198, 151644,
                                                                                                 77091, 198, 11528, 220, 151707, 22574, 151704], 9
    K, fr = 16, frames(16, 15)
    th = make(0.0, 15)
    blk = np.zeros(V, np.float32)
    blk[[151644, 151669, 151670, 151676, 151704, 151705, 151706] + list(range(151709, 151726))] = -np.inf
    s = SegmentDecoder(th, ADAPTER, PREFIX, SP, cap=32, max_flush=8, known_K=K, tokens=TT(), mode=("English", None),
                       offline_block=mx.array(blk))
    s.feed_frames(fr)
    s.finish_frames(K)
    pos_before = s.pos
    ids, ms = s.final_redecode(max_new=12)
    assert s.pos == pos_before and ms > 0
    # 참조: 같은 헤더를 한 번에 prefill, 그다음 q=1 greedy (offline_decode.py:25-43)
    hdr, a0 = TT().offline_header("English", None, K)
    e = th.embed(hdr)
    e = mx.concatenate([e[:a0], ADAPTER(fr).astype(th.in_dtype), e[a0 + K:]], axis=0)
    kv = th.new_kv(len(hdr) + 20)
    t = th.forward(e, 0, block=mx.array(blk), kv=kv)[0]
    ref, pos = [], len(hdr)
    for _ in range(12):
        if t == 151645:
            break
        ref.append(t)
        t = th.forward(th.embed([t]), pos, block=mx.array(blk), kv=kv)[0]
        pos += 1
    assert ids == ref and not set(ids) & set(np.flatnonzero(np.isinf(blk)).tolist())


# ───────────────────────── W2 의 실제 EncoderStream (작은 무작위 인코더) 와 함께
@pytest.mark.parametrize("n_real", [39496, 41000, 53760])     # 39496+16000: (n−4096) mod 5120 = 200 < 384 (K = 4c+3 경계)
def test_live_feed_with_real_encoder_stream(n_real):
    te = pytest.importorskip("test_encoder")
    enc = pytest.importorskip("engine.encoder")
    m = enc.EncoderModel(te.tiny_weights(), te.tiny_cfg(), right_context=3)
    r = np.random.default_rng(5)
    n = lambda *s, sc=0.2: mx.array((r.standard_normal(s) * sc).astype(np.float32))
    d_enc = int(m.D)
    ad = Adapter({"net.0.weight": 1 + n(d_enc, sc=0.1), "net.0.bias": n(d_enc, sc=0.1), "net.1.weight": n(48, d_enc),
                  "net.1.bias": n(48), "net.3.weight": n(D, 48), "net.3.bias": n(D)})
    wav = te.audio(n_real)
    K = k_of(n_real + 16000)
    feats = enc.offline_features(m, wav, 16000)                                # 기준: 한 번에 넣은 특징
    ref = SegmentDecoder(make(0.4, 21), ad, PREFIX, SP, cap=32, max_flush=8, known_K=K)
    ref.feed_frames(feats)
    ref.finish_frames(K)
    s = SegmentDecoder(make(0.4, 21), ad, PREFIX, SP, cap=32, max_flush=8, stream=m.new_stream())
    rounds, i = [], 0
    while i < n_real:                                                          # 마이크처럼 들쭉날쭉한 패킷
        c = int(r.integers(100, 3000))
        rounds += s.feed(wav[i:i + c])
        i += c
    assert all(rd.k < K for rd in rounds)
    rounds += s.finish(16000)
    assert s.K == K and s.stats["k_mismatch"] is None and s.stats["overrun"] == 0
    assert (s.out, s.forced, s.flush_rounds) == (ref.out, ref.forced, ref.flush_rounds)
    assert np.abs(np.array(s.features()) - np.array(feats)).max() < 1e-5
    assert [rd.k for rd in rounds if not rd.is_flush] == list(range(K))
    assert sum(rd.enc_ms for rd in rounds) > 0


class _TT:
    """final 헤더만 주는 작은 TokenTable 대역."""
    def offline_header(self, src, tgt, K):
        return [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669] + [151676] * K + [151670, 151645, 198, 151644,
                                                                                             77091, 198, 11528, 220, 151707, 22574, 151704], 9


def _final_seg(th, K, fr, blk):
    s = SegmentDecoder(th, ADAPTER, PREFIX, SP, cap=32, max_flush=8, known_K=K, tokens=_TT(), mode=("English", None),
                       offline_block=mx.array(blk))
    s.feed_frames(fr)
    s.finish_frames(K)
    return s


def _offline_blk():
    blk = np.zeros(V, np.float32)
    blk[[151644, 151669, 151670, 151676, 151704, 151705, 151706] + list(range(151709, 151726))] = -np.inf
    return blk


@pytest.mark.parametrize("max_new", [None, 12, 1, 0])
def test_final_job_stepwise_equals_blocking(max_new):
    """final_job().step() 을 한 번씩 = final_redecode(), step 수 = prefill 1 + 토큰마다 1 (마지막 토큰 뒤 낭비 forward 없음)."""
    K, fr, blk = 16, frames(16, 15), _offline_blk()
    th = make(0.0, 15)
    s = _final_seg(th, K, fr, blk)
    ref, _ = s.final_redecode(max_new=max_new)
    job = s.final_job(max_new=max_new)
    n = 0
    while not job.step():
        n += 1
        assert n < 200
    assert job.done and job.ids == ref and job.step() is True
    lim = K + 64 if max_new is None else max_new
    assert len(ref) <= max(lim, 0)
    if lim > 0:
        assert job.steps == len(ref) + (len(ref) < lim)       # 상한에 닿으면 <|im_end|> 확인 forward 가 없다
    else:
        assert job.steps == 0 and ref == []


def test_final_job_interleaved_with_next_segment_streaming():
    """끝난 세그먼트의 final 을 다음 세그먼트 스트리밍과 번갈아 돌려도 둘 다 따로 돌린 결과와 같다 (별도 KV)."""
    K, fr, blk = 16, frames(16, 15), _offline_blk()
    th = make(0.3, 16)
    a = _final_seg(th, K, fr, blk)
    ref_final, _ = a.final_redecode()
    K2, fr2 = 21, frames(21, 17)
    b0, _ = run(th, fr2, K2, True)
    ref_b = (list(b0.out), b0.forced, b0.flush_rounds)
    job = a.final_job()
    b = seg(th, K2, True)                                       # 새 세그먼트: 공유 스트리밍 KV 를 처음부터 쓴다
    job.step()
    for i in range(0, K2, 4):
        b.feed_frames(fr2[i:i + 4])
        job.step()
    b.finish_frames(K2)
    while not job.step():
        pass
    assert job.ids == ref_final
    assert (list(b.out), b.forced, b.flush_rounds) == ref_b


def test_final_job_requires_finish():
    th = make(0.0, 15)
    s = SegmentDecoder(th, ADAPTER, PREFIX, SP, cap=32, max_flush=8, known_K=8, tokens=_TT(), mode=("English", None),
                       offline_block=mx.array(_offline_blk()))
    with pytest.raises(RuntimeError):
        s.final_job()
