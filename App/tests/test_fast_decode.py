"""엔진 가속 경로가 순차 디코드와 같은 id 를 내는지 — 작은 무작위 thinker (test_decoder 의 fixture), 가중치 없음.

- q4 자기 초안 + 검증 forward + 라운드 끝 투기(RES): 순차·투기 배치 디코더와 Round·out·forced·flush·pos 가 같다
  (정확한 초안, 30 % 틀린 초안, 늘 틀림/늘 NEXT/NEXT 없음/무작위, 실제 q4 초안, cap 1–3, 전체 cap, KV 경계 sweep).
- on_token 훅: 호출마다 훅 순서 == 돌려준 Round.ids 를 이은 것 == seg.out.
- kv_full 뒤 인코더 미루기, prefix KV 캐시, final lookahead, fp32 두 단계 헤드."""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mx = pytest.importorskip("mlx.core")

import test_decoder as T  # noqa: E402
from engine.decoder import FinalJob, PrefixCache, SegmentDecoder, k_of, stream_draft  # noqa: E402
from engine.thinker import F32, Draft, MlxThinker, TwoStageHead  # noqa: E402
from engine.tokens import BLOCKED_EXPECTED  # noqa: E402


class SchedDraft(Draft):
    """목표와 같은 일정 bias 를 쓰는 초안. exact=True 면 목표 층 그대로(완전 일치), 아니면 실제 q4 층.
    kind: None | 'wrong' (늘 +7) | 'next' (늘 NEXT) | 'nonext' (NEXT 대신 텍스트) | 'rand'. p_wrong: 확률로 틀린 토큰."""

    def __init__(self, th, exact=True, kind=None, p_wrong=0.0, seed=0):
        super().__init__(th, 4)
        if exact:
            self.layers = th.layers
        self.exact, self.kind, self.p_wrong, self.rng = exact, kind, p_wrong, np.random.default_rng(seed)

    def ids_lazy(self, x, start):
        th = self.th
        h = self._layers(x, start, 1, *th.kv)
        hn = mx.fast.rms_norm(h[-1:], th.norm, th.eps)
        if self.exact:
            lg = (hn.astype(th.dtype) @ th.E.T).astype(F32)
        else:
            wq, s, b = self.Eq
            lg = mx.quantized_matmul(hn.astype(self.dtype), wq, s, b, transpose=True, group_size=64, bits=4).astype(F32)
        t = mx.argmax(lg + th._row_add(start, [0], th.block), axis=-1).astype(mx.int32)
        if self.p_wrong and self.rng.random() < self.p_wrong:
            t = (t + 1 + int(self.rng.integers(0, 100))) % 151000
        if self.kind == "wrong":
            return (t + 7) % 151000
        if self.kind == "next":
            return mx.array([T.NEXT], dtype=mx.int32)
        if self.kind == "nonext":
            return mx.where(t == T.NEXT, mx.array([1234], dtype=mx.int32), t)
        if self.kind == "rand":
            return mx.array([int(self.rng.choice([T.NEXT, 1234, int(t.item())]))], dtype=mx.int32)
        return t


def fseg(th, K, speculative=True, cap=32, mtpc=6.0, kv=None, known=True, **kw):
    return SegmentDecoder(th, T.ADAPTER, T.PREFIX, T.SP, cap=cap, max_flush=8, kv_capacity=kv, speculative=speculative,
                          max_total_per_chunk=mtpc, known_K=K if known else None, **kw)


def frun(th, fr, K, chunks=None, **kw):
    s = fseg(th, K, **kw)
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


def chunked(K, c):
    return None if c is None else [c] * (K // c) + ([K % c] if K % c else [])


# ───────────────────────── 초안 + 라운드 끝 투기 == 순차
CFGS = [(0.6, 1, 32, 6.0, None, 37), (0.8, 2, 3, 6.0, None, 41), (0.75, 6, 32, 0.3, None, 50), (0.9, 3, 32, 6.0, 170, 37),
        (0.4, 4, 32, 6.0, None, 53), (1.0, 7, 2, 6.0, None, 23), (0.2, 8, 32, 6.0, None, 64), (0.5, 9, 32, 6.0, 120, 29)]


@pytest.mark.parametrize("p_text,seed,cap,mtpc,kv,K", CFGS)
@pytest.mark.parametrize("mode", ["exact", "wrong30", "q4", "exactL6", "nospec"])
@pytest.mark.parametrize("chunks", [None, 4])
def test_draft_equals_sequential(p_text, seed, cap, mtpc, kv, K, mode, chunks):
    fr = T.frames(K, seed)
    ch = chunked(K, chunks)
    spec = mode != "nospec"
    s1, rd1 = T.run(T.make(p_text, seed), fr, K, speculative=spec, cap=cap, mtpc=mtpc, kv=kv, chunks=ch)
    th = T.make(p_text, seed)
    d = SchedDraft(th, exact=mode != "q4", p_wrong=0.3 if mode == "wrong30" else 0.0, seed=seed)
    s2, rd2 = frun(th, fr, K, chunks=ch, speculative=spec, cap=cap, mtpc=mtpc, kv=kv, draft=d, draft_len=6 if mode == "exactL6" else 4)
    assert T.summary(s1, rd1) == T.summary(s2, rd2)
    assert s2.stats["verifies"] > 0 or not s1.out
    if mode == "exact" and p_text >= 0.6 and cap > 3 and kv is None:   # 정확한 초안: 검증마다 여러 토큰, forward 수가 준다
        assert s2.stats["accepted"] == s2.stats["drafted"] > 0 and s2.stats["forwards"] < s1.stats["forwards"]
    if mode == "exact" and spec and p_text <= 0.6 and kv is None:       # 라운드 끝 투기가 실제로 쓰인다
        assert s2.stats["res_used"] > 0


@pytest.mark.parametrize("kv", [None, 22, 23, 24, 25, 27, 30, 33, 38, 44, 51, 60, 71, 89])
@pytest.mark.parametrize("kind", [None, "wrong", "next", "nonext", "rand"])
@pytest.mark.parametrize("p_text,seed,cap,chunks", [(0.7, 11, 32, 1), (0.85, 12, 2, None), (0.5, 13, 1, 3), (0.95, 14, 32, 4)])
def test_draft_kv_boundary_sweep(kv, kind, p_text, seed, cap, chunks):
    """KV 하드 클램프 경계(prefix 19)·라운드 끝 투기 용량 검사·적대적 초안에서도 순차와 같다."""
    K = 17
    fr = T.frames(K, seed)
    ch = chunked(K, chunks)
    s1, rd1 = T.run(T.make(p_text, seed), fr, K, speculative=True, cap=cap, kv=kv, chunks=ch)
    th = T.make(p_text, seed)
    s2, rd2 = frun(th, fr, K, chunks=ch, cap=cap, kv=kv, draft=SchedDraft(th, kind=kind, seed=seed))
    assert T.summary(s1, rd1) == T.summary(s2, rd2)


def test_draft_guard_turns_off_bad_draft():
    """늘 틀리는 초안: 검증 8 번 뒤 이 세그먼트는 초안을 끄고 (q=1), id 는 그대로."""
    K, fr = 41, T.frames(41, 2)
    s1, rd1 = T.run(T.make(0.85, 2), fr, K, speculative=True)
    th = T.make(0.85, 2)
    s2, rd2 = frun(th, fr, K, draft=SchedDraft(th, kind="wrong"))
    assert T.summary(s1, rd1) == T.summary(s2, rd2)
    assert s2.stats["draft_off"] == 1 and s2.stats["accepted"] == 0 and s2.stats["verifies"] >= 8


def test_make_draft_q4_and_q8():
    th = T.make(0.5, 1, max_len=256)
    d = th.make_draft(4)
    assert isinstance(d, Draft) and d.nbytes > 0
    ids = d.ids_lazy(th.embed([T.PREFIX[0]]), 0)
    assert ids.shape == (1,) and ids.dtype == mx.int32
    th8 = MlxThinker({k: mx.array(v) for k, v in T.W_NP.items()}, T.CFG, BLOCKED_EXPECTED, dtype=mx.float16, max_len=64,
                     quant_bits=8)
    assert th8.make_draft(4) is None                         # q8: 원본 가중치가 없다 → 초안 끔


# ───────────────────────── on_token 훅 (F04)
def _hook_case(s, calls):
    log = []
    s.on_token = lambda k, t: log.append((k, t))
    for kind, arg in calls:
        n0 = len(log)
        rounds = getattr(s, kind)(*arg)
        got = log[n0:]
        assert got == [(r.k, t) for r in rounds for t in r.ids], kind     # 호출마다: 훅 == 그 호출의 Round.ids
    assert log == list(s.out)
    return log


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("cfg", [dict(), dict(cap=2), dict(mtpc=0.3), dict(kv=60), dict(kv=97), dict(kv=140), dict(known=False),
                                 dict(p=1.0, cap=2)])
def test_on_token_hook_matches_rounds(draft, speculative, cfg):
    cfg = dict(cfg)
    p = cfg.pop("p", 0.6)
    K, fr = 40, T.frames(40, 9)
    th = T.make(p, 9)
    s = fseg(th, K, speculative=speculative, draft=SchedDraft(th, p_wrong=0.2) if draft else None, **cfg)
    calls = [("feed_frames", (fr[i:i + 3],)) for i in range(0, K, 3)] + [("finish_frames", (K,))]
    _hook_case(s, calls)
    s1, _ = T.run(T.make(p, 9), fr, K, speculative=speculative, **cfg)
    assert s.out == s1.out


def test_on_token_hook_pcm_path_and_engine_kwarg():
    n_real = 41000
    K = k_of(n_real + 16000)
    fr = T.frames(K + 4, 13)
    th = T.make(0.4, 13)
    s = SegmentDecoder(th, T.ADAPTER, T.PREFIX, T.SP, cap=32, max_flush=8, stream=T.FakeStream(fr), draft=SchedDraft(th))
    calls = [("feed", (np.zeros(min(1234, n_real - i), np.float32),)) for i in range(0, n_real, 1234)] + [("finish", (16000,))]
    log = _hook_case(s, calls)
    assert log and s.stats["on_token_errors"] == 0


def test_on_token_hook_errors_do_not_break_decode():
    K, fr = 29, T.frames(29, 5)
    s1, rd1 = T.run(T.make(0.7, 5), fr, K, speculative=True)
    s2 = fseg(T.make(0.7, 5), K, on_token=lambda k, t: 1 / 0)
    rd2 = s2.feed_frames(fr) + s2.finish_frames(K)
    assert T.summary(s1, rd1) == T.summary(s2, rd2) and s2.stats["on_token_errors"] == len(s2.out) > 0


# ───────────────────────── kv_full 뒤 인코더 미루기 (F17)
class CountStream(T.FakeStream):
    """FakeStream + 호출 기록 + 호출마다 sleep (인코더 시간 흉내)."""

    def __init__(self, fr, sleep=0.002):
        super().__init__(fr)
        self.calls, self.sleep, self.slept = [], sleep, 0.0

    def feed(self, pcm):
        self.calls.append(("feed", len(pcm)))
        time.sleep(self.sleep)
        self.slept += self.sleep
        return super().feed(pcm)

    def finish(self, zt):
        self.calls.append(("finish", zt))
        time.sleep(self.sleep)
        self.slept += self.sleep
        return super().finish(zt)


@pytest.mark.parametrize("draft", [False, True])
def test_kv_full_defers_encoder_and_final_is_unchanged(draft):
    n_real = 6 * 16000
    K = k_of(n_real + 16000)
    fr = T.frames(K + 4, 21)
    blk = T._offline_blk()

    def make(kv):
        th = T.make(0.8, 21)
        st = CountStream(fr)
        s = SegmentDecoder(th, T.ADAPTER, T.PREFIX, T.SP, cap=32, max_flush=8, stream=st, kv_capacity=kv, tokens=T._TT(),
                           mode=("English", None), offline_block=mx.array(blk), draft=SchedDraft(th) if draft else None)
        return s, st

    s, st = make(len(T.PREFIX) + 60)
    rounds, n_at_full, slept_at_full = [], None, None
    for i in range(0, n_real, 2560):
        rounds += s.feed(np.zeros(min(2560, n_real - i), np.float32))
        if s.kv_full and n_at_full is None:
            n_at_full, slept_at_full = len(st.calls), st.slept
    assert s.kv_full and rounds[-1].kv_full
    assert not s._enc_at and s._enc_carry == 0.0                                 # 귀속 안 된 인코더 시간이 남지 않는다
    assert sum(r.enc_ms for r in rounds) >= slept_at_full * 1000 * 0.95           # 그때까지 쓴 인코더 시간은 모두 Round 에
    rounds += s.finish(16000)
    assert len(st.calls) == n_at_full and s.stats["deferred_samples"] > 0       # kv_full 뒤엔 인코더를 부르지 않았다
    assert s.K == K and s.done
    f = np.array(s.features())                                                  # final 이 필요할 때 미룬 PCM 을 인코딩
    assert f.shape == (K, T.DIN) and np.array_equal(f, np.array(fr[:K]))
    assert st.calls[-1][0] == "finish" and len(st.calls) > n_at_full
    ids, _ = s.final_redecode()
    ref, _ = T._final_seg(T.make(0.8, 21), K, fr[:K], blk).final_redecode()     # 같은 특징·K 의 보통 세그먼트와 같은 final
    assert ids == ref


# ───────────────────────── prefix KV 캐시 (F21)
def test_prefix_cache_restores_identical_and_copies():
    K = 29
    fr = T.frames(K, 3)
    s1, rd1 = T.run(T.make(0.5, 3), fr, K, speculative=True)
    th = T.make(0.5, 3)
    pc = PrefixCache(th, size=2)
    a, _ = frun(th, T.frames(K, 9), K, prefix_cache=pc)                         # 캐시를 채우고 KV 를 다른 오디오로 더럽힌다
    assert not a.stats["prefix_cached"] and len(pc) == 1 and pc.misses == 1
    ks, vs = pc.get(tuple(T.PREFIX))
    P = len(T.PREFIX)
    assert ks[0].shape == (1, T.KVH, P, T.HD) and ks[0].nbytes == P * T.KVH * T.HD * 4
    Kc, Vc = th.kv
    for i in range(len(Kc)):                                                    # 살아 있는 KV 의 prefix 행을 지워도
        Kc[i][..., :P, :] = 0.0
        Vc[i][..., :P, :] = 0.0
    mx.eval(Kc, Vc)
    assert float(mx.abs(ks[0]).sum().item()) > 0                                # 캐시는 깊은 복사라 그대로
    s2, rd2 = frun(th, fr, K, prefix_cache=pc)
    assert s2.stats["prefix_cached"] and pc.hits >= 2
    assert T.summary(s1, rd1) == T.summary(s2, rd2)
    assert rd2[0].dec_ms >= s2.stats["prefix_ms"]                               # prefix 시간은 첫 Round 에 귀속
    other = MlxThinker({k: mx.array(v) for k, v in T.W_NP.items()}, T.CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=256)
    s3 = SegmentDecoder(other, T.ADAPTER, T.PREFIX, T.SP, cap=32, max_flush=8, prefix_cache=pc)
    assert not s3.stats["prefix_cached"]                                        # 다른 thinker 의 캐시는 쓰지 않는다
    for i in range(3):
        pc.put((i,), None)
    assert len(pc) == 2 and tuple(T.PREFIX) not in pc._d                        # LRU


# ───────────────────────── final lookahead (F19)
class StopThinker(T.SchedThinker):
    """절대 위치 stop_at 에서 <|im_end|> logit 에 +1e4."""
    stop_at = None

    def _row_add(self, start, sel, block):
        base = super()._row_add(start, sel, block)
        if self.stop_at is None:
            return base
        pos = np.array([start + s for s in sel])
        add = np.zeros((len(sel), T.V), np.float32)
        add[pos == self.stop_at, T.SP.im_end] = 1e4
        return base + mx.array(add)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("stop_rel", [None, 0, 1, 3, 9])
@pytest.mark.parametrize("max_new", [None, 1, 2, 5])
def test_final_lookahead_equals_q1(dtype, stop_rel, max_new):
    blk = T._offline_blk()
    for seed in (0, 3):
        K = 12 + 5 * seed
        fr = T.frames(K, 100 + seed)
        th = StopThinker(np.where(np.random.default_rng(seed).random(4096) < 0.5, -1e4, 1e4), dtype=dtype, max_len=4096)
        th.stop_at = None if stop_rel is None else K + 20 - 1 + stop_rel
        s = T._final_seg(th, K, fr, blk)
        ref_job = s.final_job(max_new=max_new, lookahead=0)
        ref, _ = ref_job.run()
        drafts = {"stream": None, "oracle": [*ref, T.SP.im_end], "noisy": [t if i % 7 else 12345 for i, t in enumerate(ref)],
                  "shifted": ref[1:] + [T.SP.im_end], "empty": []}
        for dk, d in drafts.items():
            for Lf in (1, 4, 11, 15):
                job = s.final_job(max_new=max_new, lookahead=Lf)
                if d is not None:
                    job.draft = list(d)
                n = 0
                while not job.step():
                    n += 1
                    assert n < 300
                assert job.ids == ref, (dk, Lf)
                assert job.steps <= ref_job.steps
                if dk == "oracle" and len(ref) > 2 * Lf + 2:
                    assert job.steps < ref_job.steps and job.accepted > 0


def test_final_lookahead_default_and_stream_draft():
    K, fr, blk = 16, T.frames(16, 15), T._offline_blk()
    th = T.make(0.3, 16)
    s = SegmentDecoder(th, T.ADAPTER, T.PREFIX, T.SP, cap=32, max_flush=8, known_K=K, tokens=T._TT(), mode=("English", "German"),
                       offline_block=mx.array(blk), final_lookahead=11)
    s.feed_frames(fr)
    s.finish_frames(K)
    d = stream_draft(s)
    assert d[0] in (T.SP.spk_a, T.SP.spk_b) and d[-1] == T.SP.im_end and T.SP.trans in d
    job = s.final_job()
    assert isinstance(job, FinalJob) and job.lookahead == 11
    ids, _ = job.run()
    assert ids == s.final_job(lookahead=0).run()[0]


# ───────────────────────── fp32 두 단계 lm_head (F20)
def _ts_thinker(p, seed, tau=1e-3):
    th = T.make(p, seed)
    assert th.enable_two_stage(tau=tau) and isinstance(th.ts, TwoStageHead)
    return th


@pytest.mark.parametrize("p_text,seed", [(0.6, 1), (0.3, 2), (0.9, 5)])
def test_two_stage_head_equals_full_head(p_text, seed):
    K = 41
    fr = T.frames(K, seed)
    s1, rd1 = T.run(T.make(p_text, seed), fr, K, speculative=True)
    th = _ts_thinker(p_text, seed)
    s2, rd2 = T.run(th, fr, K, speculative=True)
    assert T.summary(s1, rd1) == T.summary(s2, rd2)
    assert th.ts_stats["rows"] > 0


@pytest.mark.parametrize("tau", [1e9, 0.3])
def test_two_stage_fallback_rows_identical(tau):
    """인증 실패(τ 를 키워 강제) 행은 현재 헤드를 같은 행 전부로 다시 계산한 값 — 일부만 실패해도 id 가 같다."""
    K = 41
    fr = T.frames(K, 1)
    s1, rd1 = T.run(T.make(0.6, 1), fr, K, speculative=True)
    th = _ts_thinker(0.6, 1, tau=tau)
    s2, rd2 = T.run(th, fr, K, speculative=True)
    assert T.summary(s1, rd1) == T.summary(s2, rd2)
    st = th.ts_stats
    if tau > 1e8:
        assert st["fallback_rows"] == st["rows"] > 0
    else:
        assert 0 < st["fallback_rows"] < st["rows"]


def test_two_stage_multi_row_and_nonfinite():
    th32 = MlxThinker({k: mx.array(v) for k, v in T.W_NP.items()}, T.CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=64)
    ts = MlxThinker({k: mx.array(v) for k, v in T.W_NP.items()}, T.CFG, BLOCKED_EXPECTED, dtype=mx.float32, max_len=64)
    assert ts.enable_two_stage()
    X = mx.array((np.random.default_rng(9).standard_normal((12, T.D)) * 0.5).astype(np.float32))
    assert th32.forward(X, 0, sel=list(range(12))) == ts.forward(X, 0, sel=list(range(12)))
    assert th32.forward(X[:5], 3, sel=[0, 2, 4]) == ts.forward(X[:5], 3, sel=[0, 2, 4])
    ts.forward(mx.array(np.full((2, T.D), np.inf, np.float32)), 30)
    assert ts.nonfinite == 1
    th16 = MlxThinker({k: mx.array(v) for k, v in T.W_NP.items()}, T.CFG, BLOCKED_EXPECTED, dtype=mx.float16, max_len=64)
    assert not th16.enable_two_stage() and th16.ts is None                      # fp32 전용
