"""engine/fake.py 테스트: Engine API·Round 필드·청크 일정·K/tail/flush·cap·KV 클램프, 그리고 파이프라인과의 작업자 모의 실행."""
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio_pipeline as ap  # noqa: E402
from engine.fake import FakeEngine, Round, prefix_ids, SPK_A, TRANS, SEM_END, UNUSED  # noqa: E402
from text_stream import TextAssembler  # noqa: E402

SR = 16000


def noise(n, db, seed=0):
    x = np.random.default_rng(seed).standard_normal(n)
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)).astype(np.float32)


def speech(sec=4.0, seed=0):
    """−25 dB 발화 / −70 dB 틈이 섞인 신호."""
    rng = np.random.default_rng(seed); parts = []
    while sum(map(len, parts)) < sec * SR:
        parts += [noise(int(rng.integers(3000, 20000)), -25, int(rng.integers(1e6))), noise(int(rng.integers(500, 4000)), -70, 1)]
    return np.concatenate(parts)[:int(sec * SR)]


def decode(eng, src, tgt, x, tail=SR, pkts=(320,), delay=6, seed=0):
    seg = eng.new_segment(src, tgt, delay); rng = np.random.default_rng(seed)
    rounds, fed_at, i = [], [], 0
    while i < len(x):
        n = int(rng.choice(pkts)); r = seg.feed(x[i:i + n]); i += n
        rounds += r; fed_at += [seg.samples_fed] * len(r)
    fin = seg.finish(tail)
    return seg, rounds, fed_at, fin


def text(eng, rounds, keep_tr=True):
    ta = TextAssembler(eng.token_bytes, eng.spk_ids, eng.trans_id, keep_tr=keep_tr)
    ds, dt = zip(*[ta.push(r.ids) for r in rounds]) if rounds else ((), ())
    fs, ft = ta.flush(); s, t, st = ta.finish()
    assert s == "".join(ds) + fs and t == "".join(dt) + ft
    return s, t, st, list(ds) + [fs]


# ---------------------------------------------------------------- API
def test_api_surface_and_prefixes():
    eng = FakeEngine()
    assert eng.langs == ["English", "Korean", "German"] and eng.lang_labels["Korean"] == "한국어" and ("Korean", "English") in eng.pairs and eng.delays == list(range(4, 11))
    assert eng.defaults["asr_delay"] == 6 and eng.defaults["translate_delay"]["Korean>English"] == 8
    assert eng.spk_ids == (151707, 151708) and eng.trans_id == 151726
    assert {"name", "precision", "kv_capacity", "load_ms", "speculative", "k_cap"} <= set(eng.info)
    assert eng.k_cap == 388 and FakeEngine(hard_max_s=25).k_cap == 325                   # K_cap = round((hard_max+tail)·12.5)
    assert "warmup_ms" in eng.warmup()
    lens = {("English", None): 14, ("Korean", None): 15, ("German", None): 14, ("English", "German"): 19,
            ("English", "Korean"): 20, ("German", "English"): 19, ("Korean", "English"): 20}
    for (s, t), n in lens.items():
        seg = eng.new_segment(s, t, 6)
        assert seg.positions == n and seg.K is None and seg.samples_fed == 0 and seg.kv_capacity == 2048
    assert prefix_ids("English", "German", 6) == [151644, 8948, 198, 151645, 198, 151644, 77091, 198, 11528, 220, 151707,
                                                   22574, 198, 14045, 220, 151707, 32079, 151704, 151714]   # decode-contract §b.3
    for bad in (("German", "Korean", 6), ("English", None, 3), ("French", None, 6)):
        with pytest.raises(ValueError): eng.new_segment(*bad)
    for i in (SPK_A, TRANS, SEM_END, UNUSED, 151705, 151935, 5): assert eng.token_bytes(i) == b""


def test_from_spec():
    eng = FakeEngine.from_spec("round_ms=12.5, hangul=1, words_every=3, kv_capacity=999")
    assert eng.round_ms == 12.5 and eng.hangul and eng.words_every == 3 and eng.kv_capacity == 999
    with pytest.raises(ValueError): FakeEngine.from_spec("nope=1")


# ---------------------------------------------------------------- 청크 일정·Round 필드
def test_chunk_schedule_and_round_fields():
    eng = FakeEngine(); x = speech(6.0)
    seg, rounds, fed_at, fin = decode(eng, "English", None, x, pkts=(1, 160, 333, 960, 5000), seed=1)
    assert [r.k for r in rounds] == list(range(len(rounds)))
    prev_fed = 0
    for r, fed in zip(rounds, fed_at):
        assert isinstance(r, Round) and not r.is_tail and not r.is_flush and not r.kv_full
        assert r.need_samples == 4096 + 5120 * (r.k // 4) <= fed                       # 청크가 완성된 뒤에만
        if r.k % 4: assert r.enc_ms == 0.0                                            # 인코더 시간은 청크 첫 라운드에만
        assert len(r.ids) <= 32 and r.dec_ms >= 0
    # 청크 c 의 라운드는 샘플이 4096 + 5120c 에 처음 닿은 feed 에서 나온다
    firsts = [(r.need_samples, f) for r, f in zip(rounds, fed_at) if r.k % 4 == 0]
    assert all(f - n < 5000 for n, f in firsts)


def test_tokens_independent_of_packetization():
    eng = FakeEngine(); x = speech(7.0, seed=3)
    outs = []
    for pk in ((320,), (5120,), (1, 777, 4000), (len(x),)):
        seg, rounds, _, fin = decode(eng, "English", "German", x, pkts=pk, seed=2)
        outs.append([(r.k, tuple(r.ids), r.forced, r.is_tail, r.is_flush, r.need_samples) for r in rounds + fin])
    assert all(o == outs[0] for o in outs)


@pytest.mark.parametrize("n_real,tail", [(69760 - 16000, 16000), (5120 * 10 + 4096 + 100 - 16000, 16000), (40000, 3200),
                                         (5120 * 7 + 4096 + 383 - 3200, 3200), (100003, 11904), (8000, 3200)])
def test_finish_K_tail_flush(n_real, tail):
    eng = FakeEngine(); x = speech(n_real / SR, seed=n_real % 97)
    seg, rounds, _, fin = decode(eng, "English", "German", x, tail=tail)
    n_total = n_real + tail
    K = int(round(n_total / 16000 * 12.5))
    assert seg.K == K and seg.samples_fed == n_total
    if n_total == 69760: assert K == 55                                               # round(n/1280) 은 54
    audio = [r for r in rounds + fin if not r.is_flush]; flush = [r for r in fin if r.is_flush]
    assert [r.k for r in audio] == list(range(K))                                    # k < K 만, 빠짐없이
    assert [r.k for r in flush] == list(range(K, K + len(flush))) and 1 <= len(flush) <= 8
    assert not flush[-1].ids or len(flush) == 8                                      # 첫 빈 라운드에서 멈춤
    assert all(r.ids for r in flush[:-1])
    for r in fin:
        assert r.is_tail == (r.is_flush or r.k * 1280 >= n_real)
        assert r.need_samples == (n_total if r.is_flush else min(4096 + 5120 * (r.k // 4), n_total))
    assert seg.flush_rounds == len(flush)


def test_short_tail_is_raised_to_min_tail():
    """[critic] K 경계 길이(n − 4096) mod 5120 < 384: feed 는 k < K(n + MIN_TAIL) 까지만 디코드하고, finish 는 꼬리를
    MIN_TAIL(1664) 이상으로 올린다(decoder.py 와 같다) → 어떤 경우에도 k ≥ K 오디오 라운드가 없다."""
    eng = FakeEngine(); n = 5120 * 10 + 4096 + 100
    seg = eng.new_segment("English", None, 6); r = seg.feed(speech(n / SR))
    assert [q.k for q in r] == list(range(44))
    f = seg.finish(0)
    assert seg.stats["tail_raised"] == 1664 and seg.stats["overrun"] == 0 and seg.samples_fed == n + 1664
    assert [q.k for q in r + f if not q.is_flush] == list(range(seg.K)) and seg.K == 45
    with pytest.raises(RuntimeError): seg.feed(np.zeros(10, np.float32))              # 끝난 세그먼트


# ---------------------------------------------------------------- 텍스트·태그
def test_asr_and_translate_text():
    eng = FakeEngine(); x = speech(6.0, seed=4)
    seg, r, _, f = decode(eng, "English", None, x)
    s, t, st, _ = text(eng, r + f)
    assert s.startswith("alpha bravo") and t == "" and seg.flush_rounds == 1 and TRANS not in [i for q in r + f for i in q.ids]
    seg, r, _, f = decode(eng, "English", "German", x)
    s2, t2, _, _ = text(eng, r + f)
    assert s2 == s and t2.startswith("Eins Zwei") and t2.endswith(".") and seg.flush_rounds == 2


def test_asr_mode_injected_trans_is_dropped():
    eng = FakeEngine(asr_trans=True); x = speech(6.0, seed=5)
    seg, r, _, f = decode(eng, "English", None, x)
    assert TRANS in [i for q in r + f for i in q.ids]
    s, t, st, _ = text(eng, r + f, keep_tr=False)
    assert t == "" and st["tr_dropped"] > 0 and s.startswith("alpha")               # E7


def test_specials_render_nothing():
    x = speech(6.0, seed=6)
    a = text(FakeEngine(), sum(decode(FakeEngine(), "German", None, x)[1::2], []))[0]
    eng = FakeEngine(specials=True); seg, r, _, f = decode(eng, "German", None, x)
    ids = [i for q in r + f for i in q.ids]
    assert SEM_END in ids and UNUSED in ids and text(eng, r + f)[0] == a


def test_hangul_split_tokens_never_show_broken_chars():
    eng = FakeEngine(hangul=True); x = speech(8.0, seed=7)
    seg, r, _, f = decode(eng, "Korean", "English", x)
    pieces = [eng.token_bytes(i) for q in r for i in q.ids]
    bad = 0
    for p in pieces:
        try: p.decode("utf-8")
        except UnicodeDecodeError: bad += 1
    assert bad > 0                                                                    # 조각이 실제로 글자를 쪼갠다
    s, t, st, ds = text(eng, r + f)
    assert all("\ufffd" not in d for d in ds) and st["replaced_bytes"] == 0 and s.startswith("하나 둘 셋")


# ---------------------------------------------------------------- cap · KV
def test_round_and_total_caps():
    eng = FakeEngine(runaway=True, kv_capacity=100000, k_cap=5)                       # 스트리밍 총량 6·5+64 = 94
    seg, r, _, f = decode(eng, "English", None, speech(3.0))
    assert all(len(q.ids) <= 32 for q in r + f) and r[0].forced and len(r[0].ids) == 32
    assert sum(len(q.ids) for q in r) == 94                                           # 스트리밍: 6·K_cap+64 에서 멈춤
    assert sum(len(q.ids) for q in r + f) == 6 * seg.K + 64                           # finish 안에서는 정확한 6K+64 까지
    assert f[-1].is_flush and f[-1].ids == [] and seg.flush_rounds == 1               # cap 뒤 flush 는 1 라운드로 끝
    # 기본 K_cap(325): 스트리밍 중에는 6K+64 보다 더 낼 수 있다(비인과 cap 의 알려진 차이, runaway 에서만). finish 는 더 내지 않는다
    eng = FakeEngine(runaway=True, kv_capacity=100000)
    seg, r, _, f = decode(eng, "English", None, speech(2.0))
    assert sum(len(q.ids) for q in r) > 6 * seg.K + 64 and sum(len(q.ids) for q in f) == 0


def test_kv_clamp():
    eng = FakeEngine(runaway=True, kv_capacity=400); x = speech(5.0)
    seg = eng.new_segment("English", "German", 6)
    r = seg.feed(x[:SR * 3])
    assert r[-1].kv_full and r[-1].forced and not any(q.kv_full for q in r[:-1])
    assert seg.positions <= 400 and seg.kv_full
    assert seg.feed(x[SR * 3:]) == [] and seg.finish(SR) == [] and seg.K is not None


def test_round_ms_sleep_and_runtime_change():
    eng = FakeEngine(round_ms=5.0); seg = eng.new_segment("English", None, 6)
    r = seg.feed(speech(0.6))
    assert r and all(q.dec_ms >= 4.5 for q in r)
    eng.round_ms = 0.0
    r = seg.feed(speech(0.6, seed=9))
    assert r and all(q.dec_ms < 4.0 for q in r)


def test_thread_check():
    eng = FakeEngine(); seg = eng.new_segment("English", None, 6); err = []
    def other():
        try: seg.feed(np.zeros(5000, np.float32))
        except RuntimeError as e: err.append(e)
        err.append(eng.token_bytes(1000))                                             # token_bytes 는 어느 스레드에서나
    th = threading.Thread(target=other); th.start(); th.join()
    assert isinstance(err[0], RuntimeError) and err[1] == b" alpha"
    eng2 = FakeEngine(check_thread=False); seg2 = eng2.new_segment("English", None, 6)
    th = threading.Thread(target=lambda: seg2.feed(np.zeros(5000, np.float32))); th.start(); th.join()


def test_final_redecode():
    eng = FakeEngine(); x = speech(6.0, seed=8)
    seg, r, _, f = decode(eng, "Korean", "English", x)
    with pytest.raises(RuntimeError): eng.new_segment("English", None, 6).final_redecode()   # finish 전
    ids, ms = seg.final_redecode()
    assert ids[0] == SPK_A and TRANS in ids and ms >= 0
    s, t, _, _ = text(eng, r + f)
    fs, ft, _, _ = text(eng, [Round(0, ids, False, False, False, 0, 0.0, 0.0)])
    assert fs == s and ft.endswith(".") and len(ft.split()) == len(t.split())


# ---------------------------------------------------------------- 파이프라인 + 가짜 엔진 (작업자 모의)
def worker_sim(eng, cmds_iter, pipe, src="English", tgt=None, delay=6):
    """server 작업자 루프와 같은 순서: seg_start → new_segment+feed, audio → feed, seg_end → finish(zero_tail)."""
    segs, cur = [], None
    for c in cmds_iter:
        if c.kind == "seg_start":
            cur = dict(seg=c.seg, dec=eng.new_segment(src, tgt, delay), rounds=[], n=len(c.pcm), info=c.info)
            cur["rounds"] += cur["dec"].feed(c.pcm)
        elif c.kind == "audio":
            cur["rounds"] += cur["dec"].feed(c.pcm); cur["n"] += len(c.pcm)
        else:
            cur["rounds"] += cur["dec"].finish(c.info["zero_tail_samples"])
            cur["end"] = c.info; segs.append(cur)
        if cur is not None and cur["rounds"] and cur["rounds"][-1].kv_full and c.kind != "seg_end" and pipe is not None:
            yield ("kv_full", cur["seg"])
    return segs


def run_sim(x, in_rate=SR, params=None, eng=None, pkt=320, **kw):
    eng = eng or FakeEngine(); pipe = ap.AudioPipeline(in_rate, params or ap.SegParams())
    pending, segs = [], []
    def cmds():
        for i in range(0, len(x), pkt):
            yield from pipe.push(x[i:i + pkt])
            while pending: yield from pipe.force_end("kv_guard", seg=pending.pop())
        yield from pipe.stop("file_end")
    gen = worker_sim(eng, cmds(), pipe, **kw)
    try:
        while True:
            ev = next(gen)
            if ev[0] == "kv_full": pending.append(ev[1])
    except StopIteration as e:
        segs = e.value
    return pipe, segs


@pytest.mark.parametrize("seed", range(8))
def test_pipeline_to_fake_engine_end_to_end(seed):
    rng = np.random.default_rng(seed)
    parts = [noise(int(rng.integers(2000, 20000)), -70, seed)]
    for i in range(6):
        parts += [noise(int(rng.integers(8000, 120000)), float(rng.uniform(-40, -20)), seed * 10 + i),
                  noise(int(rng.integers(2000, 30000)), -70, seed * 10 + i + 5)]
    x = np.concatenate(parts)
    pipe, segs = run_sim(x, pkt=int(rng.integers(100, 6000)), tgt="German" if seed % 2 else None)
    assert segs and pipe.counters()["samples_held"] == 0
    for s in segs:
        d = s["dec"]; z = s["end"]["zero_tail_samples"]
        assert z >= 3200 and d.samples_fed == s["n"] + z
        assert d.K == int(round(d.samples_fed / 16000 * 12.5))                       # 세그먼트 자신의 표본열로 K
        ks = [r.k for r in s["rounds"] if not r.is_flush]
        assert ks == list(range(d.K))                                                # 끝 결정 전에 디코드한 프레임도 모두 < K
        txt, _, _, _ = text(FakeEngine(), s["rounds"])


def test_end_to_end_48k_and_kv_guard():
    x48 = np.concatenate([noise(48000, -70), noise(48000 * 9, -25, 1), noise(48000, -70, 2)])
    eng = FakeEngine(runaway=True, kv_capacity=600)
    pipe, segs = run_sim(x48, in_rate=48000, eng=eng, pkt=960)
    reasons = [s["end"]["reason"] for s in segs]
    assert "kv_guard" in reasons and reasons[-1] in ("pause", "file_end")
    kv = [s for s in segs if s["end"]["reason"] == "kv_guard"]
    assert all(any(r.kv_full for r in s["rounds"]) for s in kv)
    c = pipe.counters()
    assert c["samples_rx"] == 16000 * 11 and c["samples_held"] == 0
    assert c["samples_rx"] == c["samples_seg"] + c["samples_skipped"]


# ---------------------------------------------------------------- 실제 엔진과의 API 일치 (decoder.py·tokens.py 가 있을 때)
def test_matches_real_engine_api():
    pytest.importorskip("mlx.core")
    import dataclasses
    import inspect
    try:
        import engine as real
        from engine import decoder as rdec, tokens as rtok
    except ImportError as e:
        pytest.skip(f"실제 엔진 모듈 없음: {e}")
    from engine import fake
    assert [f.name for f in dataclasses.fields(fake.Round)] == [f.name for f in dataclasses.fields(rdec.Round)]
    assert fake.MIN_TAIL == rdec.MIN_TAIL
    assert fake.LANGS == list(rtok.LANGS) and fake.PAIRS == list(rtok.PAIRS) and fake.DELAYS == list(rtok.DELAYS)
    assert fake.DEFAULTS == rtok.DEFAULTS and fake.LANG_LABELS == dict(rtok.LANGS)
    pub = lambda c: {n for n, _ in inspect.getmembers(c) if not n.startswith("_")}
    assert {"warmup", "new_segment", "token_bytes", "close"} <= pub(real.Engine) & pub(fake.FakeEngine)
    assert {"feed", "finish", "final_redecode", "positions"} <= pub(rdec.SegmentDecoder) & pub(fake.FakeSegmentDecoder)
    sig = lambda f: [p for p in inspect.signature(f).parameters]
    assert sig(real.Engine.new_segment) == sig(fake.FakeEngine.new_segment)
    assert sig(real.Engine.__init__)[:6] == sig(fake.FakeEngine.__init__)[:6]
    for m in ("feed", "finish", "final_redecode"):
        assert sig(getattr(rdec.SegmentDecoder, m)) == sig(getattr(fake.FakeSegmentDecoder, m))
    for s, t in [(s, None) for s in fake.LANGS] + fake.PAIRS:                  # prefix 표 = decode-contract §b.2
        for d in fake.DELAYS:
            assert fake.prefix_ids(s, t, d)[-1] == rtok.SP_EXPECTED[f"<DELAY_{d}>"]


def test_prefix_ids_match_real_tokenizer():
    tokp = os.path.expanduser("~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/tokenizer.json")
    if not os.path.exists(tokp) or os.stat(tokp).st_flags & 0x40000000: pytest.skip("s4 tokenizer.json 없음 또는 dataless")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokp)
    from engine import fake
    for s, t in [(s, None) for s in fake.LANGS] + fake.PAIRS:
        head = f"language <SPK_A>{s}" + (f"\ntranslate <SPK_A>{t}" if t else "") + "<asr_text>"
        for d in (4, 6, 10):
            ref = fake.HEADER + tok.encode(head, add_special_tokens=False).ids + [tok.token_to_id(f"<DELAY_{d}>")]
            assert fake.prefix_ids(s, t, d) == ref
