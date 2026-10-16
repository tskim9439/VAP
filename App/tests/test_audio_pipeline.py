"""audio_pipeline 단위 테스트 (audio-ui-spec §1.3–1.5, §3.2, 수용 항목 B3/B5/C1–C9 + 실제 방 조건 F06/F07/F08/F31/F33). 합성 신호만, 모델 없음."""
import glob
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio_pipeline as ap  # noqa: E402

SR, F = 16000, 512
TESTAUDIO = "/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad/testaudio"


# ---------------------------------------------------------------- 신호·실행 도우미
def noise(n, db, seed=0):
    x = np.random.default_rng(seed).standard_normal(n)
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)).astype(np.float32)


def layout(parts, seed=0):
    """[(프레임 수, dB), ...] → 프레임 정렬 신호 (구간마다 다른 시드의 백색 잡음)."""
    return np.concatenate([noise(n * F, db, seed + i) for i, (n, db) in enumerate(parts)])


def run(x, pkt=320, params=None, in_rate=SR, feature_fn=None, reason="file_end"):
    p = ap.AudioPipeline(in_rate, params or ap.SegParams(), feature_fn=feature_fn)
    cmds, per_push = [], []
    for i in range(0, len(x), pkt):
        c = p.push(x[i:i + pkt]); cmds += c; per_push.append(c)
    c = p.stop(reason); cmds += c; per_push.append(c)
    return p, cmds, per_push


def segments(cmds):
    """Cmd 열 → 세그먼트 목록. 순서·연속성 검사를 겸한다."""
    out, cur = [], None
    for c in cmds:
        if c.kind == "seg_start":
            assert cur is None, "seg_start 가 seg_end 없이 겹침"
            assert c.seg == len(out) + 1
            lead = c.info["lead_in_samples"]; real = len(c.pcm) - lead
            cur = dict(seg=c.seg, start=c.info["s_start"], end=c.info["s_start"] + real, info=c.info, lead=c.pcm[:lead],
                       real=[c.pcm[lead:]], audio_cmds=[])
        elif c.kind == "audio":
            assert cur is not None and c.seg == cur["seg"]
            assert c.info["s0"] == cur["end"] and c.info["s1"] - c.info["s0"] == len(c.pcm) > 0, "오디오가 연속이 아님"
            cur["end"] = c.info["s1"]; cur["real"].append(c.pcm); cur["audio_cmds"].append(c)
        else:
            assert c.kind == "seg_end" and cur is not None and c.seg == cur["seg"]
            assert c.info["s_end"] == cur["end"], "seg_end 위치가 보낸 오디오 끝과 다름"
            cur["end_info"] = c.info; cur["pcm"] = np.concatenate(cur["real"]); out.append(cur); cur = None
    assert cur is None, "닫히지 않은 세그먼트"
    return out


def check_accounting(p, segs, x16=None, exact=False):
    """불변식: 겹침 없음, 받은 샘플 = 세그먼트 + 건너뜀 (+ 보류 0), 내용 = 입력(이득 0 dB 이면 비트 동일)."""
    c = p.counters()
    prev = 0
    for s in segs:
        assert s["start"] >= prev, "세그먼트 겹침"
        prev = s["end"]
        if x16 is not None:
            ref = x16[s["start"]:s["end"]]
            if s["info"]["gain_db"] == 0 and exact: assert np.array_equal(s["pcm"], ref)
            elif s["info"]["gain_db"] == 0: assert np.allclose(s["pcm"], ref)
    assert sum(s["end"] - s["start"] for s in segs) == c["samples_seg"]
    assert c["samples_rx"] == c["samples_seg"] + c["samples_skipped"] + c["samples_held"]
    if p.stopped: assert c["samples_held"] == 0
    return c


def brief(segs):
    return [(round(s["start"] / SR, 3), round(s["end"] / SR, 3), s["info"]["reason"], s["end_info"]["reason"],
             s["end_info"]["zero_tail_samples"]) for s in segs]


# ---------------------------------------------------------------- 리샘플러 (B3, B5)
@pytest.mark.parametrize("sr", [48000, 44100])
def test_resampler_impulse_alias_passband(sr):
    pk = sr // 50
    r = ap.Resampler(sr); x = np.zeros(2 * sr, np.float32); x[sr // 2] = 1.0
    y = np.concatenate([r.push(x[i:i + pk]) for i in range(0, len(x), pk)] + [r.flush()])
    assert int(np.argmax(np.abs(y))) == 8000                       # 지연 보정: 0.5 s 의 임펄스가 정확히 8000 에
    t = np.arange(sr) / sr
    for f, check in ((12000, lambda g: g < -100.0), (7500, lambda g: g >= -0.5)):
        r = ap.Resampler(sr); tone = (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        y = np.concatenate([r.push(tone[i:i + pk]) for i in range(0, len(tone), pk)])
        rms = np.sqrt(np.mean(y[1000:-1000].astype(np.float64) ** 2))
        g = 20 * math.log10(rms + 1e-15) if f == 12000 else 20 * math.log10(rms / (0.5 / math.sqrt(2)))
        assert check(g), (f, g)


@pytest.mark.parametrize("n", [96000, 96001, 96002, 48000 * 7 + 13])
def test_resampler_flush_no_lost_tail_48k(n):
    p, cmds, _ = run(noise(n, -30), pkt=960, in_rate=48000, params=ap.SegParams(segmentation="none", gain="off"))
    c = p.counters()
    assert c["samples_rx"] == round(n / 3)                          # B5: stop 뒤 정확히 round(N/3)
    assert p.res.done and c["samples_held"] == 0
    segs = segments(cmds); assert len(segs) == 1 and segs[0]["end"] == c["samples_rx"]


def test_resampler_44k_flush_length():
    n = 44100 * 3 + 77
    p, _, _ = run(noise(n, -30), pkt=882, in_rate=44100, params=ap.SegParams(segmentation="none", gain="off"))
    assert abs(p.counters()["samples_rx"] - n * 16000 / 44100) <= 1


def test_bypass_is_bit_exact_and_none_mode_tail():
    x = noise(16000 * 3 + 123, -20)
    p, cmds, _ = run(x, pkt=5120, params=ap.SegParams(segmentation="none", gain="auto"))   # none ⇒ 이득 강제 off
    segs = segments(cmds)
    assert len(segs) == 1 and p.resampler == "bypass"
    s = segs[0]
    assert s["start"] == 0 and s["end"] == len(x) and np.array_equal(s["pcm"], x)
    assert s["info"]["reason"] == "none" and s["info"]["gain_db"] == 0 and s["info"]["lead_in_ms"] == 0
    assert s["end_info"]["zero_tail_samples"] == 16000 and s["end_info"]["reason"] == "file_end"


def test_none_mode_refuses_over_30s():
    p = ap.AudioPipeline(SR, ap.SegParams(segmentation="none"))
    p.push(np.zeros(30 * SR, np.float32))                         # 정확히 30 s 는 허용
    with pytest.raises(ap.InputTooLong):
        p.push(np.zeros(1, np.float32))


# ---------------------------------------------------------------- 분할 (C1, C3, C5)
def test_onset_preroll_end_pause_and_tail():
    x = layout([(60, -70), (100, -25), (60, -70)])               # 발화 프레임 60–159
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    assert len(segs) == 1; s = segs[0]
    assert s["start"] == (60 - 15) * F and s["info"]["pre_roll_ms"] == 480 and s["info"]["reason"] == "vad"
    assert s["end"] == (160 + 25) * F                              # 마지막 유성 프레임 끝 + 정확히 800 ms
    assert s["end_info"]["reason"] == "pause" and s["end_info"]["zero_tail_samples"] == 3200   # 1000 − 800 = 200 ms
    assert s["end_info"]["voiced_s"] == pytest.approx(100 * 0.032)


def test_short_gap_merges_long_gap_splits():
    x = layout([(40, -70), (60, -25), (18, -70), (60, -25), (40, -70), (50, -25), (50, -70)])
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    assert [s["end_info"]["reason"] for s in segs] == ["pause", "pause"]       # 576 ms 틈은 합쳐지고 1.28 s 틈은 끊긴다
    assert segs[0]["end"] == (40 + 60 + 18 + 60 + 25) * F
    assert segs[1]["start"] == max((218 - 15) * F, segs[0]["end"])


def test_end_pause_2000_keeps_200ms_tail_floor():
    x = layout([(40, -70), (60, -25), (90, -70)])
    p, cmds, _ = run(x, params=ap.SegParams(end_pause_ms=2000))
    segs = segments(cmds)
    assert segs[0]["end_info"]["reason"] == "pause" and segs[0]["end_info"]["zero_tail_samples"] == 3200


def test_stop_tail_rule_and_trailing_nonspeech():
    x = layout([(40, -70), (60, -25), (10, -70)])                 # 320 ms 무음 뒤 정지
    p, cmds, _ = run(x, reason="stop")
    s = segments(cmds)[0]
    assert s["end"] == len(x) and s["end_info"]["reason"] == "stop"
    assert s["end_info"]["zero_tail_samples"] == 16000 - 10 * F


def test_soft_max_cut_at_speech_relative_pause_and_soft_start():
    # 18 s 이전의 −45 dB 틈은 휴지가 아니고(L_spk − 25 = −50 보다 크다), 18 s 이후에는 L_spk − 15 dB 기준으로 휴지가 된다
    x = layout([(40, -70), (240, -25), (10, -45), (310, -25), (10, -45), (200, -25), (40, -70)])
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    s = segs[0]
    dip = 40 + 240 + 10 + 310                                      # 두 번째 틈 시작 프레임 600
    assert s["end_info"]["reason"] == "soft_max" and s["end"] == (dip + 8) * F
    assert s["end_info"]["zero_tail_samples"] == 16000 - 8 * F     # 744 ms
    # 다음 세그먼트는 끊긴 곳에서 바로 이어진다. onset 문턱(최근 3 s 중앙값 + 6 dB)은 말소리 기준이라 −45 dB 틈 2 프레임은
    # onset 이 아니고 pre-roll 로 들어간다 (64 ms)
    assert segs[1]["start"] == s["end"] and segs[1]["info"]["reason"] == "soft" and segs[1]["info"]["pre_roll_ms"] == 64


def test_hard_max_forced_cut_quietest_unreleased_frame():
    # 32 s 연속 발화, 24.4–24.9 s 구간에 조용한 3 프레임 → 그 가운데에서 잘려야 한다
    start_f = 40 - 15
    dip = 795
    x = layout([(40, -70), (dip - 40, -25), (3, -40), (1000 - dip - 3, -25)])
    p, cmds, pushes = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    a, b = segs[0], segs[1]
    cut = a["end"]
    assert a["end_info"]["reason"] == "max_len" and a["end_info"]["zero_tail_samples"] == 16000
    assert cut == (dip + 1) * F                                    # 3 프레임 이동 평균의 최소
    L = (cut - start_f * F) / SR
    assert 24.36 <= L <= 25.0
    assert b["start"] == cut and b["info"]["reason"] == "forced" and b["info"]["forced_start"] and b["info"]["pre_roll_ms"] == 0
    assert b["end_info"]["reason"] == "file_end"
    # 소급 절단 없음: seg_end(max_len) 이전 push 들에서 보낸 seg 1 오디오는 모두 cut 이하
    i_end = next(i for i, cs in enumerate(pushes) if any(c.kind == "seg_end" and c.seg == 1 for c in cs))
    sent_before = max(c.info["s1"] for cs in pushes[:i_end] for c in cs if c.kind == "audio" and c.seg == 1)
    assert sent_before <= cut
    # hold-back: 25 − 0.64 − 0.064 = 24.296 s 부터 보류 = 640 ms (+ 부분 프레임/패킷) — 절단 창에 필요한 만큼만 (F33)
    held = [c for c in a["audio_cmds"] if c.info["hold"]]
    assert held and min(c.info["s0"] for c in held) >= (start_f + 759) * F


@pytest.mark.parametrize("seed", range(6))
def test_forced_cut_never_before_released(seed):
    """무작위 연속 발화(> 30 s), 무작위 패킷 크기: 모든 강제 절단은 그때까지 보낸 지점 이후, 길이 ∈ [24.36, 25] s."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([noise(F * 40, -70, seed), noise(SR * 62, -25, seed + 1) * rng.uniform(0.3, 1.0, SR * 62).astype(np.float32)])
    p = ap.AudioPipeline(SR, ap.SegParams()); cmds = []; sent = {}
    i = 0
    while i < len(x):
        n = int(rng.integers(1, 9000)); out = p.push(x[i:i + n]); i += n
        for c in out:
            if c.kind == "seg_end" and c.info["reason"] == "max_len":
                assert c.info["s_end"] >= sent.get(c.seg, 0)
            if c.kind == "audio": sent[c.seg] = c.info["s1"]
        cmds += out
    cmds += p.stop()
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    forced = [s for s in segs if s["end_info"]["reason"] == "max_len"]
    assert len(forced) >= 2
    for s in forced: assert 24.36 <= (s["end"] - s["start"]) / SR <= 25.0


def test_held_bounded_during_long_silence():
    """§4.5 [critic]: 무음 구간에서도 보류량이 유한해야 흐름 제어(audio_done_s)가 멈추지 않는다."""
    x = noise(SR * 20, -70)
    p = ap.AudioPipeline(SR, ap.SegParams()); mx = 0
    for i in range(0, len(x), 320):
        assert p.push(x[i:i + 320]) == []
        mx = max(mx, p.counters()["samples_held"])
    assert mx <= F * (15 + 4) + F
    assert p.stop() == [] and p.counters()["samples_skipped"] == len(x)


def test_determinism_across_packet_sizes_16k_and_48k():
    x16 = layout([(40, -70), (300, -25), (40, -70), (200, -28), (900, -25), (30, -70)], seed=5)
    ref = None
    for pkt in (320, 5120, 777, len(x16)):
        _, cmds, _ = run(x16, pkt=pkt)
        b = brief(segments(cmds))
        ref = ref or b; assert b == ref
    import soxr
    x48 = soxr.resample(x16, SR, 48000, quality="HQ").astype(np.float32)
    outs = []
    for pkt in (960, 15360, 1001):
        p, cmds, _ = run(x48, pkt=pkt, in_rate=48000)
        segs = segments(cmds); check_accounting(p, segs)
        outs.append((brief(segs), np.concatenate([s["pcm"] for s in segs])))
    assert all(o[0] == outs[0][0] and np.array_equal(o[1], outs[0][1]) for o in outs)   # soxr 는 조각 크기와 무관(비트 동일)
    assert len(outs[0][0]) == len(ref)
    for (s48, e48, *_), (s16, e16, *_) in zip(outs[0][0], ref):  # 48 kHz 경로도 거의 같은 경계(±2 프레임)
        assert abs(s48 - s16) <= 0.064 and abs(e48 - e16) <= 0.064


def test_causal_start_speech_at_sample_zero():
    x = layout([(100, -25), (40, -70)])
    p, cmds, _ = run(x, params=ap.SegParams(lead_in=True))
    s = segments(cmds)[0]
    assert s["start"] == 0 and s["info"]["lead_in_ms"] == 0       # 이력 3 s 미만 → lead-in 없음
    assert s["end_info"]["reason"] == "pause"


# ---------------------------------------------------------------- lead-in
def test_lead_in_default_off_and_on_for_back_to_back_onset():
    parts = [(120, -70), (100, -25), (27, -70), (60, -25), (40, -70)]   # 두 번째 발화는 이전 끝 2 프레임 뒤에 시작
    x = layout(parts)
    for on in (False, True):
        p, cmds, _ = run(x, params=ap.SegParams(lead_in=on))
        segs = segments(cmds); c = check_accounting(p, segs, x)
        assert segs[0]["info"]["lead_in_ms"] == 0                  # pre-roll 480 ms 는 충분
        s2 = segs[1]
        assert s2["info"]["pre_roll_ms"] < 160
        if not on:
            assert s2["info"]["lead_in_ms"] == 0 and c["lead_in_s"] == 0
        else:
            assert s2["info"]["lead_in_ms"] == 320 and len(s2["lead"]) == 5120 and c["lead_in_s"] == 5120 / SR
            lvl = 20 * math.log10(np.sqrt(np.mean(s2["lead"].astype(np.float64) ** 2)))
            assert -75.0 < lvl < -71.0                              # 배경(−70) × 0.7 ≈ −73 dB


# ---------------------------------------------------------------- 이득 (§1.5, C7, C9)
def _two_segments(db, extra=None, gain="auto"):
    x = layout([(40, -70), (80, db), (60, -70), (80, db), (40, -70)])
    if extra: extra(x)
    p, cmds, _ = run(x, params=ap.SegParams(gain=gain))
    return x, p, segments(cmds)


@pytest.mark.parametrize("db,g1,g2", [(-50, 20.0, 24.0), (-55, 20.0, 29.0), (-29, 0.0, 0.0), (-23, 0.0, 0.0), (-10, -10.0, -10.0)])
def test_segment_gain(db, g1, g2):
    x, p, segs = _two_segments(db)
    assert len(segs) == 2
    assert segs[0]["info"]["gain_db"] == pytest.approx(g1, abs=0.15)          # 첫 세그먼트: onset 창, 최대 +20
    assert segs[1]["info"]["gain_db"] == pytest.approx(g2, abs=0.15)
    for s in segs:                                     # 세그먼트 안에서 이득은 고정, 보고값(0.1 dB 단위) 그대로 적용
        g = np.float32(10 ** (s["info"]["gain_db"] / 20))
        assert np.array_equal(s["pcm"], x[s["start"]:s["end"]] * g)
    check_accounting(p, segs)


def test_gain_exact_scaling_and_off():
    x, p, segs = _two_segments(-50)
    s = segs[1]; g = 10 ** (p._gain_db / 20)
    assert np.allclose(s["pcm"], x[s["start"]:s["end"]] * g, rtol=1e-6, atol=0)
    _, p2, segs2 = _two_segments(-50, gain="off")
    assert all(s["info"]["gain_db"] == 0 for s in segs2)
    check_accounting(p2, segs2, layout([(40, -70), (80, -50), (60, -70), (80, -50), (40, -70)]), exact=True)


def test_gain_peak_cap_and_limiter():
    def click(x): x[(40 + 80 + 30) * F] = 0.5                       # 두 번째 세그먼트 직전 3 s 안의 피크 0.5
    _, p, segs = _two_segments(-50, extra=click)
    assert segs[1]["info"]["gain_db"] == pytest.approx(-1 - 20 * math.log10(0.5), abs=0.06)   # +5.0 dB 로 제한
    def spike(x): x[(40 + 80 + 60 + 40) * F] = 0.1                   # 세그먼트 시작 뒤의 스파이크 → 리미터
    _, p, segs = _two_segments(-50, extra=spike)
    assert p.counters()["limited_samples"] >= 1
    assert np.abs(segs[1]["pcm"]).max() <= 1.0                     # 소프트 리미터의 점근값


# ---------------------------------------------------------------- 매개변수 (C5, C9)
def test_params_derivation_and_checks():
    r = ap.SegParams(hard_max_s=10).check()
    assert r.soft_max_s == 6.0 and r.hold_from_s == 9.296               # C9: 10−7=3 → 0.6·10=6 로 올림; hold = 10 − 0.64 − 2 프레임 (F33)
    r = ap.SegParams().check()
    assert r.soft_max_s == 18.0 and r.hold_from_s == 24.296 and r.lead_in is False
    assert ap.SegParams(hold_from_s=23.0).check().hold_from_s == 23.0    # 명시값(서버의 hard − 2)은 그대로
    with pytest.raises(ap.ParamError): ap.SegParams(hold_from_s=24.5).check()   # 절단 창(640 ms)을 못 만든다
    with pytest.raises(ap.ParamError): ap.SegParams(tail_s=0.9).check(delay=10)   # 0.08·12 = 0.96
    ap.SegParams(tail_s=0.96).check(delay=10)
    for bad in (dict(end_pause_ms=200), dict(hard_max_s=31), dict(segmentation="x"), dict(gain="on"), dict(pre_roll_ms=100)):
        with pytest.raises(ap.ParamError): ap.SegParams(**bad).check()
    with pytest.raises(ap.ParamError): ap.AudioPipeline(4000, ap.SegParams())


def test_hard_max_10_window_and_soft():
    x = layout([(40, -70), (600, -25), (20, -70)])
    p, cmds, _ = run(x, params=ap.SegParams(hard_max_s=10))
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    forced = [s for s in segs if s["end_info"]["reason"] == "max_len"]
    assert forced and all(10 - 0.64 <= (s["end"] - s["start"]) / SR <= 10.0 for s in forced)


# ---------------------------------------------------------------- force_end · update · 스냅샷
def test_force_end_kv_guard_restarts_and_ignores_stale_seg():
    x = layout([(40, -70), (200, -25), (40, -70)])
    p = ap.AudioPipeline(SR, ap.SegParams()); cmds = []
    for i in range(0, 150 * F, 320): cmds += p.push(x[i:i + 320])
    assert p.force_end("kv_guard", seg=99) == []                   # 다른 세그먼트 id 는 무시
    out = p.force_end("kv_guard", seg=1); cmds += out
    assert [c.kind for c in out] == ["seg_end", "seg_start"]
    assert out[0].info["reason"] == "kv_guard" and out[0].info["zero_tail_samples"] == 16000
    assert out[1].info["reason"] == "kv" and out[1].info["s_start"] == out[0].info["s_end"] and len(out[1].pcm) == 0
    for i in range(150 * F, len(x), 320): cmds += p.push(x[i:i + 320])
    cmds += p.stop()
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    assert [s["end_info"]["reason"] for s in segs] == ["kv_guard", "pause"]
    assert p.force_end("kv_guard") == []                           # stop 뒤에는 무시


def test_update_params_applies_from_next_segment():
    x = layout([(40, -70), (60, -25), (30, -70), (60, -25), (40, -70)])
    p = ap.AudioPipeline(SR, ap.SegParams()); cmds = []
    for i in range(0, len(x), 320):
        cmds += p.push(x[i:i + 320])
        if i == 60 * F: p.update_params(end_pause_ms=1600)          # 첫 세그먼트 도중
    cmds += p.stop("file_end")
    segs = segments(cmds)
    assert segs[0]["end_info"]["reason"] == "pause" and segs[0]["end"] == (40 + 60 + 25) * F   # 열린 세그먼트는 800 ms 그대로
    assert segs[1]["end_info"]["reason"] == "file_end"            # 다음 세그먼트는 1600 ms 가 안 지나 file_end


def test_vad_snapshot_states():
    x = layout([(40, -70), (60, -25), (10, -70)])
    p = ap.AudioPipeline(SR, ap.SegParams())
    keys = {"t", "level_db", "floor_db", "on_db", "off_db", "state", "gain_db", "seg", "limited", "speech_evidence", "calibrating"}
    states = []
    for i in range(0, len(x), F):
        p.push(x[i:i + F]); v = p.vad_snapshot(); assert keys <= set(v); states.append(v["state"])
    assert states[0] == "silence" and p.vad_snapshot()["calibrating"] is False
    assert states[20] == "silence" and states[70] == "speech" and states[-1] == "hangover"
    v = p.vad_snapshot(); assert v["seg"] == 1
    assert v["off_db"] == pytest.approx(max(v["floor_db"] + 6, -64), abs=0.11) and v["on_db"] >= v["off_db"]   # T_on = 중앙값 + 6 ≥ T_off
    assert v["speech_evidence"] is True                               # −25 dB 발화 60 프레임, 바닥 −70
    p.stop(); assert p.vad_snapshot()["seg"] is None


def test_int16_stereo_nan_input_and_push_after_stop():
    p = ap.AudioPipeline(SR, ap.SegParams(segmentation="none", gain="off"))
    cmds = p.push(np.array([16384, -32768], np.int16)) + p.push(np.array([[0.5, 0.25], [np.nan, 0.0]], np.float32))
    cmds += p.stop()
    pcm = np.concatenate([c.pcm for c in cmds if c.pcm is not None])
    assert np.array_equal(pcm, np.array([0.5, -1.0, 0.375, 0.0], np.float32))
    assert p.push(np.zeros(10, np.float32)) == [] and p.counters()["samples_ignored"] == 10


# ---------------------------------------------------------------- 비인과 참조 구현과의 동치 (spec §1.3–1.4 + F06/F07/F08)
_A = math.exp(-2 * math.pi * 80 / SR)


def _hp(x):                                                       # seg_proto.py 의 1차 IIR (float64 로, 샘플 단위)
    x = np.asarray(x, np.float64); y = np.empty_like(x); px = py = 0.0
    for i, v in enumerate(x): py = _A * (py + v - px); px = v; y[i] = py
    return y


def _zero_flags(x, n):
    """프레임마다 |x| ≤ 1e-6 이 64 샘플 이상 연속인지 (파이썬 루프, 구현과 독립)."""
    out = []
    for r in np.abs(np.asarray(x[:n * F], np.float64)).reshape(n, F) <= 1e-6:
        run_ = best = 0
        for v in r: run_ = run_ + 1 if v else 0; best = max(best, run_)
        out.append(best >= 64)
    return out


def ref_run(x, hard_max=25.0, hold_ms=640):
    """비인과 참조: 파일 전체 배열 위에서 새 규칙을 프레임마다 그대로 계산 (스트리밍 구현의 보류·잘라내기·패킷과 무관).
    E = 1차 80 Hz IIR 에너지, D = 4 ms 소블록 8 개 에너지의 중앙값, 디지털 무음 프레임은 바닥·문턱 고정,
    N = 최근 94 비무음 프레임의 10 % 분위수(T_off = N + 6), T_on = 중앙값 + 6 (onset 4 프레임 중 E 3 개, D 1 개),
    유성 = min(E, D) ≥ 최소 추적 바닥 + 10, SPEECH 유성 = 최근 5 프레임 중 E 3 개 ≥ max(T_off, L_spk − 25), 마지막 유성은 D 도,
    soft max 뒤에는 최근 2 프레임에 E·D 모두 문턱 이상인 프레임이 있으면 끝내지 않는다."""
    P = dict(pre_roll=15, end_pause=25, soft_max=max(hard_max - 7, 0.6 * hard_max), end_pause_soft=8, hold=20,
             hold_from=round(hard_max - hold_ms / 1000 - 2 * F / SR, 6), hard_max=hard_max)
    n = len(x) // F
    pw = _hp(x)[:n * F].reshape(n, F) ** 2
    e = 10 * np.log10(np.mean(pw, 1) + 1e-10)
    d = 10 * np.log10(np.median(pw.reshape(n, 8, F // 8).mean(2), 1) + 1e-10)
    z = _zero_flags(x, n)
    t0 = next((k for k in range(n) if not z[k]), n)              # 앞쪽 디지털 무음은 바닥 정보가 없다
    ks = [k for k in range(t0, min(n, t0 + 94)) if not z[k]][:16]
    clip = lambda v: float(np.clip(v, -85.0, -35.0))
    win, vo = [], []; segs = []; cur = None; last_v = 0; prev_end = 0
    ton, toff, tonl = -60.0, -64.0, -60.0                        # 초기화 전 (앞쪽 무음): ABS 문턱
    for t in range(n):
        if t == t0 and t0 < n:
            if ks:
                N = clip(np.percentile(e[ks], 10)); nlvl = clip(min(e[ks]))
                ton = max(min(float(np.percentile(e[ks], 50)) + 6, N + 10, -29.0), -60.0)
            else:
                N = nlvl = -85.0; ton = -60.0
            toff, tonl = max(N + 6, -64.0), max(nlvl + 10, -60.0)
        if t >= t0 and not z[t]:
            win = (win + [e[t]])[-94:]; nlvl = clip(min(e[t], nlvl + 0.5 * F / SR))
            if len(win) >= 16:
                N = clip(np.percentile(win, 10)); ton = max(min(float(np.percentile(win, 50)) + 6, N + 10, -29.0), -60.0)
            toff, tonl = max(N + 6, -64.0), max(nlvl + 10, -60.0)
        if cur is None:
            w = e[max(0, t - 3): t + 1]
            if len(w) >= 3 and (w >= ton).sum() >= 3 and (d[max(0, t - 3): t + 1] >= ton).any():
                onset = t - len(w) + 1 + int(np.argmax(w >= ton))
                cur = dict(start=max(onset - P["pre_roll"], prev_end)); last_v = t
            continue
        L = (t + 1 - cur["start"]) * F / SR
        if min(e[t], d[t]) >= tonl: vo.append(e[t]); del vo[:-150]
        if L >= P["soft_max"]: thr = max(toff, (float(np.median(vo)) if vo else tonl) - 15)
        elif len(vo) >= 30: thr = max(toff, float(np.median(vo)) - 25)
        else: thr = toff
        hits = [k for k in range(max(0, t - 4), t + 1) if e[k] >= thr]
        good = [k for k in hits if d[k] >= thr]
        if len(hits) >= 3 and good: last_v = max(last_v, good[-1])
        fresh = L >= P["soft_max"] and any(e[k] >= thr and d[k] >= thr for k in range(max(0, t - 1), t + 1))
        if t - last_v >= (P["end_pause_soft"] if L >= P["soft_max"] else P["end_pause"]) and not fresh:
            cur.update(end=t + 1, why="soft" if L >= P["soft_max"] else "pause"); segs.append(cur); prev_end = t + 1; cur = None
            continue
        if L >= P["hard_max"]:
            lo = max(cur["start"] + int(P["hold_from"] * SR / F), t + 1 - P["hold"])
            cut = lo + int(np.argmin(np.convolve(e[lo:t + 1], np.ones(3) / 3, "same")))
            cur.update(end=cut, why="hard"); segs.append(cur); prev_end = cut; cur = dict(start=cut); last_v = t
    if cur is not None: cur.update(end=n, why="eof"); segs.append(cur)
    return [(s["start"] * F, s["end"] * F, s["why"]) for s in segs]


WHY = {"pause": "pause", "soft_max": "soft", "max_len": "hard", "file_end": "eof"}


def _mine_vs_ref(x, pkt):
    ref = ref_run(x)
    p, cmds, _ = run(x, pkt=pkt)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    mine = [(s["start"], s["end"], WHY[s["end_info"]["reason"]]) for s in segs]
    if mine and mine[-1][2] == "eof": mine[-1] = (mine[-1][0], len(x) // F * F, "eof")   # 참조는 마지막 부분 프레임을 버린다
    return mine, ref


@pytest.mark.parametrize("seed", [100, 102])
def test_matches_reference_synthetic(seed):
    """무작위 발화/틈 140–220 s (디지털 무음 구간 포함): 비인과 참조와 인과 스트리밍 구현의 경계가 프레임 단위로 같다."""
    rng = np.random.default_rng(seed); parts = [(30, -68)]
    for _ in range(20):
        parts += [(int(rng.integers(10, 500)), float(rng.uniform(-40, -18))), (int(rng.integers(3, 50)), float(rng.uniform(-80, -55)))]
    parts += [(900, -24), (40, -70)]                              # 끊김 없는 28.8 s → 강제 절단
    x = layout(parts, seed=seed - 100)
    for a in rng.integers(F * 40, len(x) - F * 1200, 3):           # 뮤트처럼 정확한 0 (프레임 경계와 무관한 위치·길이)
        x[a:a + int(rng.integers(100, 40000))] = 0
    for pkt in (int(rng.integers(100, 6000)), 320):
        mine, ref = _mine_vs_ref(x, pkt=pkt)
        assert mine == ref
    assert {"pause", "soft", "hard"} <= {w for *_, w in ref}       # 세 가지 끝 규칙을 모두 거친다


def test_matches_reference_with_digital_zero_lead():
    x = layout([(5, -70), (30, -70), (60, -25), (40, -70), (60, -28), (40, -70)], seed=7)
    x[:5 * F + 100] = 0                                            # 5 프레임 + 100 샘플의 0 (초기화를 미룬다)
    for pkt in (320, 777, len(x)):
        mine, ref = _mine_vs_ref(x, pkt=pkt)
        assert mine == ref and [w for *_, w in ref] == ["pause", "pause"]


@pytest.mark.skipif(not glob.glob(TESTAUDIO + "/*.wav"), reason="scratchpad testaudio 없음")
def test_matches_reference_on_testaudio_with_pauses():
    sf = pytest.importorskip("soundfile")
    clips = [sf.read(f, dtype="float32")[0] for f in sorted(glob.glob(TESTAUDIO + "/*.wav"))]
    gaps = [1.0, 1.2, 0.5, 2.0, 1.5, 0.9, 1.0]
    parts = [noise(int(gaps[0] * SR), -72, 0)]
    for i, c in enumerate(clips): parts += [c, noise(int(gaps[i + 1] * SR), -72, i + 1)]
    x = np.concatenate(parts)
    mine, ref = _mine_vs_ref(x, pkt=320)
    assert mine == ref
    assert [w for *_, w in ref].count("pause") >= 5                 # 1 s 이상의 틈은 모두 끊긴다


# ---------------------------------------------------------------- 실제 방 조건 회귀 (F06 소음·타자, F07 험, F08 디지털 0)
def set_db(x, db):
    x = np.asarray(x, np.float64); return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (db / 20)).astype(np.float32)


def speechlike(n, db, seed=0, rate=SR):
    """음절 포락선(4–6 Hz, 깊이 ~20 dB)을 씌운 백색 잡음 — 말소리 대용."""
    rng = np.random.default_rng(seed); t = np.arange(n) / rate
    env = 0.1 + 0.9 * np.abs(np.sin(np.pi * rng.uniform(4, 6) * t + rng.uniform(0, 3))) ** 1.5
    return set_db(rng.standard_normal(n) * env, db)


def fluct(n, db, seed=0, sd=4.0, rate=SR):
    """요동하는 방 소음: 백색 × 로그 정규 포락선 (표준편차 sd dB, 상관 ~40 ms)."""
    rng = np.random.default_rng(seed); hop = rate // 100; m = n // hop + 2
    u = np.convolve(rng.standard_normal(m), np.ones(4) / 2.0, "same")
    env = np.interp(np.arange(n) / hop, np.arange(m), 10 ** (sd * u / 20))
    return set_db(rng.standard_normal(n) * env, db)


def pink(n, db, seed=0):
    X = np.fft.rfft(np.random.default_rng(seed).standard_normal(n)); f = np.arange(len(X)); f[0] = 1
    return set_db(np.fft.irfft(X / np.sqrt(f), n), db)


def clicks(n, db_peak, rate_hz=5.0, seed=3, rate=SR):
    """노트북 키보드: 타건마다 누름·뗌 두 번의 4 ms 감쇠 버스트 (60–110 ms 간격), 미분으로 저역 제거."""
    rng = np.random.default_rng(seed); x = np.zeros(n); t = 0.2; L = int(0.006 * rate); env = np.exp(-np.arange(L) / (0.0012 * rate))
    while t < n / rate - 0.1:
        for dt in (0.0, rng.uniform(0.06, 0.11)):
            i = int((t + dt) * rate); x[i:i + L] += rng.standard_normal(L) * env * rng.uniform(0.5, 1.0)
        t += rng.exponential(1 / rate_hz) + 0.05
    x = np.diff(x, prepend=0.0); return (x / np.abs(x).max() * 10 ** (db_peak / 20)).astype(np.float32)


def hum(n, f0, db, harmonics=(1.0,), rate=SR):
    t = np.arange(n) / rate
    return set_db(sum(a * np.sin(2 * np.pi * f0 * (k + 1) * t + 0.3 * k) for k, a in enumerate(harmonics)), db)


def with_utterances(bg, spans, db=-26, seed=10, rate=SR):
    x = np.array(bg, np.float32, copy=True)
    for i, (a, b) in enumerate(spans):
        i0, i1 = int(a * rate), int(b * rate); x[i0:i1] += speechlike(i1 - i0, db, seed + i, rate)
    return x


def end_delays(segs, spans):
    """발화마다: 그 발화 끝 이후 처음 끝나는 세그먼트까지의 지연(s)."""
    ends = [s["end"] / SR for s in segs]
    return [min((e - b for e in ends if e >= b - 0.05), default=None) for _, b in spans]


SPANS5 = [(1.5, 4.0), (6.0, 8.2), (10.0, 13.0), (15.2, 17.0), (19.0, 21.5)]


def test_highpass_feature_is_spec_iir_across_chunks():
    """VAD 특징 = spec 의 1차 80 Hz IIR 을 프레임 사이 상태를 이어 계산한 것 (조각 크기와 무관, 오차 < 1e-9 dB)."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 4) * 0.05).astype(np.float32); x[30000:40000] = 0; x[50000:50100] += 0.5
    fr = x[: len(x) // F * F].reshape(-1, F)
    ref = 10 * np.log10(np.mean(_hp(x)[:fr.size].reshape(-1, F) ** 2, 1) + 1e-10)
    hp = ap.HighPassDb(); got, i = [], 0
    for k in (1, 7, 3, 30, 1, 200):
        got.append(hp(fr[i:i + k])); i += k
    got.append(hp(fr[i:]))
    assert np.abs(np.concatenate(got) - ref).max() < 1e-9 and np.abs(ap.frame_db(fr) - ref).max() < 1e-9


def test_digital_silence_flag():
    fr = noise(F * 4, -60, 1).reshape(4, F).copy()
    fr[0, 100:164] = 0                                             # 64 샘플 연속 0 → 디지털 무음
    fr[1, 100:163] = 0                                             # 63 샘플 → 아님
    fr[2] = np.random.default_rng(2).standard_normal(F) * 1e-9     # soxr 이 0 구간에 남기는 잔향 수준
    fr[3] = np.round(noise(F, -90, 5) * 32768) / 32768             # int16 −90 dBFS 잡음: 0 이 37 % 여도 64 연속은 없다
    assert ap.digital_silence(fr).tolist() == [True, False, True, False]


@pytest.mark.parametrize("f0,db,harm", [(50.0, -40, (1.0,)), (50.1, -40, (1.0,)), (50.0, -45, (1.0, .5, .35, .2, .12)),
                                        (60.0, -45, (1.0,)), (60.0, -40, (1.0, .5, .35, .2, .12))])
def test_mains_hum_does_not_block_pauses(f0, db, harm):
    """F07: 50/60 Hz 험(순음·배음)이 있어도 발화마다 휴지로 끝난다 (rfft 근사는 50 Hz 순음이 −8.5 dB 로 새어 휴지를 막았다)."""
    spans = [(1.0, 3.5), (5.5, 7.5), (9.5, 12.0), (14.0, 16.0)]; n = 18 * SR
    x = with_utterances(noise(n, -75, 1) + hum(n, f0, db, harm), spans, -30)
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    assert [s["end_info"]["reason"] for s in segs] == ["pause"] * 4
    assert all(0.6 <= d <= 1.2 for d in end_delays(segs, spans))


@pytest.mark.parametrize("rate,kind,db,zero", [
    (48000, "white", -45, ("lead", 0.032)), (48000, "white", -45, ("lead", 0.031)), (48000, "white", -45, ("lead", 0.063)),
    (48000, "white", -35, ("lead", 0.032)), (16000, "white", -45, ("lead", 0.032)), (16000, "pink", -50, ("lead", 0.2)),
    (48000, "white", -45, ("mute", 20.016, 2.0)), (48000, "pink", -45, ("mute", 30.031, 2.0)), (16000, "white", -45, ("mute", 20.0, 0.05))])
def test_digital_zero_runs_do_not_pin_the_floor(rate, kind, db, zero):
    """F08: 0 으로 시작하거나 중간에 뮤트(0)가 들어가도 고른 소음은 말소리로 취급되지 않는다 (예전: 32 ms 0 하나로 50–70 s 세그먼트)."""
    n = 40 * rate
    x = noise(n, db, 1) if kind == "white" else pink(n, db, 1)
    if zero[0] == "lead": x[: int(zero[1] * rate)] = 0
    else: x[int(zero[1] * rate): int((zero[1] + zero[2]) * rate)] = 0
    p, cmds, _ = run(x, pkt=rate // 50, in_rate=rate)
    segs = segments(cmds); check_accounting(p, segs)
    assert segs == []
    assert p.vad_snapshot()["floor_db"] > db - 10                   # 바닥이 −85 에 묶이지 않았다


def test_zero_lead_keeps_utterance_segments():
    R = 48000; n = 18 * R; spans = [(1.0, 3.5), (5.5, 7.5), (9.5, 12.0), (14.0, 16.0)]
    x = with_utterances(noise(n, -45, 1), spans, -26, rate=R)
    _, c0, _ = run(x, pkt=960, in_rate=R)
    y = x.copy(); y[: int(0.032 * R)] = 0
    _, c1, _ = run(y, pkt=960, in_rate=R)
    assert brief(segments(c1)) == brief(segments(c0)) and [r for *_, r, _ in brief(segments(c0))] == ["pause"] * 4


def test_quiet_int16_noise_still_detects_burst():
    """디지털 무음 판정은 샘플 연속 0 기준이라 아주 작은 int16 잡음(0 이 많다)은 무음이 아니고, 그 위의 발화를 놓치지 않는다."""
    n = 6 * SR; x = np.round(noise(n, -90, 2) * 32768)
    t = np.arange(SR) / SR; x[2 * SR:3 * SR] += np.round(10 ** (-29 / 20) * np.sqrt(2) * 32768 * np.sin(2 * np.pi * 440 * t))
    p, cmds, _ = run(x.astype(np.int16), pkt=320)
    segs = segments(cmds)
    assert len(segs) == 1 and segs[0]["start"] <= 2 * SR and segs[0]["end"] >= 3 * SR


@pytest.mark.parametrize("db", [-55, -50, -45])
def test_fluctuating_room_noise_still_ends_on_pauses(db):
    """F06: 요동하는 방 소음(포락선 ±4 dB) 위 다섯 발화가 각각 휴지로 끝난다 (최소 추적 바닥 + 6 dB 문턱은 휴지를 못 찾아 18–25 s 로 합쳤다)."""
    n = 24 * SR; x = with_utterances(fluct(n, db, 1), SPANS5, -26)
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    assert [s["end_info"]["reason"] for s in segs] == ["pause"] * 5
    assert all(0.6 <= d <= 2.0 for d in end_delays(segs, SPANS5))
    for s, (a, b) in zip(segs, SPANS5): assert s["start"] <= a * SR and s["end"] >= b * SR   # 말소리를 자르지 않는다


@pytest.mark.parametrize("pk,rate_hz", [(-18, 5.0), (-12, 5.0), (-18, 8.0)])
def test_keyboard_typing_does_not_hold_segments_open(pk, rate_hz):
    """F06: 말하는 중 타자(누름·뗌 클릭)가 휴지를 되돌리지 않는다."""
    n = 16 * SR; spans = SPANS5[:3]
    x = with_utterances(noise(n, -70, 1), spans, -26) + clicks(n, pk, rate_hz)
    p, cmds, _ = run(x)
    segs = [s for s in segments(cmds) if s["end"] > 1.5 * SR]      # 첫 발화 앞에서 타자로 열린 짧은 세그먼트는 뺀다
    ends = [s["end"] / SR for s in segs]
    for a, b in spans:                                              # 발화마다 끝난 뒤 1.5 s 안에 휴지로 끝난다
        assert any(b + 0.6 <= e <= b + 1.5 for e in ends), (b, ends)


@pytest.mark.parametrize("pk,rate_hz", [(-24, 5.0), (-12, 8.0)])
def test_typing_alone_is_not_speech(pk, rate_hz):
    """타자만 있으면 말소리가 아니다: 세그먼트가 열려도 유성 프레임이 없어 숨김 대상(< 6)이고 이득도 0."""
    p, cmds, _ = run(noise(20 * SR, -70, 1) + clicks(20 * SR, pk, rate_hz))
    segs = segments(cmds)
    assert len(segs) <= 3 and all(s["end_info"]["voiced_s"] < 6 * F / SR and s["info"]["gain_db"] == 0 for s in segs)


def test_soft_max_does_not_cut_a_224ms_dip():
    """soft max(18 s) 뒤 7 프레임(224 ms) 틈은 휴지(256 ms)가 아니다 — 말이 다시 시작될 때 5 중 3 확인을 기다리느라
    2 프레임 늦게 '유성'이 되어도 그사이에 끝내면 다음 단어 첫머리가 잘린다 (chain-en 37.28 s 에서 'recession' 손실)."""
    x = layout([(40, -70), (620, -25), (7, -70), (300, -25), (40, -70)])
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    dip = (40 + 620) * F
    assert not any(dip <= s["end"] <= dip + 10 * F for s in segs), brief(segs)


def test_continuous_speech_has_no_pause_end_inside():
    """바닥(분위수)이 연속 발화 중에 올라가도 발화 안에서 'pause' 로 끊지 않는다 — 긴 독백은 soft max/강제 절단으로만."""
    n = 50 * SR; x = noise(n, -70, 1); x[SR:46 * SR] += speechlike(45 * SR, -26, 4)
    p, cmds, _ = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=True)
    assert len(segs) >= 2 and all(s["end_info"]["reason"] in ("soft_max", "max_len") for s in segs[:-1])
    assert 46.2 <= segs[-1]["end"] / SR <= 47.0                   # 말이 끝난 뒤 (18 s 넘은 세그먼트라 256 ms 휴지의 soft_max 끝)
    assert segs[0]["start"] <= SR and sum(s["end"] - s["start"] for s in segs) >= 45 * SR


# ---------------------------------------------------------------- 이득: 소음·첫 세그먼트·번갈아 말하기 (F31)
def test_first_segment_weak_onset_gets_no_boost():
    """C9: 보통 크기(−26 dBFS) 발화가 약한 앞부분(−50 dB 숨소리 150 ms)으로 시작해도 첫 세그먼트를 올리지 않는다
    (예전: onset 창 3–4 프레임 중앙값으로 +20 dB → 리미터)."""
    n = 8 * SR; x = noise(n, -75, 1); a = 2 * SR
    x[a:a + 2400] += noise(2400, -50, 2); x[a + 2400:a + 4 * SR] += speechlike(4 * SR - 2400, -26, 3)
    p, cmds, _ = run(x)
    segs = segments(cmds)
    assert segs[0]["info"]["gain_db"] == 0.0 and p.counters()["limited_samples"] == 0


def test_noise_opened_first_segment_gets_no_boost():
    """소음이 먼저 세그먼트를 열어도(onset 이 소음) 소음 수준으로 이득을 정하지 않는다."""
    n = 10 * SR; x = with_utterances(fluct(n, -50, 3, sd=6.0), [(1.0, 4.0), (6.0, 8.5)], -26)
    p, cmds, _ = run(x)
    segs = segments(cmds)
    assert all(s["info"]["gain_db"] <= 0 for s in segs) and p.counters()["limited_samples"] == 0


def test_noise_only_session_never_boosts():
    """말소리 근거(유성 이력, 조용한 때 바닥 대비 15 dB)가 없으면 양의 이득을 주지 않는다 ('최대 이득' 경고의 근거도 없음)."""
    p, cmds, _ = run(fluct(30 * SR, -55, 2, sd=6.0))
    segs = segments(cmds)
    assert all(s["info"]["gain_db"] <= 0 for s in segs)
    assert p.vad_snapshot()["speech_evidence"] is False


def test_alternating_near_far_speakers_no_cross_boost():
    """먼 화자(−40) 이력으로 가까운 화자(−22)를 올리지 않는다 — 지금 onset 이 이미 크면 양의 이득을 막는다."""
    spans = [(1.0, 4.0), (5.2, 8.2), (9.4, 12.4), (13.6, 16.6)]; n = 18 * SR; x = noise(n, -70, 1)
    for i, ((a, b), db) in enumerate(zip(spans, (-40, -22, -40, -22))): x[int(a * SR):int(b * SR)] += speechlike(int(b * SR) - int(a * SR), db, 20 + i)
    p, cmds, _ = run(x)
    segs = segments(cmds)
    assert len(segs) == 4
    assert segs[1]["info"]["gain_db"] <= 3.0 and segs[3]["info"]["gain_db"] <= 3.0
    assert p.counters()["limited_samples"] == 0


def test_first_segment_boost_is_deferred_until_evidence():
    """올림 후보인 첫 세그먼트는 시작 명령을 유성 16 프레임(또는 onset 뒤 1 s)까지 미루고, 그 구간으로 이득을 정한다.
    잡아 둔 오디오는 시작 묶음으로 한꺼번에 나가며 표본 회계·순서는 그대로다. 짧은 발화는 끝날 때 바로 정한다."""
    n = 8 * SR; x = with_utterances(noise(n, -80, 1), [(1.0, 4.0)], -52)
    x[5 * SR:int(5.4 * SR)] += speechlike(int(0.4 * SR), -52, 9)    # 0.4 s 짧은 발화 (이력 30 프레임 전이면 다시 미룸)
    p, cmds, pushes = run(x)
    segs = segments(cmds); check_accounting(p, segs, x, exact=False)
    i_start = next(i for i, cs in enumerate(pushes) if any(c.kind == "seg_start" for c in cs))
    sent_at = (i_start + 1) * 320                                  # 시작 명령이 나온 push 까지 받은 샘플
    onset = round(segs[0]["info"]["onset_t"] * SR)
    assert 10 * F <= sent_at - onset <= 32 * F + 320                # 바로가 아니라 근거를 모은 뒤 (≤ 1 s)
    assert segs[0]["info"]["gain_db"] == 20.0 and len(segs[0]["pcm"]) >= sent_at - segs[0]["start"] - 320 - F
    assert len(segs) == 2 and segs[1]["end_info"]["reason"] == "pause"


def test_force_end_and_stop_while_first_segment_deferred():
    x = with_utterances(noise(6 * SR, -80, 1), [(1.0, 5.5)], -52)
    p = ap.AudioPipeline(SR, ap.SegParams()); cmds = []
    for i in range(0, int(1.3 * SR), 320): cmds += p.push(x[i:i + 320])
    assert p._defer is not None and not any(c.kind == "seg_start" for c in cmds)   # 아직 미룬 상태
    out = p.force_end("kv_guard", seg=1); cmds += out
    assert [c.kind for c in out][:2] == ["seg_start", "seg_end"]   # 끝내기 전에 시작을 먼저 낸다
    for i in range(int(1.3 * SR) // 320 * 320 + 320, len(x), 320): cmds += p.push(x[i:i + 320])
    cmds += p.stop()
    segs = segments(cmds); check_accounting(p, segs)
    assert segs[0]["end_info"]["reason"] == "kv_guard" and segs[-1]["end_info"]["reason"] in ("pause", "stop")


def test_quiet_talker_still_boosted_with_evidence():
    """C7: 조용한 화자(−52 dBFS, 바닥 −80)는 첫 세그먼트 +20, 이력 뒤 ≈ +26 dB 그대로."""
    spans = [(1.0, 4.0), (5.5, 8.5)]; n = 10 * SR
    x = with_utterances(noise(n, -80, 1), spans, -52)
    p, cmds, _ = run(x)
    segs = segments(cmds)
    assert segs[0]["info"]["gain_db"] == 20.0 and 22.0 <= segs[1]["info"]["gain_db"] <= 30.0
    assert p.vad_snapshot()["speech_evidence"] is True


# ---------------------------------------------------------------- hold-back 시작 (F33)
def test_hold_starts_only_for_cut_window_and_boundaries_unchanged():
    """hold 를 hard_max − 0.64 − 2 프레임(24.296 s)에 시작해도 강제 절단 창·경계·PCM 이 23 s 시작과 비트 단위로 같고, 보류 오디오만 준다."""
    rng = np.random.default_rng(3)
    x = np.concatenate([noise(F * 40, -70, 3), noise(SR * 62, -25, 4) * rng.uniform(0.3, 1.0, SR * 62).astype(np.float32)])
    out = []
    for params in (ap.SegParams(), ap.SegParams(hold_from_s=23.0)):
        p = ap.AudioPipeline(SR, params); cmds = []; i = 0; r2 = np.random.default_rng(9)
        while i < len(x):
            k = int(r2.integers(1, 9000)); cmds += p.push(x[i:i + k]); i += k
        cmds += p.stop()
        segs = segments(cmds)
        held = sum(c.info["s1"] - c.info["s0"] for s in segs for c in s["audio_cmds"] if c.info["hold"])
        out.append((brief(segs), [s["pcm"] for s in segs], held))
    (b0, p0, h0), (b1, p1, h1) = out
    assert b0 == b1 and all(np.array_equal(a, b) for a, b in zip(p0, p1))
    assert [r for *_, r, _ in b0].count("max_len") >= 2 and h0 < h1


# ---------------------------------------------------------------- 톤 버스트 · 입력 표본율 · DC
def tone_bursts(rate, seed=0):
    """배경 −70 dB 위에 300–1200 Hz 톤 버스트(−24 dBFS, 10 ms 램프) 4 개: (신호, [(시작 s, 끝 s)])."""
    rng = np.random.default_rng(seed); spans, t = [], 1.0
    for d, g in ((1.2, 0.4), (0.9, 1.3), (2.0, 0.5), (1.5, 1.0)):
        spans.append((t, t + d)); t += d + g
    n = int((t + 0.5) * rate); x = noise(n, -70, seed).astype(np.float64)
    for (a, b), f in zip(spans, (300, 520, 800, 1200)):
        i, j = int(a * rate), int(b * rate); k = np.arange(j - i)
        env = np.minimum(1, np.minimum(k, j - i - k) / (0.01 * rate))
        x[i:j] += 10 ** (-24 / 20) * np.sqrt(2) * env * np.sin(2 * np.pi * f * k / rate)
    return x.astype(np.float32), spans


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000, 96000])
def test_tone_bursts_any_rate(rate):
    x, spans = tone_bursts(rate)
    p, cmds, _ = run(x, pkt=max(1, rate // 50), in_rate=rate)
    segs = segments(cmds); c = check_accounting(p, segs)
    assert abs(c["samples_rx"] - len(x) * SR / rate) <= 1 and c["samples_held"] == 0
    # 틈 1.3 s 와 1.0 s 는 끊고, 0.4 s·0.5 s 는 합친다 → (버스트 1+2), (버스트 3+4)
    assert [s["end_info"]["reason"] for s in segs] == ["pause", "pause"]
    for s, (a, b) in zip(segs, ((spans[0][0], spans[1][1]), (spans[2][0], spans[3][1]))):
        assert abs(s["start"] / SR - (a - 0.48)) <= 0.04                # pre-roll 480 ms (±프레임 정렬·램프)
        assert abs(s["end"] / SR - (b + 0.8)) <= 0.05                    # 끝 + 800 ms
        assert s["end_info"]["zero_tail_samples"] == 3200


def test_dc_offset_does_not_trigger_vad():
    x = noise(SR * 6, -70) + np.float32(0.05)                             # −26 dBFS DC 는 VAD 경로에서 제거된다
    p, cmds, _ = run(x)
    assert segments(cmds) == [] and p.counters()["samples_skipped"] == len(x)
    y = x.copy(); y[SR * 3:SR * 4] += noise(SR, -25, 1)
    p, cmds, _ = run(y)
    segs = segments(cmds); check_accounting(p, segs, y, exact=True)
    assert len(segs) == 1 and np.array_equal(segs[0]["pcm"], y[segs[0]["start"]:segs[0]["end"]])   # 모델 쪽은 필터 없음
