"""브라우저 클라이언트(web/app.js) 동작 테스트: node vm 에 브라우저 흉내와 함께 올려 시나리오를 돌린다 (web_harness.js).
node 가 없으면 건너뛴다. 실제 브라우저 확인은 작업 기록(헤드리스 Chrome)에 있다."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(APP, "web", "app.js")
HARNESS = os.path.join(APP, "tests", "web_harness.js")
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node 없음")


def scen(name, *args, timeout=60):
    p = subprocess.run([NODE, HARNESS, JS, name, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, (p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    r = json.loads(lines[-1])
    assert "error" not in r, r["error"]
    return r


# ── F01: 빠르게 보내기 루프는 자기 실행이 끝나면 멈춘다 ───────────────────────────────────────────────────────
@needs_node
def test_fast_loop_stops_after_preempt():
    r = scen("zombie")
    assert r["after_preempt_frames"] == 0 and r["after_preempt_text"] == []
    assert r["s2_old"] == 0 and r["s2_new"] == 188, r          # 60 s / 0.32 s = 188 패킷, 옛 파일 0
    assert r["s2_text"] == ["start", "stop"], r
    assert "file" not in r["banners"], r["banners"]


@needs_node
def test_fast_loop_preempt_mid_4x_sends_nothing_more():
    r = scen("short4x")
    assert r["frames_after"] == 0 and r["text_after"] == [] and r["run"] is None, r
    assert "file" not in r["banners"], r["banners"]            # 'Cannot set properties of null' 배너 없음


@needs_node
def test_stale_decode_does_not_start_second_session():
    r = scen("decodeRace")
    assert r["text_ws2"] == ["start", "stop"] and r["a_frames"] == 0 and r["b_frames"] == 10, r


@needs_node
@pytest.mark.parametrize("how", ["full", "userstop"])
def test_fast_loop_normal_and_user_stop_unchanged(how):
    r = scen("normal", how)
    assert r["text"] == ["start", "stop"] and r["stopping"] is True and r["run"] is None, r
    if how == "full":
        assert r["frames"] == 63 and r["samples"] == 320000, r
    else:
        assert 0 < r["frames"] < 63, r


# ── F09: 시작 도중 정지·중단 ───────────────────────────────────────────────────────────────────────────
@needs_node
def test_stop_during_getusermedia_releases_mic():
    r = scen("stopDuringGum")
    assert r["mid"] == {"kind": "mic", "startSent": False}
    assert r["run_after_stop"] is None and r["run"] is None and r["cap"] is None, r      # UI 는 바로 준비됨
    assert r["tracks"] == ["ended"] and r["ctxs"] == ["closed"], r                        # 늦게 온 스트림도 닫힘
    assert r["text"] == [] and r["banners"] == {}, r                                      # 서버에 아무것도 안 감, TypeError 배너 없음


@needs_node
def test_double_click_does_not_start_session():
    r = scen("doubleClick")
    assert r["run"] is None and r["text"] == [] and r["gum"] == 0 and r["ctxs"] == ["closed"], r
    assert r["banners"] == {}, r


@needs_node
def test_stop_after_start_sent_keeps_server_in_sync():
    r = scen("stopAfterStartSent")
    assert r["sentStart"] is True and r["stopping"] is True, r
    assert r["text"] == ["start", "stop"] and r["run"] is None and r["tracks"] == ["ended"], r


@needs_node
def test_abort_during_getusermedia_releases_mic():
    r = scen("abortDuringGum")
    assert r["run_after_abort"] is None and r["run"] is None and r["tracks"] == ["ended"] and r["text"] == [], r


# ── F10: 브라우저 오류는 이름별 한국어 배너, 서버 오류는 서버 배너 하나 ────────────────────────────────────────────
@needs_node
@pytest.mark.parametrize("kind,want", [
    ("NotFoundError", "마이크 장치가 없습니다"),
    ("AbortError", "다른 앱이 마이크를"),
    ("NotReadableError", "다른 앱이 마이크를"),
    ("NotAllowedError", "주소창 왼쪽 사이트 설정"),
    ("NotAllowedSystem", "시스템 설정 › 개인정보 보호 및 보안 › 마이크"),
    ("SecurityError", "마이크 권한이"),
    ("addModule", "오디오 처리 모듈"),
    ("graph", "96 kHz 이하"),
    ("Overconstrained", "기본 마이크로 바꿨습니다"),
])
def test_mic_errors_are_shown_in_korean(kind, want):
    r = scen("micError", kind)
    assert r["run"] is None and r["text"] == [], r
    assert "mic" in r["banners"] and want in r["banners"]["mic"], r["banners"]
    assert all(c == "closed" for c in r["ctxs"]), r
    if kind == "Overconstrained":
        assert r["mic_value"] == "", r


@needs_node
def test_server_error_shows_only_server_banner():
    r = scen("micError", "busy")
    assert r["run"] is None and "busy" in r["banners"] and "mic" not in r["banners"], r


# ── F37: 단축키 ────────────────────────────────────────────────────────────────────────────────────────
@needs_node
def test_keyboard_shortcuts():
    r = scen("keys")
    assert r["esc_on_select"]["run_alive"] and r["esc_on_select"]["blurred"] == ["select"], r     # 입력란의 Esc 는 중단하지 않음
    assert "abort" not in r["esc_on_select"]["text"], r
    assert r["space_repeat"]["run_alive"] and r["space_repeat"]["prevented"], r                  # 자동 반복 무시(스크롤도 막음)
    assert r["space_shift"]["run_alive"] and not r["space_shift"]["prevented"], r                # 수정 키 조합은 그대로
    assert r["esc_on_button"]["aborting"] and "abort" in r["esc_on_button"]["text"], r           # 버튼 포커스에서는 Esc = 중단
    assert r["space_disconnected"]["run"] is None and r["space_disconnected"]["gum"] == 1, r     # 연결 끊김 중 Space 는 아무것도 안 함
    assert r["space_on_radio"]["started"] and r["space_on_radio"]["prevented"], r


# ── F38: 고급 숫자 ─────────────────────────────────────────────────────────────────────────────────────
@needs_node
def test_option_ranges_are_clamped_and_never_persisted_invalid():
    r = scen("opts")
    assert r["loaded"]["pause"] == 300 and r["loaded"]["max"] == 30, r
    assert r["loaded"]["saved"]["end_pause_ms"] == 300 and r["loaded"]["saved"]["hard_max_s"] == 30, r
    assert r["clamp"] == [300, 800, 800, 2000, 812, 10, 25, 12.5], r
    assert r["changed"]["value"] == 300 and r["changed"]["saved"] == 300 and "300–2000" in r["changed"]["msg"], r
    assert r["valid"]["msg"] == "", r
    assert r["start_opts"] == {"end_pause_ms": 900, "hard_max_s": 25}, r


# ── F35: 밀림 표시 ─────────────────────────────────────────────────────────────────────────────────────
@needs_node
def test_lag_indicator_single_banner_and_no_preroll_false_alarm():
    r = scen("lag")
    assert "lagging" not in r["after_server_lagging"], r
    assert r["burst_tick"] == "" and r["after_burst"] == "", r          # 세그먼트 시작 직후 프리롤 한 틱은 밀림 아님
    assert r["one_tick"] == "" and r["two_ticks"] == "warn", r
    assert r["over1"] == "bad" and r["lag_banner"] is True, r
    assert r["recovered"] == "" and r["lag_banner_after"] is False, r


# ── F36: 레벨 미터 ─────────────────────────────────────────────────────────────────────────────────────
@needs_node
def test_meter_loop_stops_when_idle():
    r = scen("meter")
    assert r["idle_frames"] == 1, r                                  # 쉬는 동안은 한 번 그리고 멈춘다
    assert r["pending"] == 0 and r["disp"] <= -90, r                 # 막대가 다 내려간 뒤 멈춘다
    assert 3.0 < r["decay_s"] < 7.0, r                               # −14 dBFS → −90 at 20 dB/s ≈ 3.8 s


# ── F41: Safari 처럼 잡음 억제·자동 이득을 모르는 브라우저 ─────────────────────────────────────────────────────
@needs_node
def test_unsupported_constraints_are_disabled_and_labelled():
    r = scen("safari")
    assert r["ns_disabled"] is True and r["ns_unsup"] == "1" and r["agc_unsup"] == "1", r
    assert "미지원" in r["ns_label"] and "자동 이득·잡음 억제 포함" in r["ec_label"], r
    assert "noiseSuppression" not in r["constraints"] and "autoGainControl" not in r["constraints"], r
    assert "NS 미지원 · AGC 미지원" in r["track_info"] and "undefined" not in r["track_info"], r
    assert r["mic_disabled_running"] is True, r


# ── F40: 96 kHz 를 넘는 출력 장치 ───────────────────────────────────────────────────────────────────────
@needs_node
def test_context_above_96k_is_recreated_at_48k():
    r = scen("rate192")
    assert r["ctx_rates"] == [192000, 48000] and r["ctx_states"][0] == "closed" and r["start_rate"] == 48000, r


# ── F11: WAV 직접 해석은 soundfile(dtype=float32) 과 비트 동일 ─────────────────────────────────────────────────
@needs_node
@pytest.mark.parametrize("fmt,subtype,ch,rate", [
    ("WAV", "PCM_16", 1, 16000), ("WAV", "PCM_16", 2, 44100), ("WAV", "PCM_24", 1, 48000), ("WAV", "PCM_32", 1, 22050),
    ("WAV", "PCM_U8", 1, 8000), ("WAV", "FLOAT", 1, 48000), ("WAV", "FLOAT", 2, 96000), ("WAV", "DOUBLE", 1, 16000),
    ("WAVEX", "PCM_24", 2, 48000), ("WAVEX", "FLOAT", 1, 16000),
])
def test_parse_wav_bit_exact_with_soundfile(tmp_path, fmt, subtype, ch, rate):
    sf = pytest.importorskip("soundfile")
    rng = np.random.default_rng(7)
    n = 12345
    x = np.clip(rng.standard_normal((n, ch)) * 0.3, -1.0, 1.0)
    x[:4] = [[-1.0] * ch, [1.0] * ch, [0.0] * ch, [-0.5] * ch]       # 끝값·0 포함
    src = tmp_path / f"t-{fmt}-{subtype}-{ch}.wav"
    sf.write(str(src), x, rate, format=fmt, subtype=subtype)
    ref, sr = sf.read(str(src), dtype="float32", always_2d=True)
    ref = ref.mean(axis=1).astype(np.float32)                          # tools/client_file.py 와 같은 모노 평균
    raw = tmp_path / "out.f32"
    r = scen("parsewav", src, raw)
    assert r["ok"] and r["rate"] == sr == rate and r["ch"] == ch, r
    got = np.fromfile(str(raw), dtype="<f4")
    assert got.shape == ref.shape and np.array_equal(got, ref), float(np.max(np.abs(got - ref)))


@needs_node
def test_parse_wav_rejects_non_wav(tmp_path):
    p = tmp_path / "x.bin"; p.write_bytes(b"ID3" + b"\0" * 100)
    assert scen("parsewav", p, tmp_path / "o.f32")["ok"] is False


# ── 정적 검사 ──────────────────────────────────────────────────────────────────────────────────────────
def test_pacer_worker_served_same_origin(tmp_path, monkeypatch):
    """F34: 페이서 워커는 /static/pacer.js 로 받고, CSP(script-src 'self', worker-src 없음 → script-src 로 대체)가 허용한다."""
    import asyncio
    import sys
    sys.path.insert(0, APP)
    import server as srv
    from aiohttp.test_utils import TestClient, TestServer
    monkeypatch.setattr(srv, "HOME", str(tmp_path))
    args = srv.parse_args(["--engine", "fake", "--no-browser"])
    fake = srv._import_fake()
    s = srv.Server(args, engine_factory=lambda: fake(kv_capacity=args.kv_capacity))

    async def body():
        c = TestClient(TestServer(s.make_app()))
        await c.start_server()
        try:
            r = await c.get("/static/pacer.js")
            assert r.status == 200 and r.content_type == "text/javascript"
            csp = r.headers["Content-Security-Policy"]
            assert "script-src 'self'" in csp and "worker-src" not in csp and "child-src" not in csp
        finally:
            await c.close()
    asyncio.run(body())


def test_static_web_rules():
    js = open(JS, encoding="utf-8").read()
    css = open(os.path.join(APP, "web", "style.css"), encoding="utf-8").read()
    pacer = open(os.path.join(APP, "web", "pacer.js"), encoding="utf-8").read()
    assert "e && e.code)" not in js                                   # DOMException.code(8, 20 …) 를 서버 오류로 오인하지 않는다
    assert "new Worker('/static/pacer.js')" in js and "postMessage(d.id)" in pacer   # 같은 출처 워커 (CSP script-src 'self')
    assert "OfflineAudioContext(1, 1, 16000)" not in js               # 빠르게 경로가 브라우저 16 kHz 리샘플러를 쓰지 않는다
    assert not re.search(r"https?://(?!127\.0\.0\.1|localhost)", pacer)
    # 다크 모드: 흰 글자를 얹는 채움은 4.5:1 이상인 짙은 색
    dark = css[css.index("@media (prefers-color-scheme: dark)"):]
    assert "--accent-fill: #2563eb" in dark and "--bad-fill: #b91c1c" in dark
    assert "button.primary.on { background: var(--bad-fill)" in css
    assert "prefers-reduced-motion" in css
