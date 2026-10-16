"""WS 프로토콜·서버 스레드 테스트 (audio-ui-spec §4, §5, 수용 항목 E1–E7, H3, A4). 모델 가중치 없이 W4 의 FakeEngine 으로 돈다."""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time

import aiohttp
import numpy as np
import pytest
import soundfile as sf
from aiohttp.test_utils import TestClient, TestServer

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import server as srv  # noqa: E402

FakeEngine = srv._import_fake()


# ── 신호 ─────────────────────────────────────────────────────────────────────────────────────────────────
def tone(sec, rate, db=-24.0, f=220.0):
    t = np.arange(int(round(sec * rate))) / rate
    x = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) + 0.25 * np.sin(2 * np.pi * 3 * f * t)
    return (x * 10 ** (db / 20) / np.sqrt(np.mean(x ** 2))).astype(np.float32)


def noise(sec, rate, db=-70.0, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(round(sec * rate))) * 10 ** (db / 20)).astype(np.float32)


def script(rate, parts):
    """parts: [("n", s) | ("s", s)] → 신호."""
    return np.concatenate([noise(s, rate, seed=i) if k == "n" else tone(s, rate) for i, (k, s) in enumerate(parts)])


# ── 하네스 ───────────────────────────────────────────────────────────────────────────────────────────────
class Harness:
    def __init__(self, tmp_path, monkeypatch, delay_load=0.0, **fake_kw):
        monkeypatch.setattr(srv, "HOME", str(tmp_path))
        self.engines = []
        self.args = srv.parse_args(["--engine", "fake", "--no-browser", "--kv-capacity", str(fake_kw.pop("kv_capacity", 2048))])

        def factory():
            if delay_load: time.sleep(delay_load)
            e = FakeEngine(kv_capacity=self.args.kv_capacity, **fake_kw)
            self.engines.append(e)
            return e
        self.server = srv.Server(self.args, engine_factory=factory)

    async def __aenter__(self):
        self.client = TestClient(TestServer(self.server.make_app()))
        await self.client.start_server()
        return self

    async def __aexit__(self, *exc):
        await self.client.close()

    async def ready(self, timeout=10.0):
        t0 = time.monotonic()
        while self.server.state != "ready":
            assert time.monotonic() - t0 < timeout, f"엔진 준비 안 됨: {self.server.state} {self.server.state_msg}"
            await asyncio.sleep(0.02)

    async def ws(self):
        return await self.client.ws_connect("/ws")


async def recv(ws, timeout=10.0):
    m = await ws.receive(timeout=timeout)
    assert m.type.name == "TEXT", m
    return json.loads(m.data)


async def until(ws, pred, timeout=15.0, keep=None):
    """pred(m) 가 참인 메시지까지 받는다. keep 리스트에 모두 쌓는다."""
    t0 = time.monotonic()
    while True:
        left = timeout - (time.monotonic() - t0)
        assert left > 0, "시간 초과"
        m = await recv(ws, left)
        if keep is not None: keep.append(m)
        if pred(m): return m


async def send_pcm(ws, x, rate, pkt_ms=20, fmt="f32le", speed=0.0):
    """speed 0 = 최대 속도, 아니면 그 배속으로 보낸다."""
    n = max(1, int(round(rate * pkt_ms / 1000)))
    for i in range(0, len(x), n):
        c = x[i:i + n]
        await ws.send_bytes(c.astype("<f4").tobytes() if fmt == "f32le" else (np.clip(c, -1, 1) * 32767).astype("<i2").tobytes())
        if speed: await asyncio.sleep(len(c) / rate / speed)


def start_msg(**kw):
    m = dict(type="start", src_lang="English", mode="asr", delay=6, source="file", sample_rate=48000, format="f32le", options={})
    m.update(kw)
    return m


def check_text_consistency(evs):
    """segment_end 텍스트 == 그 세그먼트 delta 연결 (§4.3)."""
    by = {}
    for m in evs:
        if m["type"] == "delta":
            s = by.setdefault(m["seg"], ["", ""]); s[0] += m["src"]; s[1] += m["tr"]
    ends = [m for m in evs if m["type"] == "segment_end"]
    for e in ends:
        s = by.get(e["seg"], ["", ""])
        assert e["src_text"] == s[0], (e["seg"], e["src_text"], s[0])
        assert e["tr_text"] == s[1], (e["seg"], e["tr_text"], s[1])
    return ends


# ── 테스트 ───────────────────────────────────────────────────────────────────────────────────────────────
def test_static_csp_health_hello(tmp_path, monkeypatch):
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            r = await h.client.get("/")
            assert r.status == 200
            assert "default-src 'self'" in r.headers["Content-Security-Policy"]
            assert "ws://127.0.0.1:*" in r.headers["Content-Security-Policy"]
            assert r.headers["Cache-Control"] == "no-store"
            html = await r.text()
            # H3: CSP 가 막는 인라인 요소가 없어야 한다
            assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "인라인 script"
            assert "<style" not in html and not re.search(r"\sstyle=", html), "인라인 style"
            assert not re.search(r"\son[a-z]+\s*=", html), "인라인 이벤트 핸들러"
            for name, ctype in (("app.js", "text/javascript"), ("worklet.js", "text/javascript"), ("style.css", "text/css")):
                r = await h.client.get(f"/static/{name}")
                assert r.status == 200 and r.content_type == ctype and "Content-Security-Policy" in r.headers
            for bad in ("/static/..%2Fserver.py", "/static/server.py", "/static/.hidden.js", "/static/nope.js"):
                assert (await h.client.get(bad)).status == 404
            # A5: 외부 URL 없음
            for name in ("index.html", "app.js", "worklet.js", "style.css"):
                txt = open(os.path.join(APP, "web", name), encoding="utf-8").read()
                assert not re.search(r"https?://(?!127\.0\.0\.1|localhost)", txt), name
            ws = await h.ws()
            hello = await recv(ws)
            assert hello["type"] == "hello" and hello["proto"] == 1
            assert hello["latency_table"]["6"] == {"mean_ms": 576, "lo_ms": 416, "hi_ms": 736}
            assert hello["laal_table"]["Korean>English"]["8"]["approx"] is True
            assert hello["defaults"]["translate_delay"]["Korean>English"] == 8
            if hello["state"] != "ready":
                await until(ws, lambda m: m["type"] == "ready")
            hj = await (await h.client.get("/api/health")).json()
            assert hj["ok"] and hj["state"] == "ready" and hj["worker_alive"]
            assert hj["worker_qos"] in ("0x21", ) or hj["worker_qos"].startswith("n/a")
            await ws.close()
    asyncio.run(body())


def test_vad_session_48k_translate(tmp_path, monkeypatch):
    """E1/C3: 48 kHz 마이크식 입력, 두 세그먼트, delta 연결 == 최종 텍스트, dropped 0, 길이 보존."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            rate = 48000
            x = script(rate, [("n", 0.8), ("s", 2.0), ("n", 1.3), ("s", 1.5), ("n", 0.6)])
            await ws.send_json(start_msg(mode="translate", tgt_lang="German", sample_rate=rate))
            st = await until(ws, lambda m: m["type"] == "started")
            assert st["resampler"] == "soxr-HQ" and st["config"]["tgt_lang"] == "German"
            evs = []
            await send_pcm(ws, x, rate)
            await ws.send_json({"type": "stop"})
            end = await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            ends = check_text_consistency(evs)
            assert len(ends) == 2, [e["reason"] for e in ends]
            assert ends[0]["reason"] == "pause" and ends[1]["reason"] == "file_end"
            assert all(e["src_text"] and e["tr_text"] for e in ends)
            assert ends[0]["zero_tail_ms"] >= 200 and ends[0]["finalize_ms"] is not None
            starts = [m for m in evs if m["type"] == "segment_start"]
            assert [s["seg"] for s in starts] == [1, 2] and starts[0]["pre_roll_ms"] > 0
            stats = [m for m in evs if m["type"] == "stats"]
            assert stats and all(s["dropped_s"] == 0 for s in stats)
            last = stats[-1]
            assert last["final"] and abs(last["audio_rx_s"] - len(x) / rate) <= 1 / 16000 + 1e-6   # soxr 끝 flush 포함
            assert last["backlog_s"] == 0
            assert abs(last["audio_done_s"] - last["audio_rx_s"]) < 1e-3
            assert end["segments"] == 2 and end["dropped_s"] == 0
            deltas = [m for m in evs if m["type"] == "delta"]
            assert all(d["proc_lag_ms"] is None or d["proc_lag_ms"] >= 0 for d in deltas)
            # 세션 로그 (JSONL)
            lines = [json.loads(l) for l in open(os.path.join(tmp_path, "sessions", f"{st['session_id']}.jsonl"), encoding="utf-8")]
            assert [l["type"] for l in lines] == ["session_start", "segment_end", "segment_end", "session_end"]
            await ws.close()
    asyncio.run(body())


def test_eval_mode_none_16k(tmp_path, monkeypatch):
    """C5: 분할 없음 = 파일 전체 한 세그먼트 + 정확히 16000 개의 0, 이득 끔. 30 s 초과·마이크는 거절."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            x = script(16000, [("s", 2.4), ("n", 0.4), ("s", 1.0)])
            await ws.send_json(start_msg(sample_rate=16000, format="s16le",
                                         options=dict(segmentation="none", gain="auto", file_duration_s=len(x) / 16000)))
            st = await until(ws, lambda m: m["type"] == "started")
            assert st["resampler"].startswith("bypass") and st["config"]["options"]["gain"] == "off"
            evs = []
            await send_pcm(ws, x, 16000, pkt_ms=320, fmt="s16le")
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            ends = check_text_consistency(evs)
            starts = [m for m in evs if m["type"] == "segment_start"]
            assert len(ends) == 1 and len(starts) == 1 and starts[0]["reason"] == "none" and starts[0]["gain_db"] == 0
            assert ends[0]["zero_tail_ms"] == 1000.0 and ends[0]["t_start"] == 0.0
            assert ends[0]["K"] == int(round((len(x) + 16000) / 16000 * 12.5))
            # 거절
            await ws.send_json(start_msg(source="mic", options=dict(segmentation="none")))
            e = await until(ws, lambda m: m["type"] == "error"); assert e["code"] == "bad_config"
            await ws.send_json(start_msg(options=dict(segmentation="none", file_duration_s=31)))
            e = await until(ws, lambda m: m["type"] == "error"); assert e["code"] == "bad_config"
            await ws.close()
    asyncio.run(body())


def test_web_cross_refs():
    """app.js 가 찾는 id 가 index.html 에 모두 있고, 워크릿 경로·프로세서 이름이 맞는다 (정적, 서버 없음)."""
    rd = lambda n: open(os.path.join(APP, "web", n), encoding="utf-8").read()  # noqa: E731
    html, js, wk = rd("index.html"), rd("app.js"), rd("worklet.js")
    ids = set(re.findall(r'\sid="([^"]+)"', html))
    used = set(re.findall(r"\$\('([^']+)'\)", js)) | set(re.findall(r"setStat\('([^']+)'", js))
    for lst in re.findall(r"for \(const (?:id|\[id, k\]) of \[(.*?)\]\)", js, re.S):
        used |= set(re.findall(r"'((?:opt|st|btn)-[a-z-]+)'", lst))
    assert used and not (used - ids), sorted(used - ids)
    assert re.findall(r'<script type="module" src="([^"]+)"', html) == ["/static/app.js"]
    assert re.findall(r"addModule\('([^']+)'\)", js) == ["/static/worklet.js"]
    assert re.findall(r"registerProcessor\('([^']+)'", wk) == re.findall(r"new AudioWorkletNode\(ctx, '([^']+)'", js) == ["capture"]


def test_bad_config_and_not_ready(tmp_path, monkeypatch):
    async def body():
        async with Harness(tmp_path, monkeypatch, delay_load=0.8) as h:
            ws = await h.ws()
            hello = await recv(ws)
            assert hello["state"] == "loading"
            await ws.send_json(start_msg())
            e = await until(ws, lambda m: m["type"] == "error"); assert e["code"] == "model_not_ready"
            await until(ws, lambda m: m["type"] == "ready")
            for bad in (dict(src_lang="French"), dict(mode="translate", src_lang="Korean", tgt_lang="German"),
                        dict(delay=3), dict(sample_rate=4000), dict(format="mp3"), dict(options=dict(tail_s=0.5), delay=10),
                        dict(options=dict(end_pause_ms=100)), dict(options=dict(hard_max_s=40)), dict(source="tv")):
                await ws.send_json(start_msg(**bad))
                e = await until(ws, lambda m: m["type"] in ("error", "started"))
                assert e["type"] == "error" and e["code"] == "bad_config", (bad, e)
            await ws.send_str("not json")
            e = await until(ws, lambda m: m["type"] == "error"); assert e["code"] == "bad_config"
            await ws.close()
    asyncio.run(body())


def test_busy_takeover_preempt(tmp_path, monkeypatch):
    """E4: 두 번째 탭은 busy, takeover 하면 첫 탭은 preempted + 중단된 session_end."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            a = await h.ws(); await recv(a)
            b = await h.ws(); await recv(b)
            await a.send_json(start_msg(source="mic"))
            await until(a, lambda m: m["type"] == "started")
            await b.send_json(start_msg(source="mic"))
            e = await until(b, lambda m: m["type"] == "error"); assert e["code"] == "busy" and not e["fatal"]
            await b.send_json(start_msg(source="mic", takeover=True))
            await until(b, lambda m: m["type"] == "started")
            e = await until(a, lambda m: m["type"] == "error"); assert e["code"] == "preempted" and e["fatal"]
            end = await until(a, lambda m: m["type"] == "session_end"); assert end["aborted"] and end["reason"] == "preempted"
            await b.send_json({"type": "stop"})
            await until(b, lambda m: m["type"] == "session_end")
            await a.close(); await b.close()
    asyncio.run(body())


def test_abort_fast_and_binary_outside(tmp_path, monkeypatch):
    """E3: abort 는 300 ms 안에 idle, 버린 양 보고. 세션 밖 바이너리는 경고 한 번."""
    async def body():
        async with Harness(tmp_path, monkeypatch, round_ms=25) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            await ws.send_bytes(b"\0" * 400); await ws.send_bytes(b"\0" * 400)
            w = await until(ws, lambda m: m["type"] == "warning")
            assert w["code"] == "binary_outside_session" and w["bytes"] == 400
            rate = 16000
            await ws.send_json(start_msg(sample_rate=rate))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(rate, [("n", 0.6), ("s", 4.0)]), rate, pkt_ms=320)
            await asyncio.sleep(0.3)
            t0 = time.monotonic()
            await ws.send_json({"type": "abort"})
            evs = []
            end = await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            dt = time.monotonic() - t0
            assert dt < 0.3, dt
            assert end["aborted"] and end["discarded_s"] > 0
            assert [m["reason"] for m in evs if m["type"] == "segment_end"] == ["aborted"]
            assert not [m for m in evs if m["type"] == "warning" and m["code"] == "binary_outside_session"]
            # 새 세션이 곧바로 열린다 (워커가 버린 명령을 지나쳐 간다)
            await ws.send_json(start_msg(sample_rate=rate))
            await until(ws, lambda m: m["type"] == "started")
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end")
            await ws.close()
    asyncio.run(body())


def test_update_applies_next_segment(tmp_path, monkeypatch):
    """E5: 세션 중 update 는 다음 segment_start.config 부터."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            rate = 16000
            await ws.send_json(start_msg(sample_rate=rate))
            await until(ws, lambda m: m["type"] == "started")
            evs = []
            await send_pcm(ws, script(rate, [("n", 0.8), ("s", 1.2)]), rate)
            await until(ws, lambda m: m["type"] == "segment_start", keep=evs)
            await ws.send_json({"type": "update", "delay": 8, "mode": "translate", "tgt_lang": "German", "options": {"end_pause_ms": 600}})
            await send_pcm(ws, script(rate, [("s", 0.5), ("n", 1.3), ("s", 1.2), ("n", 0.5)]), rate)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            up = [m for m in evs if m["type"] == "updated"]
            assert up and up[0]["pending"]["delay"] == 8
            cfgs = [m["config"] for m in evs if m["type"] == "segment_start"]
            assert len(cfgs) == 2
            assert cfgs[0] == {"src_lang": "English", "mode": "asr", "tgt_lang": None, "delay": 6}
            assert cfgs[1] == {"src_lang": "English", "mode": "translate", "tgt_lang": "German", "delay": 8}
            ends = check_text_consistency(evs)
            assert ends[0]["tr_text"] == "" and ends[1]["tr_text"]
            await ws.send_json({"type": "update", "delay": 3})     # 세션 없음 → 오류
            e = await until(ws, lambda m: m["type"] == "error"); assert e["code"] == "bad_config"
            await ws.close()
    asyncio.run(body())


def test_asr_drops_trans_and_specials(tmp_path, monkeypatch):
    """E7: 전사 모드에서 <TRANS> 구간은 tr 로 가지 않고 tr_dropped 로 센다. <SEM_END>·미사용 id 는 글자를 내지 않는다."""
    async def body():
        async with Harness(tmp_path, monkeypatch, asr_trans=True, specials=True, words_every=2) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(sample_rate=16000, options=dict(debug_tokens=True)))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.7), ("s", 3.0), ("n", 0.3)]), 16000, pkt_ms=320)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            ends = check_text_consistency(evs)
            assert ends and all(e["tr_text"] == "" for e in ends)
            assert sum(e["n_tokens"]["tr_dropped"] for e in ends) > 0
            ds = [m for m in evs if m["type"] == "delta"]
            assert all(d["tr"] == "" for d in ds)
            ids = [i for d in ds for i in d.get("ids", [])]
            assert 151726 in ids and 151723 in ids                 # 디버그에는 보인다
            txt = "".join(e["src_text"] for e in ends)
            assert txt and "<" not in txt and "�" not in txt
    asyncio.run(body())


def test_hangul_split_tokens_never_broken(tmp_path, monkeypatch):
    """E6: 음절이 토큰 경계에 걸쳐도 delta 에 깨진 글자가 없다."""
    async def body():
        async with Harness(tmp_path, monkeypatch, hangul=True, words_every=1) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(src_lang="Korean", mode="translate", tgt_lang="English", delay=8, sample_rate=16000))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.7), ("s", 2.5), ("n", 0.3)]), 16000, pkt_ms=20)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            ends = check_text_consistency(evs)
            for d in (m for m in evs if m["type"] == "delta"):
                assert "�" not in d["src"] + d["tr"], d
            src = "".join(e["src_text"] for e in ends)
            assert re.search(r"[가-힣]{2,}", src), src
            assert all(e["replaced_bytes"] == 0 for e in ends)
    asyncio.run(body())


def test_lagging_backlog_recovers(tmp_path, monkeypatch):
    """E2: 라운드 지연 주입(RTF > 1) → backlog 증가·lagging 경고·버림 없음, 지연 해제 후 회복."""
    async def body():
        async with Harness(tmp_path, monkeypatch, round_ms=120) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            await ws.send_json(start_msg(source="mic", sample_rate=16000))
            sid = (await until(ws, lambda m: m["type"] == "started"))["session_id"]
            evs = []
            await send_pcm(ws, script(16000, [("n", 0.6), ("s", 6.0)]), 16000, pkt_ms=320)
            w = await until(ws, lambda m: m["type"] == "warning" and m["code"] == "lagging", keep=evs, timeout=5)
            assert w["sid"] == sid                                 # 세션 경고는 sid 를 단다 (같은 소켓의 다음 세션과 구분)
            peak = max(m["backlog_s"] for m in evs if m["type"] == "stats")
            assert peak > 2
            h.engines[0].round_ms = 0.0                            # 지연 해제
            s = await until(ws, lambda m: m["type"] == "stats" and m["backlog_s"] < 0.05, keep=evs, timeout=10)
            assert s["dropped_s"] == 0
            await ws.send_json({"type": "stop"})
            end = await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            assert end["backlog_max_s"] > 2 and end["dropped_s"] == 0
            check_text_consistency(evs)
    asyncio.run(body())


def test_silence_does_not_stall_flow_control(tmp_path, monkeypatch):
    """E1 [critic]: 15 s 무음 구간이 있어도 audio_done_s 가 따라와 빠른 전송이 멈추지 않는다."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            await ws.send_json(start_msg(sample_rate=16000, options=dict(speed=0)))
            await until(ws, lambda m: m["type"] == "started")
            x = script(16000, [("n", 0.8), ("s", 1.5), ("n", 15.0)])
            await send_pcm(ws, x, 16000, pkt_ms=320)
            sent = len(x) / 16000
            s = await until(ws, lambda m: m["type"] == "stats" and m["audio_rx_s"] >= sent - 0.05, timeout=5)
            # 무음은 건너뜀으로 세어져 done 이 보낸 양을 바짝 따라온다 (차이 ≤ VAD 보류 ≈ pre-roll + onset)
            s = await until(ws, lambda m: m["type"] == "stats" and sent - m["audio_done_s"] < 1.0, timeout=5)
            assert s["audio_skipped_s"] > 10 and "throughput_x" in s
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end")
    asyncio.run(body())


def test_kv_guard_forces_segment_end(tmp_path, monkeypatch):
    """§1.6: 폭주로 KV 가 0.85 를 넘으면 kv_guard 로 세그먼트를 끝내고 바로 이어 연다(샘플 손실 없음)."""
    async def body():
        async with Harness(tmp_path, monkeypatch, runaway=True, kv_capacity=256) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(sample_rate=16000))
            await until(ws, lambda m: m["type"] == "started")
            x = script(16000, [("n", 0.6), ("s", 3.0), ("n", 1.0)])
            await send_pcm(ws, x, 16000, pkt_ms=20, speed=10)       # 결정은 워커 결과를 본 뒤 → 세그먼트 도중에 와야 한다
            await ws.send_json({"type": "stop"})
            end = await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            reasons = [m["reason"] for m in evs if m["type"] == "segment_end"]
            assert "kv_guard" in reasons, reasons
            assert any(m["type"] == "warning" and m["code"] == "kv_guard" for m in evs)
            assert any(m["type"] == "segment_start" and m["reason"] == "kv" for m in evs)
            assert all(m["positions"] <= 256 for m in evs if m["type"] == "segment_end")
            last = [m for m in evs if m["type"] == "stats"][-1]
            assert abs(last["audio_rx_s"] - len(x) / 16000) < 1e-6 and end["dropped_s"] == 0
    asyncio.run(body())


def test_final_redecode(tmp_path, monkeypatch):
    """§7: final 재디코드(옵션)는 segment_end 뒤에 segment_final 로 온다."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(mode="translate", tgt_lang="Korean", delay=10, sample_rate=16000, options=dict(final_redecode=True)))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.7), ("s", 2.0), ("n", 1.2), ("s", 1.0)]), 16000, pkt_ms=20)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            ends = {m["seg"]: m for m in evs if m["type"] == "segment_end"}
            fin = [m for m in evs if m["type"] == "segment_final"]
            assert len(fin) == len(ends) == 2
            for f in fin:
                assert "skipped" not in f, f
                assert f["src"] == ends[f["seg"]]["src_text"] and f["tr"]
            lines = [json.loads(l)["type"] for l in open(os.path.join(tmp_path, "sessions", f"{ends[1]['sid']}.jsonl"), encoding="utf-8")]
            assert lines.count("segment_final") == 2
    asyncio.run(body())


def test_final_redecode_stepwise_interleaves(tmp_path, monkeypatch):
    """§5.2: final 재디코드는 한 걸음씩(FinalJob.step) 돌며 다음 세그먼트 스트리밍 사이에 끼어든다 — 세그먼트 1 의 final 이
    세그먼트 2 가 끝나기 전에 온다. 걸음 수는 id 수와 같고(가짜 작업은 걸음당 id 하나), 결과는 블로킹 final 과 같다."""
    steps = []
    orig = srv._import_fake()
    import engine.fake as fk
    real_step = fk.FakeFinalJob.step

    def spy(self):
        steps.append(self); return real_step(self)                     # 객체를 붙잡아 둬야 id 가 재사용되지 않는다
    monkeypatch.setattr(fk.FakeFinalJob, "step", spy)

    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(sample_rate=16000, options=dict(final_redecode=True)))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.7), ("s", 2.0), ("n", 1.2), ("s", 4.0), ("n", 0.3)]), 16000, pkt_ms=20, speed=8)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            order = [(m["type"], m["seg"]) for m in evs if m["type"] in ("segment_final", "segment_end")]
            assert order.index(("segment_final", 1)) < order.index(("segment_end", 2)), order
            ends = {m["seg"]: m for m in evs if m["type"] == "segment_end"}
            fin = {m["seg"]: m for m in evs if m["type"] == "segment_final"}
            assert set(fin) == set(ends) == {1, 2} and all("skipped" not in f for f in fin.values())
            for s, f in fin.items():
                assert f["src"] == ends[s]["src_text"]
            assert len(steps) >= 2 and len({id(j) for j in steps}) == 2   # 작업 두 개, 여러 걸음
    assert orig is not None
    asyncio.run(body())


def test_nonfinite_warning(tmp_path, monkeypatch):
    """저정밀 thinker 의 유한성 검사 실패(누계 증가)는 세션에 nonfinite 경고 한 번으로 알린다 (thinker-spec §6.2)."""
    import types

    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            h.engines[0].thinker = types.SimpleNamespace(nonfinite=0)
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(sample_rate=16000))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.7), ("s", 1.0)]), 16000)
            h.engines[0].thinker.nonfinite = 2
            await send_pcm(ws, script(16000, [("s", 1.0), ("n", 1.0)]), 16000)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            w = [m for m in evs if m["type"] == "warning" and m["code"] == "nonfinite"]
            assert len(w) == 1 and w[0]["count"] == 2 and "fp32" in w[0]["message"]
            assert h.server.nonfinite_seen == 2
    asyncio.run(body())


def test_disconnect_aborts_and_worker_thread(tmp_path, monkeypatch):
    """소켓이 닫히면 세션은 중단되고 엔진은 워커 스레드에서만 쓰인다 (FakeEngine check_thread)."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            await ws.send_json(start_msg(sample_rate=16000))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.6), ("s", 1.0)]), 16000)
            await ws.close()
            for _ in range(100):
                if not h.server.sessions: break
                await asyncio.sleep(0.02)
            assert not h.server.sessions and h.server.active is None
            assert h.server.worker.thread.is_alive() and h.server.worker.ident != __import__("threading").get_ident()
            ws = await h.ws(); hello = await recv(ws)
            assert hello["busy"] is False
            await ws.send_json({"type": "ping", "t": 5})
            p = await until(ws, lambda m: m["type"] == "pong"); assert p["t"] == 5
    asyncio.run(body())


def test_port_selection():
    assert srv.candidate_ports(None) == list(range(8796, 8806))
    with pytest.raises(SystemExit):
        srv.candidate_ports(8790)
    with pytest.raises(SystemExit):
        srv.candidate_ports(8765)
    sock, port = srv.bind_port("127.0.0.1", None, in_use=lambda h, p: p < 8800)
    try:
        assert 8800 <= port <= 8805 and port not in (8765, 8790)
    finally:
        sock.close()


def test_seg_params_derivation():
    p = srv.seg_params(dict(srv.DEFAULT_OPTIONS, hard_max_s=10)).check()
    assert p.soft_max_s == 6.0 and p.hold_from_s == pytest.approx(9.296)   # C9; hold_from = hard − hold 640 ms − 2 프레임 (F33)
    p = srv.seg_params(dict(srv.DEFAULT_OPTIONS)).check()
    assert p.soft_max_s == 18.0 and p.hold_from_s == pytest.approx(24.296)
    assert srv.seg_params(dict(srv.DEFAULT_OPTIONS)).hold_from_s is None      # 서버는 파이프라인 기본식에 맡긴다


def test_hidden_noise_segments(tmp_path, monkeypatch):
    """§1.4 잡음 세그먼트: 텍스트 없는 세그먼트는 hidden (세션 로그에는 남는다). final 재디코드를 켜면 유성 6 프레임 이상은
    남긴다 (재디코드 텍스트를 보일 줄; 정확히 6 프레임은 보임). gain off, fake 엔진은 −55 dBFS 를 무성으로 본다."""
    async def run(h, final):
        ws = await h.ws(); await recv(ws)
        await ws.send_json(start_msg(sample_rate=16000, options=dict(gain="off", final_redecode=final)))
        st = await until(ws, lambda m: m["type"] == "started")
        parts = [noise(1.0, 16000, seed=1)]
        for i, ms in enumerate((128, 160, 192, 320)):
            parts += [noise(ms / 1000, 16000, db=-55.0, seed=10 + i), noise(1.5, 16000, seed=20 + i)]
        parts += [tone(1.5, 16000), noise(1.5, 16000, seed=30)]
        await send_pcm(ws, np.concatenate(parts), 16000, pkt_ms=320)
        await ws.send_json({"type": "stop"})
        evs = []
        await until(ws, lambda m: m["type"] == "session_end", keep=evs)
        await ws.close()
        lines = [json.loads(l) for l in open(os.path.join(tmp_path, "sessions", f"{st['session_id']}.jsonl"), encoding="utf-8")]
        return check_text_consistency(evs), lines

    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ends, lines = await run(h, False)
            assert len(ends) >= 5 and ends[-1]["src_text"], [(e["voiced_s"], e["src_text"]) for e in ends]
            assert all(e["hidden"] == (not e["src_text"] and not e["tr_text"]) for e in ends)
            assert any(e["hidden"] and round(e["voiced_s"] / 0.032) >= 6 for e in ends)     # 유성이 길어도 글자가 없으면 숨김
            assert not ends[-1]["hidden"]
            assert len([l for l in lines if l["type"] == "segment_end"]) == len(ends)        # 숨긴 것도 로그에는 남는다
            ends, _ = await run(h, True)
            empty = [e for e in ends if not e["src_text"] and not e["tr_text"]]
            assert all(e["hidden"] == (round(e["voiced_s"] / 0.032) < 6) for e in empty)
            assert any(e["hidden"] for e in empty) and any(not e["hidden"] for e in empty)
    asyncio.run(body())


def test_restart_same_socket_and_save_audio(tmp_path, monkeypatch):
    """같은 소켓의 두 번째 start = stop 후 start (이전 세션은 정상 마무리). save_audio 는 입력 wav 를 남긴다."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            rate = 48000
            await ws.send_json(start_msg(sample_rate=rate, options=dict(save_audio=True)))
            st1 = await until(ws, lambda m: m["type"] == "started")
            x = script(rate, [("n", 0.7), ("s", 1.5), ("n", 0.2)])
            await send_pcm(ws, x, rate)
            await ws.send_json(start_msg(sample_rate=16000))          # stop → start
            evs = []
            st2 = await until(ws, lambda m: m["type"] == "started", keep=evs)
            end1 = await until(ws, lambda m: m["type"] == "session_end" and m["session_id"] == st1["session_id"], keep=evs)
            assert st2["session_id"] != st1["session_id"] and not end1.get("aborted")
            ends = [m for m in evs if m["type"] == "segment_end"]
            assert ends and ends[-1]["reason"] == "file_end" and ends[-1]["sid"] == st1["session_id"]
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end" and m["session_id"] == st2["session_id"])
            import soundfile as sf
            y, sr = sf.read(os.path.join(tmp_path, "sessions", f"{st1['session_id']}.wav"), dtype="float32")
            assert sr == rate and len(y) == len(x) and np.allclose(y, x)
    asyncio.run(body())


def test_disconnect_logs_open_segment(tmp_path, monkeypatch):
    """§5.1: 소켓이 닫히면(탭 닫기·새로고침) 세션은 중단되지만, 열린 세그먼트의 이미 보인 텍스트는 세션 로그에
    segment_end(aborted) 로 남는다 — 탭이 사라진 뒤에는 JSONL 이 유일한 기록."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            await ws.send_json(start_msg(source="mic", sample_rate=16000))
            st = await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.6), ("s", 2.0)]), 16000, pkt_ms=320)
            d = await until(ws, lambda m: m["type"] == "delta" and m["src"])
            await ws.close()
            for _ in range(100):
                if not h.server.sessions: break
                await asyncio.sleep(0.02)
            assert not h.server.sessions
            lines = [json.loads(l) for l in open(os.path.join(tmp_path, "sessions", f"{st['session_id']}.jsonl"), encoding="utf-8")]
            ends = [l for l in lines if l["type"] == "segment_end"]
            assert len(ends) == 1 and ends[0]["reason"] == "aborted" and d["src"].strip() in ends[0]["src_text"], ends
            assert lines[-1]["type"] == "session_end" and lines[-1]["reason"] == "disconnected" and lines[-1]["segments"] == 1
    asyncio.run(body())


def test_token_streaming_partials(tmp_path, monkeypatch):
    """토큰 콜백: 토큰은 확정되는 즉시 partial delta 로 간다 (feed()/finish() 가 끝나기를 기다리지 않는다).
    partial + Round 나머지 = 엔진이 낸 (k, id) 전체·순서 그대로, delta 연결 = segment_end 텍스트, 한글 바이트가
    토큰 사이에 갈라져도 깨진 글자 없음, <TRANS> 전환 포함."""
    decs = []
    orig_new = FakeEngine.new_segment

    def new_segment(self, *a, **kw):
        d = orig_new(self, *a, **kw); decs.append(d); return d
    monkeypatch.setattr(FakeEngine, "new_segment", new_segment)

    async def body():
        async with Harness(tmp_path, monkeypatch, hangul=True, words_every=1) as h:
            await h.ready(); decs.clear()                            # 워밍업 세그먼트는 뺀다
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(src_lang="Korean", mode="translate", tgt_lang="English", delay=8, sample_rate=16000,
                                         options=dict(debug_tokens=True)))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.7), ("s", 2.5), ("n", 1.3), ("s", 1.5), ("n", 0.3)]), 16000, pkt_ms=320)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            ends = check_text_consistency(evs)
            assert len(ends) == 2 and len(decs) == 2
            ds = [m for m in evs if m["type"] == "delta"]
            got = [(m["seg"], m["k"], i) for m in ds for i in m.get("ids", [])]
            want = [(n + 1, k, t) for n, d in enumerate(decs) for k, t in d.out]
            assert got == want and want
            part = [m for m in ds if m.get("partial")]
            assert len(part) == len(want) and all(len(m["ids"]) == 1 for m in part)          # 토큰마다 하나
            assert all(m["ids"] == [] for m in ds if not m.get("partial") and not m.get("final"))   # 라운드는 나머지(없음)만
            assert 151726 in [m["ids"][0] for m in part]
            assert all("�" not in m["src"] + m["tr"] for m in ds)
            assert all(e["replaced_bytes"] == 0 and e["src_text"] and e["tr_text"] for e in ends)
            assert re.search(r"[가-힣]{2,}", ends[0]["src_text"])
    asyncio.run(body())


def test_proc_lag_measured_at_send(tmp_path, monkeypatch):
    """proc_lag 는 결과가 실제로 나가는 순간 기준: feed() 한 번이 라운드 4 개를 내면 앞 라운드도 호출이 끝난 뒤에 나가므로
    옛 추정(계산 끝 = compute_lag_ms)보다 크거나 같고 라운드 시간만큼 차이 난다. 토큰 partial 은 그 라운드보다 먼저·작은 지연."""
    async def body():
        async with Harness(tmp_path, monkeypatch, words_every=1) as h:
            await h.ready()
            h.engines[0].round_ms = 30.0
            ws = await h.ws(); await recv(ws)
            evs = []
            await ws.send_json(start_msg(sample_rate=16000, options=dict(debug_tokens=True)))
            await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, script(16000, [("n", 0.6), ("s", 2.5), ("n", 0.4)]), 16000, pkt_ms=320, speed=1)
            await ws.send_json({"type": "stop"})
            await until(ws, lambda m: m["type"] == "session_end", keep=evs)
            t_end = {m["seg"]: m["t_end"] for m in evs if m["type"] == "segment_end"}
            rd = [m for m in evs if m["type"] == "delta" and not m.get("partial") and not m.get("final")]
            assert rd and all(m["proc_lag_ms"] + 0.2 >= m["compute_lag_ms"] for m in rd)
            assert max(m["proc_lag_ms"] - m["compute_lag_ms"] for m in rd) >= 40, [(m["proc_lag_ms"], m["compute_lag_ms"]) for m in rd]
            part = {(m["seg"], m["k"]): m for m in evs if m["type"] == "delta" and m.get("partial")}
            both = [(part[(m["seg"], m["k"])], m) for m in rd if (m["seg"], m["k"]) in part and m["t_avail"] < t_end[m["seg"]] - 0.3]
            assert both and all(p["proc_lag_ms"] <= m["proc_lag_ms"] + 0.2 for p, m in both)
            assert sum(p["proc_lag_ms"] + 20 < m["proc_lag_ms"] for p, m in both) >= len(both) // 4     # 첫 라운드 토큰은 훨씬 먼저
    asyncio.run(body())


def test_stop_finals_yield_to_other_session(tmp_path, monkeypatch):
    """§5.2: stop 이 그 세션의 final 을 돌리는 중 다른 탭이 스트리밍을 시작하면, 남은 final 은 lagging 으로 건너뛰고
    새 세션은 기다리지 않는다 (예전엔 final 이 끝날 때까지 수 초 막혔다)."""
    import engine.fake as fk
    real_step = fk.FakeFinalJob.step

    def slow_step(self):
        time.sleep(0.06); return real_step(self)                    # 토큰 forward 흉내 (60 ms/걸음)
    monkeypatch.setattr(fk.FakeFinalJob, "step", slow_step)

    async def body():
        async with Harness(tmp_path, monkeypatch, words_every=1) as h:
            await h.ready()
            a = await h.ws(); await recv(a)
            b = await h.ws(); await recv(b)
            await a.send_json(start_msg(sample_rate=16000, options=dict(final_redecode=True, speed=0)))
            await until(a, lambda m: m["type"] == "started")
            ea = []
            await send_pcm(a, script(16000, [("n", 0.6), ("s", 3.0), ("n", 0.3)]), 16000, pkt_ms=320)
            await a.send_json({"type": "stop"})
            await until(a, lambda m: m["type"] == "segment_end", keep=ea)          # A 의 마지막 세그먼트 끝 → 워커는 곧 stop(final)
            await b.send_json(start_msg(source="mic", sample_rate=16000))
            await until(b, lambda m: m["type"] == "started")
            await send_pcm(b, script(16000, [("n", 0.6), ("s", 2.0), ("n", 0.3)]), 16000, pkt_ms=20, speed=1)
            await b.send_json({"type": "stop"})
            endb = await until(b, lambda m: m["type"] == "session_end", timeout=15)
            await until(a, lambda m: m["type"] == "session_end", keep=ea)
            fin = [m for m in ea if m["type"] == "segment_final"]
            assert fin and fin[-1].get("skipped") == "lagging", fin
            assert endb["proc_lag_p95_ms"] is not None and endb["proc_lag_p95_ms"] < 400, endb
    asyncio.run(body())


def test_client_file_exits_when_preempted_in_flow_wait(tmp_path, monkeypatch):
    """client_file: 흐름 제어로 기다리는 중 서버가 세션을 끝내면(다른 탭 takeover) 멈추지 않고 1 로 끝난다."""
    sys.path.insert(0, os.path.join(APP, "tools"))
    import client_file as cf
    wav = os.path.join(tmp_path, "tone.wav"); sf.write(wav, tone(20.0, 16000), 16000)

    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            h.engines[0].round_ms = 100.0                               # RTF 1.25 → audio_done_s 가 보낸 양을 못 따라온다
            url = str(h.client.make_url("/ws")).replace("http", "ws", 1)
            a = cf.parse_args([wav, "--src", "English", "--speed", "0", "--packet-ms", "2000", "--max-inflight", "2", "--url", url, "--quiet"])
            task = asyncio.ensure_future(cf.main_async(a))
            for _ in range(250):
                if h.server.active is not None: break
                await asyncio.sleep(0.02)
            assert h.server.active is not None
            await asyncio.sleep(1.0)                                    # 흐름 제어 대기에 들어간다
            assert not task.done()
            b = await h.ws(); await recv(b)
            await b.send_json(start_msg(source="mic", sample_rate=16000, takeover=True))
            await until(b, lambda m: m["type"] == "started")
            t0 = time.monotonic()
            code = await asyncio.wait_for(task, 5.0)
            assert code == 1 and time.monotonic() - t0 < 2.0
            await b.send_json({"type": "stop"})
            await until(b, lambda m: m["type"] == "session_end")
    asyncio.run(body())


def test_quiet_input_needs_speech(tmp_path, monkeypatch):
    """최대 이득 경고(quiet_input)는 지금 열린 세그먼트가 말소리일 때만: 세그먼트가 없거나(지난 이득만 남음)
    잡음뿐(글자 없음)이면 내지 않는다."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            ws = await h.ws(); await recv(ws)
            await ws.send_json(start_msg(source="mic", sample_rate=16000))
            await until(ws, lambda m: m["type"] == "started")
            s = next(iter(h.server.sessions.values()))
            sent = []
            monkeypatch.setattr(s, "send", sent.append)
            snap = dict(t=1.0, level_db=-80.0, floor_db=-90.0, state="silence", gain_db=30.0, seg=None)
            monkeypatch.setattr(s.pipe, "vad_snapshot", lambda: dict(snap))
            quiet = lambda: [m for m in sent if m.get("code") == "quiet_input"]  # noqa: E731
            s.tick_vad(); assert not quiet()                                     # 세그먼트 없음: 지난 +30 dB 만 남은 조용한 방
            ss = srv.SegState(7, dict(s.cur), {"t_start": 0.0}, None, 0); s.segs[7] = ss
            snap.update(seg=7, state="speech", level_db=-60.0)
            s.tick_vad(); assert not quiet()                                     # 열렸지만 글자 없음 = 잡음
            ss.src_parts += ["hello", " world"]
            s.tick_vad(); s.tick_vad()
            assert len(quiet()) == 1 and "최대 이득" in quiet()[0]["message"]    # 말소리 + 최대 이득 → 한 번 (30 s 마다)
            del s.segs[7]
            await ws.close()
    asyncio.run(body())


# ── 종료·신호·포트 (F26–F28) ─────────────────────────────────────────────────────────────────────────────────
def _free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0)); return so.getsockname()[1]


def _spawn(tmp_path, *extra):
    """별도 프로세스 서버 (fake). 주소 줄이 나올 때까지 기다린다 → (proc, port, 지금까지 출력)."""
    port = _free_port()
    env = dict(os.environ, VAPKT_APP_HOME=str(tmp_path), PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, os.path.join(APP, "server.py"), "--engine", "fake", "--no-browser", "--port", str(port), *extra],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
    out, t0 = [], time.monotonic()
    while time.monotonic() - t0 < 30:
        line = p.stdout.readline()
        if not line: break
        out.append(line)
        if "VAPKT 실시간 전사·번역:" in line: return p, port, out
    p.kill(); raise AssertionError("".join(out))


def test_sigint_during_load_exits_fast(tmp_path):
    """적재 중 Ctrl-C: 적재가 끝나기(또는 5 s join)를 기다리지 않고 곧 0 으로 끝나며, 종료 뒤 '준비됨' 을 알리거나 찍지 않는다."""
    p, port, out = _spawn(tmp_path, "--fake-spec", "load_ms=4000,warmup_ms=1000")

    async def body():
        async with aiohttp.ClientSession() as cs:
            ws = await cs.ws_connect(f"http://127.0.0.1:{port}/ws")
            hello = await recv(ws, 5)
            assert hello["state"] == "loading"
            await asyncio.sleep(0.3)
            p.send_signal(signal.SIGINT); t0 = time.monotonic()
            msgs = []
            while True:
                m = await ws.receive(timeout=8)
                if m.type != aiohttp.WSMsgType.TEXT: break
                msgs.append(json.loads(m.data))
            rc = await asyncio.get_running_loop().run_in_executor(None, p.wait, 8)
            return rc, time.monotonic() - t0, msgs
    try:
        rc, dt, msgs = asyncio.run(body())
    finally:
        if p.poll() is None: p.kill()
    rest = "".join(out) + p.stdout.read()
    assert rc == 0 and dt < 1.5, (rc, dt, rest)
    assert not [m for m in msgs if m["type"] in ("ready", "hello")], msgs
    assert "종료 중" in rest and "엔진 준비" not in rest.split("종료 중")[1], rest


def test_sighup_closes_session_log_and_wav(tmp_path):
    """SIGHUP(터미널 창 닫기)도 정상 종료: 열린 세그먼트·session_end(server_shutdown) 가 로그에 남고 --save-audio wav 가 온전하다."""
    p, port, out = _spawn(tmp_path, "--save-audio")
    x = script(16000, [("n", 0.6), ("s", 2.0)])

    async def body():
        async with aiohttp.ClientSession() as cs:
            ws = await cs.ws_connect(f"http://127.0.0.1:{port}/ws")
            hello = await recv(ws, 5)
            if hello["state"] != "ready": await until(ws, lambda m: m["type"] == "ready")
            await ws.send_json(start_msg(sample_rate=16000))
            st = await until(ws, lambda m: m["type"] == "started")
            await send_pcm(ws, x, 16000, pkt_ms=320)
            got = []
            await until(ws, lambda m: m["type"] == "stats" and m["audio_rx_s"] >= len(x) / 16000 - 1e-3, keep=got)
            if not any(m["type"] == "delta" and m["src"] for m in got):
                await until(ws, lambda m: m["type"] == "delta" and m["src"])
            p.send_signal(signal.SIGHUP)
            end = await until(ws, lambda m: m["type"] == "session_end")
            rc = await asyncio.get_running_loop().run_in_executor(None, p.wait, 10)
            return st["session_id"], end, rc
    try:
        sid, end, rc = asyncio.run(body())
    finally:
        if p.poll() is None: p.kill()
    assert rc == 0 and end["reason"] == "server_shutdown", (rc, end, "".join(out) + p.stdout.read())
    lines = [json.loads(l) for l in open(os.path.join(tmp_path, "sessions", f"{sid}.jsonl"), encoding="utf-8")]
    assert [l["type"] for l in lines][-2:] == ["segment_end", "session_end"] and lines[-1]["reason"] == "server_shutdown"
    y, sr = sf.read(os.path.join(tmp_path, "sessions", f"{sid}.wav"), dtype="float32")
    assert sr == 16000 and len(y) == len(x) and np.allclose(y, x)


def test_wav_header_readable_before_close(tmp_path):
    """--save-audio wav 는 닫기 전에도(프로세스가 정리 없이 죽어도) 1 s 단위로 헤더 길이가 맞아 읽힌다."""
    path = os.path.join(tmp_path, "a.wav")
    w = srv.WavF32(path, 16000)
    x = tone(2.5, 16000)
    for i in range(0, len(x), 320): w.write(x[i:i + 320])
    y, sr = sf.read(path, dtype="float32")
    assert sr == 16000 and 2 * 16000 <= len(y) <= len(x) and np.allclose(y, x[:len(y)])
    w.close()
    y, _ = sf.read(path, dtype="float32")
    assert len(y) == len(x)


def test_say_survives_dead_tty(monkeypatch):
    """터미널이 닫혀 stdout 쓰기가 EIO 를 내도 say 는 예외를 내지 않는다 (종료 정리가 건너뛰어지지 않게)."""
    class Dead:
        def write(self, *_): raise OSError(5, "Input/output error")
        def flush(self): raise OSError(5, "Input/output error")
    monkeypatch.setattr(sys, "stdout", Dead()); monkeypatch.setattr(sys, "stderr", Dead())
    srv.say("종료 중…"); srv.say("x", err=True)


def test_second_signal_exits_now():
    """첫 신호는 종료를 시작하고, 종료 중 두 번째 신호(Ctrl-C 두 번)는 기다리지 않고 130 으로 끝낸다."""
    calls = []

    class S:
        closing = threading.Event()
    stop = asyncio.Event()
    on_sig = srv.make_signal_handler(S, stop, hard_exit=calls.append)
    on_sig()
    assert S.closing.is_set() and stop.is_set() and calls == []
    on_sig()
    assert calls == [130]
    hup = srv.make_signal_handler(S, stop, force_on_repeat=False, hard_exit=calls.append)
    hup(); hup()                                                     # SIGHUP 반복은 정리를 끊지 않는다
    assert calls == [130]


def test_closing_ignores_ready_and_refuses_start(tmp_path, monkeypatch):
    """종료가 시작되면 늦게 온 워커 ready 는 알리지 않고, 새 start 는 거절한다."""
    async def body():
        async with Harness(tmp_path, monkeypatch, delay_load=0.6) as h:
            ws = await h.ws(); hello = await recv(ws)
            assert hello["state"] == "loading"
            h.server.closing.set()
            await asyncio.sleep(1.0)                                   # 그사이 적재가 끝나도
            assert h.server.state != "ready" and not h.server.worker.in_loop
            await ws.send_json(start_msg())
            e = await until(ws, lambda m: m["type"] == "error")
            assert e["code"] == "model_not_ready" and "종료" in e["message"]
            await ws.close()
    asyncio.run(body())


def test_reuse_running_instance(tmp_path, monkeypatch):
    """이미 VAPKT 가 떠 있으면 두 번째 실행은 그 주소를 알리고 끝난다 (모델 사본을 하나 더 만들지 않는다).
    다른 프로그램이 듣는 포트는 건너뛰고, 명령줄 옵션이 다르면 1, --new-instance 는 찾지 않는다, 고정 포트가 바쁘면 무엇이 쓰는지 알려 준다."""
    async def body():
        async with Harness(tmp_path, monkeypatch) as h:
            await h.ready()
            port = h.client.port
            loop = asyncio.get_running_loop()
            j = await loop.run_in_executor(None, srv.probe_health, "127.0.0.1", port)
            assert j and j["app"] == "vapkt" and j["requested"]["engine"] == "fake" and j["requested"]["kv_capacity"] == 2048
            other = socket.socket(); other.bind(("127.0.0.1", 0)); other.listen(4)
            op = other.getsockname()[1]
            try:
                assert await loop.run_in_executor(None, srv.probe_health, "127.0.0.1", op, 0.3) is None
                hit = await loop.run_in_executor(None, lambda: srv.find_running("127.0.0.1", [op, port], probe=lambda hh, pp: srv.probe_health(hh, pp, 0.3)))
                assert hit and hit[0] == port
                made = []
                monkeypatch.setattr(srv, "Server", lambda *a, **k: made.append(1))
                monkeypatch.setattr(srv, "PORT_RANGE", [port])
                assert await srv.amain(srv.parse_args(["--engine", "fake", "--no-browser"])) == 0
                assert await srv.amain(srv.parse_args(["--engine", "fake", "--no-browser", "--kv-capacity", "1024"])) == 1
                assert await srv.amain(srv.parse_args(["--no-browser"])) == 1             # 엔진 종류가 다르면(mlx ≠ fake) 늘 다른 설정
                hj = dict(j, requested=dict(j["requested"], model="/x/q06"), model=dict(j["model"], dir="/x/q06/final"))
                assert srv.option_conflicts(srv.parse_args(["--engine", "fake", "--model", "/x/q06"]), hj) == []
                assert srv.option_conflicts(srv.parse_args(["--engine", "fake", "--model", "/x/q06/final"]), hj) == []
                assert srv.option_conflicts(srv.parse_args(["--engine", "fake", "--model", "/y"]), hj)
                with pytest.raises(SystemExit):                       # 빈 포트 없음 → 엔진 적재 전에 끝난다
                    await srv.amain(srv.parse_args(["--engine", "fake", "--no-browser", "--new-instance"]))
                assert not made
                with pytest.raises(SystemExit) as ei:
                    await loop.run_in_executor(None, srv.bind_port, "127.0.0.1", port)
                assert f"이미 VAPKT 가 실행 중입니다: http://127.0.0.1:{port}/" in str(ei.value)
                with pytest.raises(SystemExit) as ei:
                    await loop.run_in_executor(None, lambda: srv.bind_port("127.0.0.1", op, probe=lambda hh, pp: None))
                assert "사용 중입니다" in str(ei.value) and "lsof" in str(ei.value)
            finally:
                other.close()
    asyncio.run(body())
