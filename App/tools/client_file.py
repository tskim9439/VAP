#!/usr/bin/env python
"""헤드리스 WS 클라이언트: 오디오 파일 → 서버 (/ws, proto 1) 를 1× / N× / 최대 속도로 보낸다 (audio-ui-spec §3.4, §4).

delta·세그먼트 끝·통계를 출력하고, 받은 모든 이벤트를 JSON 으로 저장한다. 오류 이벤트·텍스트 불일치·parity 불일치면 0 이 아닌 값으로 끝난다.
  python App/tools/client_file.py a.wav --src Korean                         # 1×, VAD 분할, 마이크와 같은 20 ms 패킷
  python App/tools/client_file.py a.wav --src English --mode translate --tgt German --speed 0 --json out.json
  python App/tools/client_file.py a.wav --src German --parity                # 평가 방식: 분할 없음 + 이득 끔 (+ 1.0 s 0 은 서버가)
  python App/tools/client_file.py --golden /Volumes/Samsung_T5/vapkt-models/app-work/golden-s4            # 모든 골든 케이스 token id 대조
종료 코드: 0 정상, 1 오류 이벤트/연결 실패, 2 사용법, 3 delta 연결 ≠ 최종 텍스트, 4 parity 불일치, 5 시간 초과.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave

import numpy as np
import aiohttp

REASON_KO = {"pause": "멈춤", "soft_max": "긴 발화", "max_len": "최대 길이", "kv_guard": "KV 한도", "stop": "정지",
             "file_end": "파일 끝", "aborted": "중단"}


class Fail(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code = code


def read_audio(path: str) -> tuple[np.ndarray, int]:
    """float32 모노 + 샘플레이트. soundfile 이 있으면 그것으로(모든 형식), 없으면 wave(PCM)."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32).reshape(-1), 16000
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1).astype(np.float32), int(sr)
    except ImportError:
        pass
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2: x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 4: x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else: raise Fail(2, f"wave 모듈은 16/32-bit PCM 만 읽습니다 (soundfile 을 설치하세요): {path}")
    return x.reshape(-1, ch).mean(axis=1).astype(np.float32), sr


def fmt_t(s):
    if s is None: return "—"
    m = int(s // 60); return f"{m:02d}:{s - 60 * m:04.1f}"


class Client:
    def __init__(self, a):
        self.a = a; self.events: list[dict] = []; self.hello = None; self.ws = None
        self.done_s = 0.0; self.stats_n = 0

    def log(self, *x):
        if not self.a.quiet: print(*x, flush=True)

    async def recv_loop(self, sink: asyncio.Queue):
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                m = json.loads(msg.data); m["_t"] = round(time.monotonic(), 4)
                self.events.append(m)
                if m["type"] == "stats": self.done_s = m.get("audio_done_s") or 0.0
                await sink.put(m)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
        await sink.put({"type": "_closed"})

    async def wait_for(self, q, pred, timeout, what):
        t0 = time.monotonic()
        while True:
            left = timeout - (time.monotonic() - t0)
            if left <= 0: raise Fail(5, f"시간 초과: {what}")
            try:
                m = await asyncio.wait_for(q.get(), left)
            except asyncio.TimeoutError:
                raise Fail(5, f"시간 초과: {what}")
            if m["type"] == "_closed": raise Fail(1, f"서버 연결이 끊겼습니다 ({what})")
            self.show(m)
            if m["type"] == "error" and (m.get("fatal") or what.startswith("started")):
                raise Fail(1, f"서버 오류 {m.get('code')}: {m.get('message')}")
            if pred(m): return m

    def drain(self, q):
        """받아 둔 이벤트를 보여 주며 비운다. 서버가 세션을 끝냈으면(치명 오류·session_end·연결 끊김) 바로 실패."""
        while not q.empty():
            m = q.get_nowait()
            if m["type"] == "_closed": raise Fail(1, "서버 연결이 끊겼습니다")
            self.show(m)
            if m["type"] == "error" and m.get("fatal"): raise Fail(1, f"서버 오류 {m['code']}: {m['message']}")
            if m["type"] == "session_end": raise Fail(1, f"서버가 세션을 끝냈습니다 ({m.get('reason')})")

    def show(self, m):
        t = m["type"]; a = self.a
        if t == "delta" and (m["src"] or m["tr"]):
            lag = m.get("proc_lag_ms"); lag = f"{lag:.0f} ms" if lag is not None else "—"
            s = f"  #{m['seg']} k={m['k']:<4} 지연 {lag:>7} │ {m['src']}" + (f" │ → {m['tr']}" if m["tr"] else "")
            self.log(s)
        elif t == "segment_start":
            self.log(f"▶ #{m['seg']} {fmt_t(m['t_start'])} {m['reason']} 이득 {m['gain_db']} dB δ{m['config']['delay']} {m['config']['mode']}")
        elif t == "segment_end":
            self.log(f"■ #{m['seg']} {fmt_t(m['t_start'])}–{fmt_t(m['t_end'])} {REASON_KO.get(m['reason'], m['reason'])}"
                     f" · 마무리 {m.get('finalize_ms')} ms · 첫 글자 {m.get('first_text_ms')} ms{' · (숨김)' if m.get('hidden') else ''}")
            self.log(f"   원문: {m['src_text']}")
            if m["tr_text"]: self.log(f"   번역: {m['tr_text']}")
        elif t == "segment_final":
            self.log(f"   final #{m['seg']}: " + (f"건너뜀({m['skipped']})" if m.get("skipped") else f"{m.get('src')} │ {m.get('tr')} ({m.get('ms')} ms)"))
        elif t == "stats":
            self.stats_n += 1
            if self.stats_n % max(1, int(4 * a.stats_every)) == 0 or m.get("final"):
                pl = m["proc_lag_ms"]
                self.log(f"  [통계] 받음 {m['audio_rx_s']} s · 끝남 {m['audio_done_s']} s · 밀림 {m['backlog_s']} s · 지연 p50/p95 "
                         f"{pl['p50']}/{pl['p95']} ms · RTF {m['rtf_10s']} · 청크 p95 {m['chunk_ms']['p95']} ms · KV {m['kv_pct']} %"
                         + (f" · 처리 속도 {m['throughput_x']}×" if m.get("throughput_x") is not None else ""))
        elif t in ("warning", "error"):
            self.log(f"  [{t} {m.get('code')}] {m.get('message')}")
        elif t == "session_end":
            self.log(f"세션 끝: 오디오 {m.get('audio_s')} s · 벽시계 {m.get('wall_s')} s · 세그먼트 {m.get('segments')} · RTF {m.get('rtf')} · "
                     f"지연 p95 {m.get('proc_lag_p95_ms')} ms · 최대 밀림 {m.get('backlog_max_s')} s" + (" · 중단됨" if m.get("aborted") else ""))

    async def connect(self, q):
        self.session = aiohttp.ClientSession()
        try:
            self.ws = await self.session.ws_connect(self.a.url, max_msg_size=1 << 22, heartbeat=10)
        except Exception as e:
            await self.session.close()
            raise Fail(1, f"서버에 연결할 수 없습니다: {self.a.url} ({e})")
        self.rx_task = asyncio.ensure_future(self.recv_loop(q))
        h = await self.wait_for(q, lambda m: m["type"] == "hello", 10, "hello")
        self.hello = h
        if h["state"] != "ready":
            self.log(f"모델 준비 대기 중… ({h.get('message')})")
            await self.wait_for(q, lambda m: m["type"] == "ready" or (m["type"] == "status" and m.get("state") == "error"),
                                self.a.wait_ready, "모델 준비")
            if any(e["type"] == "status" and e.get("state") == "error" for e in self.events):
                raise Fail(1, "서버가 모델을 불러오지 못했습니다")

    async def close(self):
        try:
            await self.ws.close()
        finally:
            await self.session.close()

    async def run_one(self, q, x, rate, cfg: dict, opts: dict):
        """세션 하나: start → 오디오 → stop → session_end. 이 세션의 이벤트 목록을 돌려준다."""
        a = self.a
        i0 = len(self.events)
        msg = dict(type="start", src_lang=cfg["src"], mode=cfg["mode"], delay=cfg["delay"], source=a.source, sample_rate=rate,
                   format=a.format, takeover=a.takeover, options=opts)
        if cfg["mode"] == "translate": msg["tgt_lang"] = cfg["tgt"]
        if cfg["delay"] is None: del msg["delay"]
        await self.ws.send_json(msg)
        st = await self.wait_for(q, lambda m: m["type"] in ("started", "error"), 30, "started")
        if st["type"] == "error": raise Fail(1, f"start 거절 {st['code']}: {st['message']}")
        self.log(f"세션 {st['session_id']} · {st['config']['src_lang']} {st['config']['mode']} {st['config'].get('tgt_lang') or ''} "
                 f"δ{st['config']['delay']} · 입력 {rate} Hz {st['resampler']} · 속도 {'최대' if a.speed == 0 else str(a.speed) + '×'}")
        self.done_s = 0.0
        pkt_ms = a.packet_ms or (20 if a.speed == 1 else 320)
        n = max(1, int(round(rate * pkt_ms / 1000)))
        t0 = time.monotonic(); i = 0
        while i < len(x):
            if a.speed == 0 or a.speed > 1:                   # 흐름 제어 (§4.5): 보낸 초 − audio_done_s ≤ 10 s
                p0 = time.monotonic()
                while i / rate - self.done_s > a.max_inflight:
                    await asyncio.sleep(0.05)
                    self.drain(q)                             # 기다리는 사이 서버가 세션을 끝냈으면 멈추지 않고 실패 (선점·오류)
                    if self.rx_task.done(): raise Fail(1, "서버 연결이 끊겼습니다")
                t0 += time.monotonic() - p0
            if a.speed > 0:
                wait = t0 + (i / rate) / a.speed - time.monotonic()
                if wait > 0: await asyncio.sleep(wait)
            c = x[i:i + n]
            b = c.astype("<f4").tobytes() if a.format == "f32le" else (np.clip(c, -1, 1) * 32767).round().astype("<i2").tobytes()
            await self.ws.send_bytes(b); i += len(c)
            self.drain(q)
        await self.ws.send_json({"type": "stop"})
        timeout = max(60.0, 3 * len(x) / rate)
        await self.wait_for(q, lambda m: m["type"] == "session_end", timeout, "session_end")
        return self.events[i0:]


def consistency(evs) -> list[str]:
    """segment_end 텍스트 == delta 연결 (§4.3)."""
    acc, bad = {}, []
    for m in evs:
        if m["type"] == "delta":
            s = acc.setdefault((m.get("sid"), m["seg"]), ["", ""]); s[0] += m["src"]; s[1] += m["tr"]
        elif m["type"] == "segment_end":
            s = acc.get((m.get("sid"), m["seg"]), ["", ""])
            if s[0] != m["src_text"] or s[1] != m["tr_text"]:
                bad.append(f"seg {m['seg']}: delta 연결 {s!r} ≠ 최종 {[m['src_text'], m['tr_text']]!r}")
    return bad


def emitted(evs) -> list[list[int]]:
    return [[m["k"], i] for m in evs if m["type"] == "delta" for i in m.get("ids", [])]


def compare(meta: dict, evs: list[dict]) -> dict:
    """골든 meta.emitted 와 서버가 낸 (k, id) 비교."""
    ref = [list(map(int, e)) for e in meta["emitted"]]; got = emitted(evs)
    ends = [m for m in evs if m["type"] == "segment_end"]
    r = dict(case=meta.get("case"), n_ref=len(ref), n_got=len(got), ids_equal=[i for _, i in ref] == [i for _, i in got], pairs_equal=ref == got,
             total_cap_hit=bool(meta.get("total_cap_hit")), segments=len(ends))
    if not r["pairs_equal"]:
        j = next((j for j in range(min(len(ref), len(got))) if ref[j] != got[j]), min(len(ref), len(got)))
        r["first_diff"] = dict(index=j, ref=ref[j:j + 5], got=got[j:j + 5])
    if ends:
        r["src_text_equal"] = ends[0]["src_text"] == meta.get("src_text")
        r["tr_text_equal"] = ends[0]["tr_text"] == (meta.get("tr_text_stream", meta.get("tr_text")) or "")
        r["K"] = ends[0].get("K"); r["K_ref"] = meta.get("K")
    fin = [m for m in evs if m["type"] == "segment_final"]
    if fin and meta.get("final_src") is not None and not fin[0].get("skipped"):
        r["final_src_equal"] = fin[0].get("src") == meta.get("final_src")
        if "ids" in fin[0] and meta.get("final_ids") is not None:
            r["final_ids_equal"] = [int(i) for i in fin[0]["ids"]] == [int(i) for i in meta["final_ids"]]
    return r


def golden_cases(path: str, only: str | None) -> list[str]:
    if os.path.isfile(os.path.join(path, "meta.json")): return [path]
    out = []
    for d in sorted(os.listdir(path)):
        p = os.path.join(path, d)
        if not d.startswith((".", "_")) and os.path.isfile(os.path.join(p, "meta.json")) and (not only or only in d):
            out.append(p)
    if not out: raise Fail(2, f"골든 케이스가 없습니다: {path}")
    return out


async def main_async(a) -> int:
    c = Client(a); q: asyncio.Queue = asyncio.Queue()
    report: dict = dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    code = 0
    try:
        await c.connect(q)
        report["hello"] = {k: c.hello.get(k) for k in ("proto", "state", "model", "version")}
        opts = dict(segmentation=a.segmentation, gain=a.gain, debug_tokens=a.debug_tokens or bool(a.golden),
                    final_redecode=a.final_redecode, speed=a.speed)
        if a.end_pause_ms: opts["end_pause_ms"] = a.end_pause_ms
        if a.hard_max_s: opts["hard_max_s"] = a.hard_max_s
        if a.lead_in: opts["lead_in"] = True
        if a.golden:
            results = []
            for d in golden_cases(a.golden, a.cases):
                meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
                wav = np.load(os.path.join(d, "wav.npy")).astype(np.float32).reshape(-1)[: int(meta["n_real"])]
                cfg = dict(src=meta["src"], tgt=meta.get("tgt"), mode="translate" if meta.get("tgt") else "asr", delay=int(meta["delta"]))
                o = dict(opts, segmentation="none", gain="off", file_name=os.path.basename(d), file_duration_s=len(wav) / 16000)
                c.log(f"\n=== {meta.get('case', d)} ({len(wav) / 16000:.2f} s, K {meta.get('K')}) ===")
                evs = await c.run_one(q, wav, 16000, cfg, o)
                r = compare(meta, evs); r["consistency"] = consistency(evs); results.append(r)
                ok = r["pairs_equal"] or r["total_cap_hit"]
                print(f"{'OK ' if ok else 'BAD'} {meta.get('case', d)}: ref {r['n_ref']} / got {r['n_got']} 토큰, K {r.get('K')}/{r.get('K_ref')}" +
                      (f", final ids {'같음' if r['final_ids_equal'] else '다름'}" if "final_ids_equal" in r else ""), flush=True)
                c.log(f"PARITY {'OK' if ok else '불일치'}: ref {r['n_ref']} / got {r['n_got']} 토큰" +
                      ("" if ok else f" · 첫 차이 {r.get('first_diff')}") + (" (total cap hit → 대조 제외)" if r["total_cap_hit"] else ""))
            report["parity"] = results
            n_ok = sum(r["pairs_equal"] for r in results); n_skip = sum((not r["pairs_equal"]) and r["total_cap_hit"] for r in results)
            print(f"\nparity: {n_ok}/{len(results)} 케이스 일치" + (f", {n_skip} 개는 total cap hit 로 제외" if n_skip else ""), flush=True)
            if any(r["consistency"] for r in results): code = 3
            if n_ok + n_skip < len(results): code = 4
        else:
            if not a.wav: raise Fail(2, "오디오 파일 또는 --golden 을 주세요")
            x, rate = read_audio(a.wav)
            if a.parity:
                if rate != 16000: raise Fail(2, f"--parity 는 16 kHz 파일만 (받은 {rate} Hz) — 평가와 같은 샘플을 보내야 합니다")
                if len(x) > 30 * 16000: raise Fail(2, "--parity(분할 없음)는 30 s 이하만")
                opts.update(segmentation="none", gain="off")
            opts.update(file_name=os.path.basename(a.wav), file_duration_s=round(len(x) / rate, 3))
            cfg = dict(src=a.src, tgt=a.tgt, mode=a.mode, delay=a.delay)
            evs = await c.run_one(q, x, rate, cfg, opts)
            bad = consistency(evs)
            for b in bad: print(f"[불일치] {b}", file=sys.stderr)
            if bad: code = 3
            ends = [m for m in evs if m["type"] == "segment_end" and not m.get("hidden")]
            report["result"] = dict(src_text=" ".join(e["src_text"] for e in ends if e["src_text"]),
                                    tr_text=" ".join(e["tr_text"] for e in ends if e["tr_text"]), segments=len(ends),
                                    emitted=emitted(evs) if opts["debug_tokens"] else None)
        if any(m["type"] == "error" for m in c.events): code = code or 1
    except Fail as e:
        print(f"[실패] {e}", file=sys.stderr); code = e.code
    finally:
        if c.ws is not None:
            await c.close()
        if a.json:
            report["exit_code"] = code
            report["events"] = [m for m in c.events if a.json_vad or m["type"] != "vad"]
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
            print(f"이벤트 {len(report['events'])} 개 → {a.json}", flush=True)
    return code


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="VAPKT 서버 헤드리스 파일 클라이언트")
    ap.add_argument("wav", nargs="?", help="오디오 파일 (wav/flac/… soundfile, 또는 16 kHz .npy)")
    ap.add_argument("--url", default=None, help="기본 ws://127.0.0.1:<port>/ws")
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--src", default="Korean", choices=("Korean", "English", "German"))
    ap.add_argument("--mode", default="asr", choices=("asr", "translate"))
    ap.add_argument("--tgt", default=None, choices=(None, "Korean", "English", "German"))
    ap.add_argument("--delay", type=int, default=None, help="δ 4–10 (기본: 서버 기본값)")
    ap.add_argument("--speed", type=float, default=1.0, help="1 = 실시간, N = N 배속, 0 = 최대(흐름 제어)")
    ap.add_argument("--packet-ms", type=int, default=0, help="패킷 길이 (기본: 1× 는 20, 그 외 320)")
    ap.add_argument("--max-inflight", type=float, default=10.0, help="보낸 초 − audio_done_s 상한 (s)")
    ap.add_argument("--segmentation", default="vad", choices=("vad", "none"))
    ap.add_argument("--gain", default="auto", choices=("auto", "off"))
    ap.add_argument("--parity", action="store_true", help="평가 방식: segmentation none + gain off (16 kHz 파일)")
    ap.add_argument("--end-pause-ms", type=int, default=None)
    ap.add_argument("--hard-max-s", type=float, default=None)
    ap.add_argument("--lead-in", action="store_true", help="lead-in 켬 (soft/강제 절단 뒤 320 ms 배경 앞붙임, 수용 C6 A/B 용)")
    ap.add_argument("--format", default="f32le", choices=("f32le", "s16le"))
    ap.add_argument("--source", default="file", choices=("file", "mic"), help="mic 로 보내면 서버가 실시간 입력처럼 다룬다(경고 등)")
    ap.add_argument("--takeover", action="store_true")
    ap.add_argument("--debug-tokens", action="store_true", help="delta 에 토큰 id 포함 (parity 에 필요)")
    ap.add_argument("--final-redecode", action="store_true")
    ap.add_argument("--golden", default=None, help="골든 케이스 폴더 또는 그 상위 폴더 → token id 대조")
    ap.add_argument("--cases", default=None, help="--golden 상위 폴더에서 이 문자열이 들어간 케이스만")
    ap.add_argument("--json", default=None, help="모든 이벤트·요약 JSON 경로")
    ap.add_argument("--json-vad", action="store_true", help="JSON 에 vad 이벤트도 넣는다 (10 Hz)")
    ap.add_argument("--stats-every", type=float, default=1.0, help="통계 출력 간격 (s)")
    ap.add_argument("--wait-ready", type=float, default=900.0, help="모델 준비 대기 상한 (s)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "translate" and not a.tgt and not a.golden:
        ap.error("--mode translate 에는 --tgt 가 필요합니다")
    if a.speed < 0: ap.error("--speed 는 0 이상")
    a.url = a.url or f"ws://127.0.0.1:{a.port}/ws"
    return a


def main(argv=None):
    sys.exit(asyncio.run(main_async(parse_args(argv))))


if __name__ == "__main__":
    main()
