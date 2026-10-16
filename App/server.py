#!/usr/bin/env python
"""VAPKT 실시간 전사·번역 서버 — aiohttp(이벤트 루프) + MLX 워커 스레드 하나 (audio-ui-spec §4, §5).

이벤트 루프: HTTP(/, /static/*, /api/health), WS 읽기 → AudioPipeline(리샘플·VAD·분할·이득) → 워커 입력 큐,
            워커 이벤트 → TextAssembler → delta/segment_end, stats 4 Hz, vad 10 Hz.
워커 스레드: Engine(모든 MLX 객체)을 만들고·쓰고·버린다. 루프와는 numpy 배열·파이썬 값만 주고받는다.
실행: bash App/run.sh   (또는 python App/server.py --engine fake --no-browser)
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import ctypes
import dataclasses
import datetime as _dt
import importlib.util
import inspect
import json
import logging
import os
import platform
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from urllib.parse import urlsplit

import numpy as np
from aiohttp import WSMsgType, web

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(APP_DIR, "web")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from audio_pipeline import AudioPipeline, InputTooLong, ParamError, SegParams  # noqa: E402  (W4, 순수 numpy)
from text_stream import TextAssembler  # noqa: E402

APP_VERSION = "0.1.0"
PROTO = 1
SR = 16000
PORT_RANGE = range(8796, 8806)
FORBIDDEN_PORTS = {8765, 8790}          # 8790 은 옛 데모 서버가 쓰는 중 — 절대 건드리지 않는다
HOME = os.path.expanduser(os.environ.get("VAPKT_APP_HOME", "~/.vapkt-app"))
log = logging.getLogger("vapkt")

# ── 엔진이 준비되기 전에도 UI 가 그릴 수 있는 정적 표 (엔진 값이 오면 덮어쓴다) ─────────────────────────────
LANGS = ["Korean", "English", "German"]
PAIRS = [("English", "German"), ("English", "Korean"), ("German", "English"), ("Korean", "English")]
DELAYS = [4, 5, 6, 7, 8, 9, 10]
DEFAULT_OPTIONS = {"segmentation": "vad", "pre_roll_ms": 480, "end_pause_ms": 800, "hard_max_s": 25.0, "gain": "auto",
                   "lead_in": False, "final_redecode": False, "debug_tokens": False, "save_audio": False, "tail_s": 1.0}
DEFAULTS = {"asr_delay": 6,
            "translate_delay": {"English>German": 10, "English>Korean": 10, "German>English": 10, "Korean>English": 8},
            "options": DEFAULT_OPTIONS}
# FLEURS LAAL(s): δ4/δ10 은 위키 보정값, δ6/δ8 은 raw + 0.12 s (≈) — audio-ui-spec §2.1 [critic]
_LAAL = {"English>German": {4: 1.52, 6: 1.68, 8: 1.88, 10: 2.00}, "English>Korean": {4: 3.06, 6: 3.16, 8: 3.39, 10: 3.65},
         "German>English": {4: 1.55, 6: 1.63, 8: 1.78, 10: 1.90}, "Korean>English": {4: 2.11, 6: 2.31, 8: 2.55, 10: 2.72}}
LAAL_TABLE = {p: {str(d): {"s": v, "approx": d in (6, 8)} for d, v in t.items()} for p, t in _LAAL.items()}
LATENCY_TABLE = {str(d): {"mean_ms": 80 * d + 96, "lo_ms": 80 * d - 64, "hi_ms": 80 * d + 256} for d in DELAYS}
COMMON_RATES = {8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
       "connect-src 'self' ws://127.0.0.1:* ws://localhost:*; object-src 'none'; base-uri 'none'; "
       "frame-ancestors 'none'; form-action 'none'")
STATIC_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json"}


class BadConfig(ValueError):
    pass


def _jdefault(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (set, tuple)): return list(o)
    return str(o)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_jdefault)


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def pct(vals, q):
    if not vals: return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def err(code, message, fatal=False, **kw):
    return dict(type="error", code=code, message=message, fatal=fatal, **kw)


def warn(code, message, **kw):
    return dict(type="warning", code=code, message=message, **kw)


# ── 시작 점검 (§5.4) ────────────────────────────────────────────────────────────────────────────────────────
def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def env_checks(engine_kind: str) -> tuple[list[str], list[str]]:
    """(치명 오류, 경고). fake 엔진이면 치명 오류도 경고로 낮춘다."""
    errs, warns = [], []
    if sys.version_info < (3, 11):                 # setup.sh·requirements.lock 과 같은 하한 (numpy 2.4.6·filelock 4.2.0 은 3.11+)
        errs.append(f"Python 3.11 이상이 필요합니다 (현재 {platform.python_version()})")
    if platform.system() != "Darwin":
        errs.append("macOS(Apple Silicon) 전용입니다")
    else:
        if platform.machine() != "arm64":
            errs.append(f"arm64 Python 이 아닙니다 ({platform.machine()}) — Rosetta/Intel Python 으로는 MLX 를 쓸 수 없습니다")
        if _sysctl("sysctl.proc_translated") == "1":
            errs.append("Rosetta 로 실행 중입니다 — arm64 Python 으로 다시 만드세요 (bash App/setup.sh --recreate)")
        ver = platform.mac_ver()[0]
        try:
            if int(ver.split(".")[0]) < 14:
                warns.append(f"macOS {ver}: 14 이상을 권장합니다 (sw_vers {ver})")
        except ValueError:
            pass
        mem = _sysctl("hw.memsize")
        if mem.isdigit() and int(mem) < 24 * 2**30:
            warns.append(f"메모리 {int(mem) / 2**30:.0f} GB — 24 GB 미만이면 fp32 모드가 느리거나 메모리가 부족할 수 있습니다")
    if engine_kind == "fake":
        return [], errs + warns
    return errs, warns


def set_qos_user_interactive() -> str:
    """워커 스레드를 성능 코어 우선(QOS_CLASS_USER_INTERACTIVE = 0x21)으로. 결과 QoS 를 16진 문자열로."""
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        rc = lib.pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))
        lib.pthread_self.restype = ctypes.c_void_p
        lib.pthread_get_qos_class_np.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]
        q, r = ctypes.c_uint(0), ctypes.c_int(0)
        lib.pthread_get_qos_class_np(lib.pthread_self(), ctypes.byref(q), ctypes.byref(r))
        return hex(q.value) if rc == 0 else f"실패 rc={rc}"
    except Exception as e:  # 맥이 아니거나 심볼 없음
        return f"n/a ({type(e).__name__})"


# ── 포트 (§5.4: 8796–8805, 8765·8790 제외) ─────────────────────────────────────────────────────────────────
def _port_in_use(host: str, port: int) -> bool:
    probe = "127.0.0.1" if host in ("", "0.0.0.0") else host
    try:
        with socket.create_connection((probe, port), timeout=0.2):
            return True
    except OSError:
        return False


def candidate_ports(want: int | None) -> list[int]:
    if want is not None:
        if want in FORBIDDEN_PORTS:
            raise SystemExit(f"포트 {want} 은 쓰지 않습니다 (8765·8790 은 다른 서버용)")
        return [want]
    return [p for p in PORT_RANGE if p not in FORBIDDEN_PORTS]


def busy_port_message(host: str, port: int, probe=None) -> str:
    """고정한 --port 가 사용 중일 때: 그게 VAPKT 면 그 주소를, 아니면 확인 명령을 알려 준다."""
    if (probe or probe_health)(host, port) is not None:
        return f"포트 {port} 에 이미 VAPKT 가 실행 중입니다: http://127.0.0.1:{port}/"
    return f"포트 {port} 이 사용 중입니다 (확인: lsof -nP -iTCP:{port} -sTCP:LISTEN)"


def bind_port(host: str, want: int | None, in_use=_port_in_use, probe=None) -> tuple[socket.socket, int]:
    """이미 누가 듣고 있는 포트는 건너뛰고(connect 확인 — macOS 는 SO_REUSEADDR 로 겹쳐 bind 될 수 있다), 첫 빈 포트에 bind."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    for p in candidate_ports(want):
        if in_use(host, p):
            if want is not None: raise SystemExit(busy_port_message(host, p, probe))
            continue
        s = socket.socket(fam, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, p)); s.listen(128); s.setblocking(False)
            return s, p
        except OSError:
            s.close()
    if want is not None:
        raise SystemExit(f"포트 {want} 을 열 수 없습니다 (확인: lsof -nP -iTCP:{want} -sTCP:LISTEN)")
    ps = candidate_ports(want)
    raise SystemExit(f"빈 포트가 없습니다: {ps[0]}–{ps[-1]} (8765·8790 제외)")


# ── 이미 떠 있는 VAPKT 찾기 (run.sh 를 두 번 실행해도 모델 사본을 하나 더 만들지 않게) ─────────────────────────────
def probe_health(host: str, port: int, timeout: float = 1.0) -> dict | None:
    """그 포트의 /api/health 가 VAPKT 서버(같은 proto)면 그 JSON, 아니면 None. 프록시는 쓰지 않는다."""
    import urllib.request
    h = "127.0.0.1" if host in ("", "0.0.0.0", "localhost") else host
    if ":" in h: h = f"[{h}]"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{h}:{port}/api/health", timeout=timeout) as r:
            j = json.loads(r.read(1 << 20).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(j, dict) or j.get("proto") != PROTO or not j.get("version"): return None
    return j if j.get("app") == "vapkt" or ("worker_alive" in j and "model" in j) else None


def find_running(host: str, ports, in_use=_port_in_use, probe=None) -> tuple[int, dict] | None:
    """후보 포트 중 이미 VAPKT 가 듣고 있는 첫 포트 → (포트, health)."""
    for p in ports:
        if in_use(host, p):
            j = (probe or probe_health)(host, p)
            if j is not None: return p, j
    return None


_REQ_KEYS = ("engine", "model", "precision", "encoder_precision", "kv_capacity", "no_speculative", "draft", "save_audio",
             "fake_spec", "fake_round_ms")


def requested_options(args) -> dict:
    """서버를 띄울 때 받은 엔진 관련 옵션 (/api/health 로 알린다 — 같은 설정의 재실행인지 판단용)."""
    return {k: getattr(args, k, None) for k in _REQ_KEYS}


def option_conflicts(args, health: dict) -> list[str]:
    """실행 중인 서버와 다른 설정: 엔진 종류(늘), 그리고 명령줄에서 기본값과 다르게 준 옵션 (예: --precision fp32 인데 서버는 auto)."""
    dflt = vars(parse_args([]))
    req = health.get("requested")
    model = health.get("model") or {}
    out = []
    for k in _REQ_KEYS:
        v = getattr(args, k, None)
        have = (req or {}).get(k, model.get("engine") if k == "engine" else None)
        if k == "engine":
            if v != have: out.append(f"--engine {v} (실행 중: {have})")
            continue
        if v == dflt.get(k): continue                     # 명령줄에서 따로 주지 않았다
        if req is None: out.append(f"--{k.replace('_', '-')}"); continue    # 옛 서버: 확인할 수 없다
        if k == "model":                                  # run.sh 는 찾은 폴더를 늘 --model 로 준다: <dir> 또는 <dir>/final 이면 같다
            cand = {os.path.realpath(str(x)) for x in (have, model.get("dir")) if x}
            mine = os.path.realpath(str(v))
            if mine in cand or os.path.join(mine, "final") in cand: continue
        elif v == have:
            continue
        out.append(f"--{k.replace('_', '-')} {v} (실행 중: {have})")
    return out


def say(msg: str, err: bool = False):
    """터미널 출력. 터미널 창이 닫혀(SIGHUP) tty 가 죽었으면 EIO 를 삼킨다 — 종료 정리가 건너뛰어지지 않게."""
    try:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)
    except (OSError, ValueError):
        pass


# ── 엔진 생성 (워커 스레드에서만 부른다) ──────────────────────────────────────────────────────────────────────
def _import_fake():
    try:
        from engine.fake import FakeEngine
        return FakeEngine
    except ImportError:  # engine/__init__ 이 아직 없거나 깨졌을 때: fake.py 만 단독 적재
        path = os.path.join(APP_DIR, "engine", "fake.py")
        spec = importlib.util.spec_from_file_location("vapkt_fake_engine", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.FakeEngine


def make_engine_factory(args):
    def fake():
        cls = _import_fake()
        want = {"kv_capacity": args.kv_capacity, "round_ms": args.fake_round_ms}
        try:
            names = inspect.signature(cls).parameters
        except (TypeError, ValueError):
            names = {}
        kw = {k: v for k, v in want.items() if k in names}
        if args.fake_spec and hasattr(cls, "from_spec"):     # 예: "round_ms=120,hangul=1,asr_trans=1"
            return cls.from_spec(args.fake_spec, **kw)
        return cls(**kw)

    def mlx():
        import mlx.core as mx
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU 를 쓸 수 없습니다 (mx.metal.is_available() = False)")
        from engine import Engine
        return Engine(model_dir=args.model, kv_capacity=args.kv_capacity, precision=args.precision,
                      encoder_precision=args.encoder_precision, speculative=not args.no_speculative,
                      draft=getattr(args, "draft", "auto") or "auto")

    return fake if args.engine == "fake" else mlx


def _mem_gb(kind: str = "active"):
    """MLX 메모리 (GB): active = 지금 쓰는 양, peak = 시작 뒤 최대. 워커 스레드에서 부른다."""
    mx = sys.modules.get("mlx.core")
    if mx is None: return None
    try:
        f = getattr(mx, f"get_{kind}_memory", None) or getattr(mx.metal, f"get_{kind}_memory")
        return round(f() / 1e9, 2)
    except Exception:
        return None


def _round_dict(r, K) -> dict:
    g = lambda n, d=None: getattr(r, n, d)  # noqa: E731
    k = int(g("k", 0))
    fl = g("is_flush")
    if fl is None: fl = K is not None and k >= K
    return dict(k=k, ids=[int(x) for x in (g("ids") or [])], forced=bool(g("forced", False)), is_tail=bool(g("is_tail", False)),
                is_flush=bool(fl), need_samples=int(g("need_samples", 0) or 0), enc_ms=float(g("enc_ms", 0.0) or 0.0),
                dec_ms=float(g("dec_ms", 0.0) or 0.0), kv_full=bool(g("kv_full", False)))


class Worker:
    """MLX 를 소유하는 유일한 스레드. 명령: seg_start/audio/seg_end/stop/abort/shutdown (FIFO)."""

    def __init__(self, server: "Server", factory):
        self.server, self.factory = server, factory
        self.q: queue.Queue = queue.Queue()
        self.aborted: set[str] = set()
        self.released = 0            # 루프만 쓴다 (워커로 보낸 16 kHz 샘플 수)
        self.consumed = 0            # 워커만 쓴다 (엔진이 소비했거나 버린 샘플 수)
        self.tb: dict[int, bytes] = {}   # token_bytes 캐시: 워커가 채우고 루프가 읽는다
        self.ident = None; self.qos = None; self.loop = None; self._eng = None   # _eng 은 워커 스레드에서만 쓴다
        self._counted = False
        self.in_loop = False         # 적재·워밍업을 마치고 명령 루프에 들어갔는가 (종료 때 join 을 얼마나 기다릴지)
        self.thread = threading.Thread(target=self._run, name="vapkt-engine", daemon=True)

    def start(self, loop):
        self.loop = loop; self.thread.start()

    def put(self, cmd, n: int = 0):
        self.released += n; self.q.put(cmd)

    def abort(self, sid):
        self.aborted.add(sid); self.q.put(("abort", sid))

    def backlog_s(self) -> float:
        return max(0, self.released - self.consumed) / SR

    def post(self, *ev):
        try:
            self.loop.call_soon_threadsafe(self.server.on_worker_event, ev)
        except RuntimeError:     # 루프가 이미 닫힘(종료 중)
            pass

    # ─ 스레드 본체
    def _run(self):
        self.ident = threading.get_ident(); self.qos = set_qos_user_interactive()
        t0 = time.monotonic()
        self.post("status", "loading", "모델 불러오는 중…")
        closing = self.server.closing
        try:
            eng = self.factory()
            load_ms = (time.monotonic() - t0) * 1000
            if closing.is_set():                         # 적재 중 종료 요청: 워밍업·ready 없이 끝낸다
                box = [eng]; del eng; self._release(box); return
            self.post("status", "warmup", "준비 중(커널 컴파일)")
            t1 = time.monotonic(); wu = eng.warmup() or {}
            warm_ms = (time.monotonic() - t1) * 1000
            if closing.is_set():
                box = [eng]; del eng; self._release(box); return
            meta = dict(langs=list(getattr(eng, "langs", LANGS)), pairs=[list(p) for p in getattr(eng, "pairs", PAIRS)],
                        delays=[int(d) for d in getattr(eng, "delays", DELAYS)], defaults=getattr(eng, "defaults", None) or {},
                        spk_ids=[int(x) for x in eng.spk_ids], trans_id=int(eng.trans_id),
                        info=json.loads(dumps(getattr(eng, "info", {}) or {})), warmup=json.loads(dumps(wu)), qos=self.qos,
                        load_peak_gb=_mem_gb("peak"), mem_gb=_mem_gb())
            mx = sys.modules.get("mlx.core")
            if mx is not None and hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()                  # 세션 통계의 mem_peak_gb 는 적재 이후(실행 중) 최대값
            log.info("MLX 메모리: 적재 최대 %s GB, 상주 %s GB", meta["load_peak_gb"], meta["mem_gb"])
            for t in meta["spk_ids"] + [meta["trans_id"]]:
                self.tb[t] = eng.token_bytes(t)
        except Exception as e:
            log.exception("엔진 로드 실패")
            self.post("load_error", str(e) if type(e).__name__ == "ModelNotFound" else f"{type(e).__name__}: {e}")
            while self.q.get()[0] != "shutdown":
                pass
            return
        self._eng = eng
        self.in_loop = True
        self.post("ready", meta, load_ms, warm_ms)
        self._loop(eng)
        self._eng = None
        box = [eng]; del eng
        self._release(box)

    @staticmethod
    def _release(box):
        """엔진을 닫고 MLX 캐시를 비운다. 마지막 참조를 여기서 버리도록 리스트로 받는다."""
        eng = box.pop()
        try:
            if hasattr(eng, "close"): eng.close()       # MLX 메모리 해제 (워커 스레드에서)
        except Exception:
            log.exception("엔진 close 실패")
        del eng
        mx = sys.modules.get("mlx.core")
        if mx is not None:
            try: (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
            except Exception: pass

    def _loop(self, eng):
        cur = None           # [sid, seg, decoder] — 워커에서 열린 세그먼트
        finals: list = []    # [[sid, seg, decoder, job|None]] — final 재디코드 대기 (앞에서부터, 한 걸음씩)
        while True:
            if finals and self.q.empty():            # 스트리밍 일이 없을 때만 final 한 걸음 (§5.2) — 별도 KV 라 열린 세그먼트와 섞여도 된다
                t_w = time.monotonic()
                self._final_step(finals)
                self._slow("final 한 걸음", t_w)
                continue
            cmd = self.q.get()
            t_w = time.monotonic()
            try:
                cur = self._handle(eng, cmd, cur, finals)
            finally:
                self._slow(cmd[0], t_w)
            if cur == "shutdown":
                return

    @staticmethod
    def _slow(what, t0):
        """워커 한 번의 작업이 길면 남긴다 (진단: 400 ms 넘으면 INFO, 100 ms 넘으면 DEBUG)."""
        dt = (time.monotonic() - t0) * 1000
        if dt > 400: log.info("워커 %s %.0f ms", what, dt)
        elif dt > 100: log.debug("워커 %s %.0f ms", what, dt)

    def _handle(self, eng, cmd, cur, finals):
        """명령 하나 처리 → 새 cur (열린 세그먼트) 또는 "shutdown"."""
        kind = cmd[0]
        if kind == "shutdown":
            return "shutdown"
        sid = cmd[1]
        if sid in self.aborted:              # 중단된 세션의 남은 명령은 버린다 (유일한 폐기 경로)
            self.consumed += self._pcm_len(cmd)
            if kind == "abort":
                self.aborted.discard(sid)
                if cur and cur[0] == sid: cur = None
                finals[:] = [f for f in finals if f[0] != sid]
            return cur
        n = self._pcm_len(cmd); self._counted = False
        try:
            if kind == "seg_start":
                _, _, seg, (src, tgt, delay), pcm, lead = cmd
                dec = eng.new_segment(src, tgt, delay)
                try:
                    dec.on_token = self._token_hook(eng, sid, seg, dec)    # finish() 의 꼬리 토큰도 흘려 보낸다 (cur 가 비어도 둔다)
                except AttributeError:
                    pass
                cur = [sid, seg, dec]
                self._feed(eng, cur, pcm, lead)
            elif kind == "audio":
                if cur and cur[0] == sid and cur[1] == cmd[2]:
                    self._feed(eng, cur, cmd[3], 0)
                else:
                    self.consumed += n
            elif kind == "seg_end":
                _, _, seg, zero_tail, want_final = cmd
                if cur and cur[0] == sid and cur[1] == seg:
                    dec = cur[2]; cur = None
                    t_call = time.monotonic(); rounds = dec.finish(int(zero_tail)); t_ret = time.monotonic()
                    self._post_rounds("finished", eng, sid, seg, dec, rounds, t_call, t_ret, 0, 0)
                    if want_final and (hasattr(dec, "final_job") or hasattr(dec, "final_redecode")):
                        finals.append([sid, seg, dec, None])
                        while len(finals) > 3:          # 밀리면 아직 시작하지 않은 가장 오래된 것부터 건너뛴다
                            i = next((j for j, f in enumerate(finals) if f[3] is None), 0)
                            o = finals.pop(i); self.post("final", o[0], o[1], None, 0.0, "lagging")
                else:
                    self.post("finished", sid, seg, [], {})
            elif kind == "stop":
                finals[:] = self._run_finals(finals, only=sid)
                self.post("stopped", sid)
        except Exception as e:
            log.exception("워커 오류 (%s)", kind)
            if not self._counted: self.consumed += n      # 버린 샘플도 소비로 세어야 backlog 가 남지 않는다
            cur = None
            self.post("error", sid, f"{type(e).__name__}: {e}")
        return cur

    @staticmethod
    def _pcm_len(cmd) -> int:
        if cmd[0] == "seg_start": return len(cmd[4])
        if cmd[0] == "audio": return len(cmd[3])
        return 0

    def _token_hook(self, eng, sid, seg, dec):
        """디코더가 토큰 하나를 확정하는 즉시 루프로 보내는 콜백 (feed()/finish() 가 끝나기를 기다리지 않는다).
        need·tail·flush 는 그 토큰이 속할 Round 와 같은 식 (decoder._round). 예외는 디코더 루프로 새지 않게 삼킨다."""
        spf = int(getattr(dec, "spf", 1280) or 1280)

        def on_token(k, tid):
            try:
                k, tid = int(k), int(tid)
                if tid not in self.tb: self.tb[tid] = eng.token_bytes(tid)   # 루프가 글자를 만들기 전에 채운다
                K, n_real = getattr(dec, "K", None), getattr(dec, "n_real", None)
                fed = int(getattr(dec, "samples_fed", 0) or 0)
                flush = K is not None and k >= K
                tail = flush or (n_real is not None and k * spf >= n_real)
                need = fed if flush else 4096 + 5120 * (k // 4)
                if not flush and n_real is not None: need = min(need, fed)
                self.post("partial", sid, seg, k, tid, need, tail, flush)
            except Exception:
                log.exception("토큰 콜백 실패 (seg %s)", seg)
        return on_token

    def _feed(self, eng, cur, pcm, lead):
        n = len(pcm)
        t_call = time.monotonic()
        try:
            rounds = cur[2].feed(pcm) if n else []
        finally:
            self.consumed += n; self._counted = True
        t_ret = time.monotonic()
        self._post_rounds("rounds", eng, cur[0], cur[1], cur[2], rounds, t_call, t_ret, n, n - lead)

    def _post_rounds(self, kind, eng, sid, seg, dec, rounds, t_call, t_ret, n, n_real):
        K = getattr(dec, "K", None)
        out, acc = [], t_call
        for r in rounds:
            d = _round_dict(r, K)
            for t in d["ids"]:
                if t not in self.tb: self.tb[t] = eng.token_bytes(t)
            acc += (d["enc_ms"] + d["dec_ms"]) / 1000
            d["ready"] = min(acc, t_ret)
            out.append(d)
        info = dict(consumed=n, consumed_real=max(0, n_real), positions=int(getattr(dec, "positions", 0) or 0),
                    kv_capacity=int(getattr(dec, "kv_capacity", 0) or 0), K=K, samples_fed=getattr(dec, "samples_fed", None),
                    call_ms=(t_ret - t_call) * 1000, mem_gb=_mem_gb(), mem_peak_gb=_mem_gb("peak"),
                    nonfinite=int(getattr(getattr(eng, "thinker", None), "nonfinite", 0) or 0))   # fp16 유한성 검사 실패 누계
        self.post(kind, sid, seg, out, info)

    def _final_step(self, finals) -> None:
        """맨 앞 final 작업을 forward 한 번만 진행한다 (§5.2: 사이사이 입력 큐를 본다). 끝나면 결과를 보내고 뺀다."""
        f = finals[0]
        sid, seg, dec, job = f
        if sid in self.aborted:
            finals.pop(0); return
        try:
            if job is None:                              # 시작할 때만 밀림 검사 (§7: backlog > 0.5 s 면 건너뜀)
                if self.backlog_s() > 0.5:
                    finals.pop(0); self.post("final", sid, seg, None, 0.0, "lagging"); return
                job = f[3] = dec.final_job() if hasattr(dec, "final_job") else _BlockingFinal(dec)
            if not job.step():
                return
            finals.pop(0)
            ids = [int(x) for x in job.ids]
            for t in ids:
                if t not in self.tb: self.tb[t] = self._eng.token_bytes(t)
            self.post("final", sid, seg, ids, float(job.ms), None)
        except Exception as e:
            log.exception("final 재디코드 실패")
            if finals and finals[0] is f: finals.pop(0)
            self.post("final", sid, seg, None, 0.0, f"error: {type(e).__name__}")

    def _run_finals(self, finals, only):
        """stop: 그 세션의 final 작업을 끝낸다 (session_end 전에 segment_final 이 가도록). 나머지는 남긴다.
        stop 은 그 세션의 마지막 명령이라 그 뒤 입력 큐에 든 것은 다른 세션(또는 abort·shutdown)의 일이다 —
        그게 오면 남은 작업은 lagging 으로 건너뛴다 (§5.2: 스트리밍이 먼저, 다른 탭을 몇 초씩 막지 않는다)."""
        mine = [f for f in finals if f[0] == only]
        keep = [f for f in finals if f[0] != only]
        for f in mine:
            q = [f]
            while q and q[0] is f:
                if not self.q.empty():
                    q.pop(0); self.post("final", f[0], f[1], None, 0.0, "lagging"); break
                self._final_step(q)
        return keep


class _BlockingFinal:
    """final_job 이 없는 엔진(옛 API)용: final_redecode 를 한 걸음으로 감싼다."""

    def __init__(self, dec):
        self.dec, self.ids, self.ms = dec, [], 0.0

    def step(self) -> bool:
        self.ids, self.ms = self.dec.final_redecode()
        return True


# ── 세션 측 통계 ────────────────────────────────────────────────────────────────────────────────────────────
class Rolling:
    """(벽시계, 값) 를 최근 win 초만 보관."""

    def __init__(self, win=10.0):
        self.win, self.d = win, deque()

    def add(self, t, v):
        self.d.append((t, v))

    def vals(self, now):
        while self.d and self.d[0][0] < now - self.win:
            self.d.popleft()
        return [v for _, v in self.d]


class SegState:
    def __init__(self, seg, cfg, info, asm, lead):
        self.seg, self.cfg, self.info, self.asm, self.lead = seg, cfg, info, asm, lead
        self.t_start = float(info.get("t_start", 0.0))
        self.t_end = None; self.end_wall = None; self.end_info = {}
        self.onset_wall = None; self.first_text_wall = None
        self.src_parts: list[str] = []; self.tr_parts: list[str] = []
        self.forced = 0; self.flush = 0; self.enc = 0.0; self.dec = 0.0; self.last_k = -1
        self.chunk_key = None; self.chunk_acc = 0.0
        self.kv_forced = False; self.positions = 0; self.K = None
        self.burst_t = self.t_start                # 시작 묶음(pre-roll)이 끝나는 오디오 시각
        self.streamed: dict[int, list[int]] = {}   # k → 토큰 콜백으로 미리 보낸 id (Round 가 오면 나머지만 민다)


class WavF32:
    """IEEE float32 모노 wav 를 이어 쓴다 (--save-audio). 닫을 때 헤더 길이를 고친다.
    프로세스가 정리 없이 죽어도(SIGKILL·충돌) 읽히도록 오디오 1 s 마다 헤더 길이를 고치고 버퍼를 내보낸다."""

    def __init__(self, path, rate):
        self.f = open(path, "wb"); self.rate = int(rate); self.n = 0; self.n_hdr = 0; self._hdr()

    def _hdr(self):
        data = self.n * 4
        self.f.write(b"RIFF" + struct.pack("<I", 36 + data) + b"WAVEfmt " +
                     struct.pack("<IHHIIHH", 16, 3, 1, self.rate, self.rate * 4, 4, 32) + b"data" + struct.pack("<I", data))
        self.n_hdr = self.n

    def write(self, x):
        self.f.write(np.asarray(x, "<f4").tobytes()); self.n += len(x)
        if self.n - self.n_hdr >= self.rate:
            try:
                self.f.seek(0); self._hdr(); self.f.seek(0, 2); self.f.flush()
            except Exception:
                pass

    def close(self):
        try:
            self.f.seek(0); self._hdr(); self.f.close()
        except Exception:
            pass


class Session:
    def __init__(self, server: "Server", conn: "Conn", cfg: dict):
        self.server, self.conn, self.cfg = server, conn, cfg
        self.worker: Worker = server.worker
        self.sid = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.state = "active"
        self.cur = {k: cfg[k] for k in ("src_lang", "mode", "tgt_lang", "delay")}
        self.cur.update(final_redecode=cfg["options"]["final_redecode"], debug_tokens=cfg["options"]["debug_tokens"])
        self.pending: dict = {}
        self.in_rate = int(cfg["sample_rate"]); self.f32 = cfg["format"] == "f32le"
        self.params = seg_params(cfg["options"])
        self.pipe = AudioPipeline(self.in_rate, self.params)
        self.params = self.pipe.params                       # 해석된 값 (soft_max_s·hold_from_s 채움)
        self.hold_from = float(self.params.hold_from_s)
        self.t0 = time.monotonic(); self.byte_rem = b""
        self.rx_native = 0; self.arr_t: list[float] = []; self.arr_w: list[float] = []
        self.segs: dict[int, SegState] = {}; self.open_seg = None; self.n_segments = 0
        self.seg_cfg: dict[int, dict] = {}     # 끝난 세그먼트의 설정 (final 재디코드 텍스트용)
        self.released = 0; self.consumed = 0; self.consumed_real = 0
        self.positions = 0; self.kv_cap = server.args.kv_capacity; self.mem_gb = None; self.mem_peak_gb = None
        self.lag = Rolling(); self.enc = Rolling(); self.dec = Rolling(); self.chunk = Rolling()
        self.compute = Rolling(); self.audio = Rolling(); self.tok = Rolling()
        self.lag_all: list[float] = []; self.compute_total = 0.0; self.audio_total = 0.0; self.backlog_max = 0.0
        self.chunk_all: list[float] = []; self.finalize_all: list[float] = []; self.first_text_all: list[float] = []   # 세션 전체 요약용
        self.warned: dict[str, float] = {}; self.zero_checked = False; self.zero_peak = 0.0
        self.quiet_since = None
        self.logf = None; self.wav = None
        os.makedirs(os.path.join(HOME, "sessions"), exist_ok=True)
        try:
            self.logf = open(os.path.join(HOME, "sessions", f"{self.sid}.jsonl"), "a", encoding="utf-8")
        except OSError:
            log.warning("세션 로그를 열 수 없습니다")
        if cfg["options"]["save_audio"] or server.args.save_audio:
            self.wav = WavF32(os.path.join(HOME, "sessions", f"{self.sid}.wav"), self.in_rate)

    # ─ 보조
    def send(self, obj):
        if obj.get("type") in ("warning", "error") and "sid" not in obj:
            obj = dict(obj, sid=self.sid)          # 같은 소켓의 다음 세션과 섞이지 않게
        self.conn.send(obj)

    def jlog(self, obj):
        if self.logf:
            try:
                self.logf.write(dumps(obj) + "\n"); self.logf.flush()
            except Exception:
                pass

    def warn_once(self, code, message, every=10.0, **kw):
        now = time.monotonic()
        if now - self.warned.get(code, -1e9) >= every:
            self.warned[code] = now; self.send(warn(code, message, **kw))

    def arrival_wall(self, t_audio: float) -> float:
        """세션 오디오 시각 t 의 샘플이 서버에 도착한 벽시계."""
        i = bisect.bisect_left(self.arr_t, t_audio - 1e-9)
        if i >= len(self.arr_w):
            return self.arr_w[-1] if self.arr_w else time.monotonic()
        return self.arr_w[i]

    def begin(self):
        rs = "bypass(16 kHz)" if self.in_rate == SR else "soxr-HQ"
        self.send(dict(type="started", session_id=self.sid, config=self.cfg, input_rate=self.in_rate, resampler=rs,
                       params=dataclasses.asdict(self.params) if dataclasses.is_dataclass(self.params) else {}))
        self.jlog(dict(type="session_start", session_id=self.sid, wall=_dt.datetime.now().isoformat(timespec="seconds"),
                       config=self.cfg, model=self.server.model_info(), version=APP_VERSION))
        if self.in_rate not in COMMON_RATES:
            self.send(warn("unusual_rate", f"드문 샘플레이트 {self.in_rate} Hz — 서버가 16 kHz 로 변환합니다"))

    # ─ 오디오 입력
    def push_bytes(self, data: bytes):
        buf = self.byte_rem + data
        w = 4 if self.f32 else 2
        n = len(buf) // w * w
        self.byte_rem = buf[n:]
        if n == 0: return
        if self.f32:
            x = np.frombuffer(buf[:n], "<f4").astype(np.float32)
            if not np.isfinite(x).all(): x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            x = np.frombuffer(buf[:n], "<i2").astype(np.float32) / 32768.0
        now = time.monotonic()
        self.rx_native += len(x)
        self.arr_t.append(self.rx_native / self.in_rate); self.arr_w.append(now)
        if len(self.arr_t) > 40000:
            del self.arr_t[:20000]; del self.arr_w[:20000]
        if self.wav: self.wav.write(x)
        self._input_checks(x)
        try:
            cmds = self.pipe.push(x)
        except InputTooLong as e:
            self.send(err("bad_config", str(e), fatal=True)); self.abort("bad_config"); return
        except Exception as e:
            log.exception("AudioPipeline.push 실패")
            self.send(err("internal", f"오디오 처리 오류: {type(e).__name__}: {e}", fatal=True)); self.abort("internal"); return
        self.dispatch(cmds)

    def _input_checks(self, x):
        if len(x) == 0: return
        peak = float(np.max(np.abs(x)))
        if not self.zero_checked:
            self.zero_peak = max(self.zero_peak, peak)
            if self.rx_native >= self.in_rate:
                self.zero_checked = True
                if self.zero_peak == 0.0 and self.cfg["source"] == "mic":
                    self.send(warn("zero_input", "마이크 입력이 0 입니다 — 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 브라우저를 허용하세요"))
        if peak >= 0.999:
            self.warn_once("clipping", "입력이 잘립니다(클리핑) — 마이크 음량을 낮추세요", every=10.0)

    # ─ 파이프라인 명령 → 워커
    def dispatch(self, cmds):
        for c in cmds:
            if c.kind == "seg_start": self._seg_start(c)
            elif c.kind == "audio": self._seg_audio(c)
            elif c.kind == "seg_end": self._seg_end(c)

    def _seg_start(self, c):
        if self.pending:
            self.cur.update(self.pending); self.pending = {}
        cfg = dict(self.cur)
        if cfg["mode"] != "translate": cfg["tgt_lang"] = None
        info = dict(c.info or {})
        lead = int(info.get("lead_in_samples", round(float(info.get("lead_in_ms", 0) or 0) * SR / 1000)))
        pcm = c.pcm if c.pcm is not None else np.zeros(0, np.float32)
        asm = TextAssembler(lambda t, tb=self.worker.tb: tb.get(t, b""), self.server.meta["spk_ids"], self.server.meta["trans_id"],
                            keep_tr=cfg["mode"] == "translate")
        ss = SegState(c.seg, cfg, info, asm, lead)
        ss.burst_t = ss.t_start + max(0, len(pcm) - lead) / SR
        ss.onset_wall = self.arrival_wall(float(info.get("onset_t", ss.t_start + float(info.get("pre_roll_ms", 0) or 0) / 1000)))
        self.segs[c.seg] = ss; self.open_seg = c.seg; self.n_segments += 1
        self.released += len(pcm)
        self.worker.put(("seg_start", self.sid, c.seg, (cfg["src_lang"], cfg["tgt_lang"], int(cfg["delay"])), pcm, lead), len(pcm))
        self.send(dict(type="segment_start", sid=self.sid, seg=c.seg, t_start=_r(ss.t_start), reason=info.get("reason", "vad"),
                       pre_roll_ms=info.get("pre_roll_ms", 0), lead_in_ms=info.get("lead_in_ms", 0), gain_db=_r(info.get("gain_db", 0.0), 1),
                       forced_start=bool(info.get("forced_start", False)), config={k: cfg[k] for k in ("src_lang", "mode", "tgt_lang", "delay")}))

    def _seg_audio(self, c):
        pcm = c.pcm
        if pcm is None or len(pcm) == 0: return
        self.released += len(pcm)
        self.worker.put(("audio", self.sid, c.seg, pcm), len(pcm))

    def _seg_end(self, c):
        ss = self.segs.get(c.seg)
        info = dict(c.info or {})
        if ss is not None:
            ss.end_info = info; ss.end_wall = time.monotonic(); ss.t_end = float(info.get("t_end", ss.t_start))
        if self.open_seg == c.seg: self.open_seg = None
        want_final = bool(ss.cfg.get("final_redecode")) if ss else False
        self.worker.put(("seg_end", self.sid, c.seg, int(info.get("zero_tail_samples", 0) or 0), want_final))

    # ─ 클라이언트 명령
    def update(self, m: dict):
        base = dict(self.cur); base.update(self.pending)
        upd = {k: m[k] for k in ("src_lang", "mode", "tgt_lang", "delay") if k in m}
        if "delay" in upd:
            try: upd["delay"] = int(upd["delay"])
            except (TypeError, ValueError): raise BadConfig(f"지연 δ 가 올바르지 않습니다: {upd['delay']!r}")
        cand = dict(base); cand.update(upd)
        if cand.get("mode") == "translate" and "tgt_lang" not in upd and \
                [cand["src_lang"], cand.get("tgt_lang")] not in [list(p) for p in self.server.meta["pairs"]]:
            cand["tgt_lang"] = self.server.default_tgt(cand["src_lang"])
        self.server.check_mode(cand["src_lang"], cand["mode"], cand.get("tgt_lang"), cand["delay"])
        opts = m.get("options") or {}
        pipe = {}
        if "end_pause_ms" in opts: pipe["end_pause_ms"] = _num(opts["end_pause_ms"], "end_pause_ms", 300, 2000, int)
        if "gain" in opts:
            if opts["gain"] not in ("auto", "off"): raise BadConfig("gain 은 auto|off")
            if self.pipe.params.segmentation != "none": pipe["gain"] = opts["gain"]
        if "final_redecode" in opts: cand["final_redecode"] = bool(opts["final_redecode"])
        if "debug_tokens" in opts: cand["debug_tokens"] = bool(opts["debug_tokens"])
        self.pending = {k: v for k, v in cand.items() if self.cur.get(k) != v}
        if pipe:                                    # 파이프라인이 다음 세그먼트 시작부터 적용한다
            try:
                self.pipe.update_params(**pipe)
            except ParamError as e:
                raise BadConfig(str(e))
        self.send(dict(type="updated", pending=cand, pipeline=pipe, applies="next_segment"))

    def stop(self):
        if self.state != "active": return
        self.state = "stopping"
        if self.server.active is self: self.server.active = None
        try:
            cmds = self.pipe.stop("file_end" if self.cfg["source"] == "file" else "stop")
        except Exception as e:
            log.exception("AudioPipeline.stop 실패")
            self.send(err("internal", f"오디오 처리 오류: {e}", fatal=True)); self.abort("internal"); return
        self.dispatch(cmds)
        self.worker.put(("stop", self.sid))

    def abort(self, reason="aborted", notify=True):
        if self.state == "closed": return
        c = self._counters()
        discarded = max(0.0, (self.released - self.consumed) / SR) + float(c.get("audio_held_s", 0.0) or 0.0)
        self.worker.abort(self.sid)
        for ss in list(self.segs.values()):          # 소켓이 닫혀도(notify=False) 열린 세그먼트의 텍스트는 로그에 남긴다 (§5.1)
            self._end_segment(ss, aborted=True, send=notify)
        if notify:
            self.send(self.stats_event(final=True, discarded_s=discarded))
        self._close(dict(type="session_end", session_id=self.sid, aborted=True, reason=reason, discarded_s=_r(discarded),
                         **self._summary()), notify)

    def _close(self, end_ev, notify=True):
        self.state = "closed"
        if self.server.active is self: self.server.active = None
        self.server.sessions.pop(self.sid, None)
        if notify: self.send(end_ev)
        self.jlog(end_ev)
        if self.logf: self.logf.close(); self.logf = None
        if self.wav: self.wav.close(); self.wav = None

    def _summary(self):
        c = self._counters()
        return dict(audio_s=_r(c.get("audio_rx_s", self.rx_native / self.in_rate)), wall_s=_r(time.monotonic() - self.t0, 2),
                    segments=self.n_segments, rtf=_r(self.compute_total / 1000 / self.audio_total if self.audio_total else None),
                    backlog_max_s=_r(self.backlog_max), proc_lag_p95_ms=_r(pct(self.lag_all, 0.95), 1), dropped_s=0.0,
                    proc_lag_p50_ms=_r(pct(self.lag_all, 0.5), 1), n_lag=len(self.lag_all),
                    chunk_ms=dict(p50=_r(pct(self.chunk_all, .5), 1), p95=_r(pct(self.chunk_all, .95), 1),
                                  max=_r(max(self.chunk_all) if self.chunk_all else None, 1), n=len(self.chunk_all),
                                  over_320=sum(c > 320 for c in self.chunk_all)),
                    finalize_ms=dict(p50=_r(pct(self.finalize_all, .5), 1), p95=_r(pct(self.finalize_all, .95), 1),
                                     max=_r(max(self.finalize_all) if self.finalize_all else None, 1)),
                    first_text_ms=dict(p50=_r(pct(self.first_text_all, .5), 1), max=_r(max(self.first_text_all) if self.first_text_all else None, 1)),
                    mem_gb=self.mem_gb, mem_peak_gb=self.mem_peak_gb)

    # ─ 워커 이벤트
    def on_worker(self, ev):
        kind = ev[0]
        if kind == "partial":
            _, _, seg, k, tid, need, tail, flush = ev
            ss = self.segs.get(seg)
            if ss is not None: self._partial(ss, k, tid, need, tail, flush)
        elif kind in ("rounds", "finished"):
            _, _, seg, rounds, info = ev
            self.consumed += info.get("consumed", 0); self.consumed_real += info.get("consumed_real", 0)
            if info.get("kv_capacity"): self.kv_cap = info["kv_capacity"]
            if "positions" in info: self.positions = info["positions"]
            if info.get("mem_gb") is not None: self.mem_gb = info["mem_gb"]
            if info.get("mem_peak_gb") is not None: self.mem_peak_gb = info["mem_peak_gb"]
            nf = info.get("nonfinite") or 0
            if nf > self.server.nonfinite_seen:      # 저정밀 thinker 의 inf/NaN 새로 생김 (thinker-spec §6.2) — 결과를 믿기 어렵다
                self.server.nonfinite_seen = nf
                log.warning("thinker 유한성 검사 실패 누계 %d", nf)
                self.warn_once("nonfinite", "수치 오류(inf/NaN)가 감지되었습니다 — 결과가 틀릴 수 있습니다. "
                               "--precision fp32 로 다시 실행하세요", every=30.0, count=nf)
            ss = self.segs.get(seg)
            if ss is None: return
            if info.get("K") is not None: ss.K = info["K"]
            ss.positions = info.get("positions", ss.positions)
            kv_full = False
            for r in rounds:
                self._round(ss, r); kv_full |= r["kv_full"]
            if kind == "finished":
                self._push_chunk(ss); self._end_segment(ss)
            elif (self.open_seg == seg and not ss.kv_forced and self.state == "active"
                  and (kv_full or (self.kv_cap and ss.positions > 0.85 * self.kv_cap))):
                ss.kv_forced = True
                self.send(warn("kv_guard", f"KV 사용량 {100 * ss.positions / max(1, self.kv_cap):.0f} % — 세그먼트를 강제로 끝냅니다"))
                self.dispatch(self.pipe.force_end("kv_guard", seg=seg))
        elif kind == "final":
            _, _, seg, ids, ms, skipped = ev
            if skipped or ids is None:
                e = dict(type="segment_final", sid=self.sid, seg=seg, skipped=skipped or "lagging")
            else:
                cfg = self.seg_cfg.get(seg, self.cur)
                asm = TextAssembler(lambda t, tb=self.worker.tb: tb.get(t, b""), self.server.meta["spk_ids"],
                                    self.server.meta["trans_id"], keep_tr=cfg.get("mode") == "translate")
                asm.push(ids); s, t, _ = asm.finish()
                e = dict(type="segment_final", sid=self.sid, seg=seg, src=s, tr=t, ms=_r(ms, 1), n_ids=len(ids))
                if cfg.get("debug_tokens"): e["ids"] = ids
            self.send(e); self.jlog(e)
        elif kind == "stopped":
            for ss in list(self.segs.values()):      # 정상이라면 비어 있다 (pipe.stop 이 열린 세그먼트를 끝낸다)
                self._end_segment(ss)
            self.send(self.stats_event(final=True))
            self._close(dict(type="session_end", session_id=self.sid, **self._summary()))
        elif kind == "error":
            self.send(err("internal", f"엔진 오류: {ev[2]}", fatal=True))
            self.abort("internal")

    def _timing(self, ss: SegState, need, tail, flush):
        """(t_avail, 기준 벽시계, late, held, burst). 꼬리·flush·끝 결정 뒤 라운드는 끝 결정 시각이 기준 (§3.3)."""
        t_avail = ss.t_start + max(0, need - ss.lead) / SR
        late = flush or tail or (ss.end_wall is not None and ss.t_end is not None and t_avail >= ss.t_end - 1e-9)
        if late and ss.end_wall is not None:
            t_avail, base = ss.t_end, ss.end_wall
        else:
            base = self.arrival_wall(t_avail)
        held = (not late) and (t_avail - ss.t_start) >= self.hold_from - 1e-6
        # 시작 묶음: VAD 가 onset 확인까지 잡아 둔 pre-roll 오디오 — 지연에 VAD 대기가 섞이므로 held 처럼 백분위에서 뺀다
        burst = (not late) and t_avail <= ss.burst_t + 1e-9
        return t_avail, base, late, held, burst

    def _partial(self, ss: SegState, k: int, tid: int, need, tail, flush):
        """토큰 콜백: 확정된 토큰 하나를 곧바로 delta(partial) 로 보낸다. 지연은 보내는 순간 기준이고,
        통계(p50/p95)는 라운드 단위로만 센다 (같은 라운드를 두 번 세지 않게)."""
        now = time.monotonic()
        ss.streamed.setdefault(k, []).append(tid)
        src_d, tr_d = ss.asm.push([tid])
        if ss.cfg["mode"] != "translate": tr_d = ""
        if src_d or tr_d:
            ss.src_parts.append(src_d); ss.tr_parts.append(tr_d)
            if ss.first_text_wall is None: ss.first_text_wall = now
        dbg = ss.cfg.get("debug_tokens")
        if not (src_d or tr_d or dbg): return
        t_avail, base, late, held, burst = self._timing(ss, need, tail, flush)
        e = dict(type="delta", sid=self.sid, seg=ss.seg, k=k, src=src_d, tr=tr_d, t_avail=_r(t_avail),
                 proc_lag_ms=_r(max(0.0, (now - base) * 1000), 1), partial=True)
        if tail and not flush: e["tail"] = True
        if flush: e["flush"] = True
        if held: e["held"] = True
        if burst: e["burst"] = True
        if dbg: e.update(ids=[tid], forced=False, tokens=self._pieces([tid], ss))
        self.send(e)

    def _round(self, ss: SegState, r: dict):
        now = time.monotonic()
        ids, k = r["ids"], r["k"]
        pre = ss.streamed.pop(k, None)               # 토큰 콜백으로 이미 보낸 앞부분은 빼고 민다
        rest = ids
        if pre:
            if ids[:len(pre)] != pre:
                log.warning("seg %s k %s: 미리 보낸 토큰 %s ≠ 라운드 %s", ss.seg, k, pre, ids)
            rest = ids[len(pre):]
        src_d, tr_d = ss.asm.push(rest)
        if ss.cfg["mode"] != "translate": tr_d = ""
        t_avail, base, late, held, burst = self._timing(ss, r["need_samples"], r["is_tail"], r["is_flush"])
        lag = max(0.0, (now - base) * 1000)          # 결과가 실제로 나가는 순간 기준 (추정 ready 가 아님)
        if not (held or burst):
            self.lag.add(now, lag); self.lag_all.append(lag)
        if r["enc_ms"] > 0: self.enc.add(now, r["enc_ms"])
        self.dec.add(now, r["dec_ms"])
        c = r["enc_ms"] + r["dec_ms"]
        self.compute.add(now, c); self.compute_total += c
        if not r["is_flush"]:
            self.audio.add(now, 0.08); self.audio_total += 0.08
            key = k // 4
            if ss.chunk_key is not None and key != ss.chunk_key: self._push_chunk(ss)
            ss.chunk_key = key; ss.chunk_acc += c
        self.tok.add(now, len(ids))
        ss.enc += r["enc_ms"]; ss.dec += r["dec_ms"]; ss.forced += r["forced"]; ss.flush += r["is_flush"]; ss.last_k = k
        if src_d or tr_d:
            ss.src_parts.append(src_d); ss.tr_parts.append(tr_d)
            if ss.first_text_wall is None: ss.first_text_wall = now
        dbg = ss.cfg.get("debug_tokens") and (ids or r["forced"])     # 디버그: 미리 보낸 라운드도 enc/dec·지연을 남긴다 (ids 는 나머지만)
        if src_d or tr_d or dbg:
            e = dict(type="delta", sid=self.sid, seg=ss.seg, k=k, src=src_d, tr=tr_d, t_avail=_r(t_avail), proc_lag_ms=_r(lag, 1))
            if r["is_tail"] and not r["is_flush"]: e["tail"] = True
            if r["is_flush"]: e["flush"] = True
            if held: e["held"] = True
            if burst: e["burst"] = True
            if ss.cfg.get("debug_tokens"):
                e.update(ids=rest, forced=r["forced"], tokens=self._pieces(rest, ss), enc_ms=_r(r["enc_ms"], 2), dec_ms=_r(r["dec_ms"], 2),
                         compute_lag_ms=_r(max(0.0, (r["ready"] - base) * 1000), 1))   # 옛 추정(계산 끝) — 진단용
            self.send(e)

    def _pieces(self, ids, ss):
        spk, tr = self.server.meta["spk_ids"], self.server.meta["trans_id"]
        run, out = getattr(ss, "_dbg_run", 0), []
        for t in ids:
            if t in spk: run = 0; out.append(["<SPK_A>" if t == spk[0] else "<SPK_B>", "tag"]); continue
            if t == tr: run = 1; out.append(["<TRANS>", "tag"]); continue
            b = self.worker.tb.get(t, b"")
            out.append([b.decode("utf-8", "replace") if b else f"<{t}>", "tr" if run else "src"])
        ss._dbg_run = run
        return out

    def _push_chunk(self, ss):
        if ss.chunk_key is not None:
            self.chunk.add(time.monotonic(), ss.chunk_acc); self.chunk_all.append(ss.chunk_acc)
        ss.chunk_key = None; ss.chunk_acc = 0.0

    def _end_segment(self, ss: SegState, aborted=False, send=True):
        """segment_end 를 만들어 보내고(send) 세션 로그에 쓴다. send=False 는 소켓이 닫힌 뒤 — 로그만."""
        if self.segs.pop(ss.seg, None) is None: return
        now = time.monotonic()
        src_text, tr_text, st = ss.asm.finish()
        fs, ft = st.get("final_src", ""), st.get("final_tr", "")
        if ss.cfg["mode"] != "translate": tr_text, ft = "", ""
        if (fs or ft) and not aborted:
            ss.src_parts.append(fs); ss.tr_parts.append(ft)
            if send: self.send(dict(type="delta", sid=self.sid, seg=ss.seg, k=ss.last_k, src=fs, tr=ft, t_avail=_r(ss.t_end), proc_lag_ms=None, final=True))
        if not aborted and ("".join(ss.src_parts) != src_text or "".join(ss.tr_parts) != tr_text):
            log.warning("seg %s: delta 연결 ≠ 최종 텍스트", ss.seg)
        info = ss.end_info
        voiced = float(info.get("voiced_s", 0.0) or 0.0)
        t_end = ss.t_end if ss.t_end is not None else (self.rx_native / self.in_rate)
        n_tr_dropped = int(st.get("tr_dropped", 0))
        # 숨김: 텍스트 없는 세그먼트(잡음). 단 final 재디코드가 올 세그먼트는 유성 6 프레임 이상이면 남긴다 (그 텍스트를 보일 줄)
        want_final = bool(ss.cfg.get("final_redecode")) and not aborted
        hidden = bool(not src_text and not tr_text and (not want_final or round(voiced / 0.032) < 6))
        e = dict(type="segment_end", sid=self.sid, seg=ss.seg, t_start=_r(ss.t_start), t_end=_r(t_end),
                 reason="aborted" if aborted else info.get("reason", "stop"), src_text=src_text, tr_text=tr_text,
                 audio_s=_r(t_end - ss.t_start), zero_tail_ms=_r(int(info.get("zero_tail_samples", 0) or 0) / 16, 1),
                 n_tokens=dict(src=int(st.get("n_src", 0)), tr=int(st.get("n_tr", 0)), tr_dropped=n_tr_dropped),
                 replaced_bytes=int(st.get("replaced_bytes", 0)), forced_rounds=ss.forced, flush_rounds=ss.flush,
                 finalize_ms=_r((now - ss.end_wall) * 1000, 1) if ss.end_wall and not aborted else None,
                 first_text_ms=_r((ss.first_text_wall - ss.onset_wall) * 1000, 1) if ss.first_text_wall and ss.onset_wall else None,
                 enc_ms_sum=_r(ss.enc, 1), dec_ms_sum=_r(ss.dec, 1), voiced_s=_r(voiced), gain_db=_r(ss.info.get("gain_db", 0.0), 1),
                 lead_in_ms=ss.info.get("lead_in_ms", 0), delay=ss.cfg["delay"], K=ss.K, positions=ss.positions,
                 hidden=hidden, config={k: ss.cfg[k] for k in ("src_lang", "mode", "tgt_lang", "delay")})
        self.seg_cfg[ss.seg] = ss.cfg
        if e["finalize_ms"] is not None: self.finalize_all.append(e["finalize_ms"])
        if e["first_text_ms"] is not None: self.first_text_all.append(e["first_text_ms"])
        if send: self.send(e)
        self.jlog(e)

    # ─ 통계
    def _counters(self) -> dict:
        try:
            return dict(self.pipe.counters())
        except Exception:
            return {}

    def stats_event(self, final=False, **extra):
        now = time.monotonic()
        c = self._counters()
        rx = float(c.get("audio_rx_s", self.rx_native / self.in_rate))
        backlog = max(0, self.released - self.consumed) / SR
        self.backlog_max = max(self.backlog_max, backlog)
        proc = self.consumed_real / SR
        done = min(rx, float(c.get("audio_skipped_s", 0.0) or 0.0) + proc)
        lag, enc, dec, chunk = self.lag.vals(now), self.enc.vals(now), self.dec.vals(now), self.chunk.vals(now)
        comp, aud, tok = sum(self.compute.vals(now)), sum(self.audio.vals(now)), sum(self.tok.vals(now))
        span = max(0.25, min(10.0, now - self.t0))
        e = dict(type="stats", session_id=self.sid, audio_rx_s=_r(rx), audio_seg_s=_r(c.get("audio_seg_s")),
                 audio_skipped_s=_r(c.get("audio_skipped_s")), audio_held_s=_r(c.get("audio_held_s")), audio_proc_s=_r(proc),
                 audio_done_s=_r(done), backlog_s=_r(backlog), discarded_s=0.0, dropped_s=0.0,
                 lead_in_s=_r(c.get("lead_in_s")), zero_tail_s=_r(c.get("zero_tail_s")), limited_samples=c.get("limited_samples"),
                 proc_lag_ms=dict(p50=_r(pct(lag, .5), 1), p95=_r(pct(lag, .95), 1), max=_r(max(lag) if lag else None, 1)),
                 enc_ms=dict(p50=_r(pct(enc, .5), 2), p95=_r(pct(enc, .95), 2)), dec_ms=dict(p50=_r(pct(dec, .5), 2), p95=_r(pct(dec, .95), 2)),
                 chunk_ms=dict(p50=_r(pct(chunk, .5), 1), p95=_r(pct(chunk, .95), 1)),
                 rtf_10s=_r(comp / 1000 / aud if aud else None), rtf_session=_r(self.compute_total / 1000 / self.audio_total if self.audio_total else None),
                 load_10s=_r(comp / 1000 / span), tok_per_s=_r(tok / span, 1),
                 kv_pct=_r(100 * self.positions / self.kv_cap if self.kv_cap else None, 1), positions=self.positions, mem_gb=self.mem_gb, mem_peak_gb=self.mem_peak_gb,
                 loop_lag_ms=self.server.loop_lag_p99(), state=self.state, final=final)
        if self.cfg["source"] == "file":
            e["throughput_x"] = _r(done / max(1e-3, now - self.t0), 2)
        e.update({k: _r(v) if isinstance(v, float) else v for k, v in extra.items()})
        return e

    def tick_stats(self):
        e = self.stats_event()
        self.send(e)
        b = e["backlog_s"] or 0.0
        if b > 600:
            self.send(err("overload", f"처리가 {b:.0f} s 밀려 세션을 멈춥니다", fatal=True)); self.abort("overload"); return
        realtime = self.cfg["source"] == "mic" or float(self.cfg["options"].get("speed") or 0) == 1
        if b > 2 and realtime:
            self.warn_once("lagging", f"처리가 실시간보다 {b:.1f} s 늦습니다", every=10.0, backlog_s=_r(b))

    def tick_vad(self):
        try:
            snap = dict(self.pipe.vad_snapshot())
        except Exception:
            return
        snap.update(type="vad")
        self.send(snap)
        # 작은 입력 경고: 말소리가 있는데(바닥 + 10 dB 이상) −50 dBFS 아래가 3 s 이어질 때
        lvl, floor, gain = snap.get("level_db"), snap.get("floor_db"), snap.get("gain_db") or 0.0
        if self.pipe.params.gain == "off":       # 파이프라인이 다음 세그먼트에서 params 를 바꿔 끼우므로 매번 읽는다
            quiet = lvl is not None and floor is not None and snap.get("state") == "speech" and lvl < -50
            if quiet:
                self.quiet_since = self.quiet_since or time.monotonic()
                if time.monotonic() - self.quiet_since >= 3:
                    self.warn_once("quiet_input", "입력이 너무 작습니다 — 마이크 음량을 올리거나 가까이에서 말하세요", every=30.0)
            else:
                self.quiet_since = None
        elif gain >= 29.5 and self._open_seg_has_speech(snap):
            self.warn_once("quiet_input", "입력이 너무 작습니다 — 마이크 음량을 올리거나 가까이에서 말하세요 (최대 이득)", every=30.0)

    def _open_seg_has_speech(self, snap) -> bool:
        """최대 이득 경고의 조건: 지금 열린 세그먼트가 말하는 중이고 원문이 두 단어 이상 나왔다.
        잡음만으로도 이득이 +30 dB 에 붙고(조용한 방), 세그먼트가 끝난 뒤에도 gain_db 는 지난 값을 보이므로."""
        seg = snap.get("seg")
        if seg is None or snap.get("state") != "speech": return False
        ss = self.segs.get(seg)
        return ss is not None and len("".join(ss.src_parts).split()) >= 2


def _num(v, name, lo, hi, typ=float):
    try:
        x = typ(v)
    except (TypeError, ValueError):
        raise BadConfig(f"{name} 값이 숫자가 아닙니다: {v!r}")
    if not (lo <= x <= hi):
        raise BadConfig(f"{name} 는 {lo}–{hi} 범위여야 합니다 (받은 값 {v})")
    return x


def seg_params(opts: dict) -> SegParams:
    """UI 옵션 → SegParams. soft_max·hold_from 은 hard_max 에서 유도 (audio-ui-spec §1.4 [critic])."""
    hard = float(opts["hard_max_s"])
    want = dict(segmentation=opts["segmentation"], pre_roll_ms=int(opts["pre_roll_ms"]), end_pause_ms=int(opts["end_pause_ms"]),
                hard_max_s=hard, soft_max_s=max(hard - 7.0, 0.6 * hard), hold_from_s=None, tail_s=float(opts["tail_s"]),
                lead_in=bool(opts["lead_in"]), gain=opts["gain"])   # soft 는 W4 식; hold_from=None ⇒ SegParams 가 hard − hold − 2 프레임 (F33)
    names = {f.name for f in dataclasses.fields(SegParams)}
    return SegParams(**{k: v for k, v in want.items() if k in names})


# ── 연결 ────────────────────────────────────────────────────────────────────────────────────────────────────
class Conn:
    """웹소켓 하나 — 송신 순서를 큐 하나로 보장한다."""

    def __init__(self, ws, peer):
        self.ws, self.peer = ws, peer
        self.q: asyncio.Queue = asyncio.Queue()
        self.closed = False; self.bin_warned = False
        self.task = asyncio.ensure_future(self._writer())

    def send(self, obj):
        if not self.closed: self.q.put_nowait(obj)

    async def _writer(self):
        while True:
            obj = await self.q.get()
            if obj is None: return
            try:
                await self.ws.send_str(dumps(obj))
            except Exception:
                self.closed = True; return

    async def close(self):
        self.closed = True; self.q.put_nowait(None)
        try:
            await asyncio.wait_for(self.task, 2.0)
        except Exception:
            pass


class Server:
    def __init__(self, args, engine_factory=None):
        self.args = args
        self.worker = Worker(self, engine_factory or make_engine_factory(args))
        self.state, self.state_msg, self.phase = "loading", "모델 로딩 준비 중", "loading"
        self.load_ms = self.warmup_ms = None
        self.meta = dict(langs=list(LANGS), pairs=[list(p) for p in PAIRS], delays=list(DELAYS), defaults=json.loads(json.dumps(DEFAULTS)),
                         spk_ids=[151707, 151708], trans_id=151726, info={})
        self.conns: set[Conn] = set(); self.sessions: dict[str, Session] = {}; self.active: Session | None = None
        self.port = None; self.t_boot = time.monotonic(); self.lag_ms = deque(maxlen=200)
        self.nonfinite_seen = 0      # 엔진 유한성 검사 실패 누계 (세션이 새 증가만 경고하도록)
        self.tasks: list[asyncio.Task] = []
        self.closing = threading.Event()   # 종료 시작 (신호 또는 앱 shutdown) — 워커도 읽는다

    # ─ aiohttp 앱
    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.h_index)
        app.router.add_get("/static/{name}", self.h_static)
        app.router.add_get("/api/health", self.h_health)
        app.router.add_get("/ws", self.h_ws)
        app.router.add_get("/favicon.ico", self.h_favicon)
        app.on_startup.append(self._on_startup)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def _on_startup(self, app):
        loop = asyncio.get_running_loop()
        self.worker.start(loop)
        self.tasks = [asyncio.ensure_future(self._tick(0.25, "stats")), asyncio.ensure_future(self._tick(0.1, "vad")),
                      asyncio.ensure_future(self._heartbeat())]

    async def _on_shutdown(self, app):
        self.closing.set()
        for s in list(self.sessions.values()):
            s.send(err("server_shutdown", "서버가 종료됩니다", fatal=True)); s.abort("server_shutdown")
        self.worker.q.put(("shutdown", None))
        # 적재·워밍업 중인 워커는 그 단계가 끝나야 큐를 본다 — 오래 기다리지 않는다 (daemon 스레드, amain 이 바로 끝낸다)
        wait = 5.0 if self.worker.in_loop or not self.worker.thread.is_alive() else 0.3
        await asyncio.get_running_loop().run_in_executor(None, self.worker.thread.join, wait)
        for t in self.tasks: t.cancel()
        for c in list(self.conns):
            await c.close()
            try:
                await c.ws.close()
            except Exception:
                pass

    async def _tick(self, every, what):
        while True:
            await asyncio.sleep(every)
            for s in list(self.sessions.values()):
                try:
                    if what == "stats": s.tick_stats()
                    elif s.state == "active": s.tick_vad()
                except Exception:
                    log.exception("tick %s", what)

    async def _heartbeat(self):
        while True:
            t = time.monotonic(); await asyncio.sleep(0.05)
            self.lag_ms.append((time.monotonic() - t - 0.05) * 1000)

    def loop_lag_p99(self):
        v = list(self.lag_ms)
        return _r(pct(v, 0.99), 2) if v else None

    def _headers(self):
        return {"Cache-Control": "no-store", "Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer"}

    async def h_favicon(self, req):
        return web.Response(status=204)

    async def h_index(self, req):
        return await self._file("index.html")

    async def h_static(self, req):
        return await self._file(req.match_info["name"])

    async def _file(self, name):
        ext = os.path.splitext(name)[1]
        if "/" in name or name.startswith(".") or ext not in STATIC_TYPES:
            raise web.HTTPNotFound()
        p = os.path.join(WEB_DIR, name)
        if not os.path.isfile(p):
            raise web.HTTPNotFound()
        with open(p, "rb") as f:
            body = f.read()
        return web.Response(body=body, content_type=STATIC_TYPES[ext], charset="utf-8", headers=self._headers())

    async def h_health(self, req):
        return web.json_response(dict(app="vapkt", ok=self.state == "ready", state=self.state, message=self.state_msg, version=APP_VERSION,
                                      proto=PROTO, model=self.model_info(), requested=requested_options(self.args),
                                      load_ms=self.load_ms, warmup_ms=self.warmup_ms,
                                      worker_alive=self.worker.thread.is_alive(), worker_qos=self.worker.qos,
                                      sessions=len(self.sessions), active=self.active.sid if self.active else None,
                                      backlog_s=round(self.worker.backlog_s(), 3), uptime_s=round(time.monotonic() - self.t_boot, 1),
                                      loop_lag_ms_p99=self.loop_lag_p99(), port=self.port), dumps=dumps, headers={"Cache-Control": "no-store"})

    def model_info(self):
        i = self.meta.get("info") or {}
        a = self.args
        return dict(name=i.get("name") or ("fake" if a.engine == "fake" else os.path.basename(str(a.model or "q06-s4-ast").rstrip("/"))),
                    dir=i.get("dir") or a.model, dtype=i.get("precision") or a.precision,
                    encoder_dtype=i.get("encoder_precision") or a.encoder_precision, kv_capacity=i.get("kv_capacity") or a.kv_capacity,
                    speculative=i.get("speculative", not a.no_speculative), engine=a.engine, device=i.get("device"), mlx=i.get("mlx"),
                    precision_note=i.get("precision_note"))

    def hello(self, conn=None):
        return dict(type="hello", proto=PROTO, state=self.state, phase=self.phase, message=self.state_msg, model=self.model_info(),
                    langs=self.meta["langs"], pairs=self.meta["pairs"], delays=self.meta["delays"], defaults=self.meta["defaults"],
                    busy=bool(self.active and (conn is None or self.active.conn is not conn)), latency_table=LATENCY_TABLE,
                    laal_table=LAAL_TABLE, load_ms=self.load_ms, warmup_ms=self.warmup_ms, version=APP_VERSION)

    def broadcast(self, obj):
        for c in list(self.conns): c.send(obj)

    # ─ 워커 이벤트 (루프 스레드)
    def on_worker_event(self, ev):
        kind = ev[0]
        if self.closing.is_set() and kind in ("status", "ready", "load_error"):   # 종료 중: '준비됨' 을 알리거나 찍지 않는다
            log.debug("종료 중 — 워커 %s 무시", kind)
            return
        if kind == "status":
            self.state_msg, self.phase = ev[2], ev[1]
            self.broadcast(dict(type="status", state=ev[1], message=ev[2], elapsed_s=round(time.monotonic() - self.t_boot, 1)))
        elif kind == "ready":
            meta, load_ms, warm_ms = ev[1], ev[2], ev[3]
            d = json.loads(json.dumps(DEFAULTS)); ed = meta.get("defaults") or {}
            if isinstance(ed.get("asr_delay"), int): d["asr_delay"] = ed["asr_delay"]
            if isinstance(ed.get("translate_delay"), dict): d["translate_delay"].update(ed["translate_delay"])
            meta["defaults"] = d
            self.meta.update(meta)
            self.state, self.state_msg, self.phase = "ready", "준비됨", "ready"
            self.load_ms, self.warmup_ms = round(load_ms), round(warm_ms)
            log.info("엔진 준비: load %.0f ms, warmup %.0f ms, qos %s, info %s", load_ms, warm_ms, meta.get("qos"), meta.get("info"))
            note = (meta.get("info") or {}).get("precision_note")
            if note:
                log.info("%s", note)
                say(f"{note} — 바꾸려면 --precision fp32|fp16 --encoder-precision fp32|fp16")
            for c in list(self.conns): c.send(self.hello(c))
            self.broadcast(dict(type="ready", load_ms=self.load_ms, warmup_ms=self.warmup_ms, model=self.model_info()))
        elif kind == "load_error":
            self.state, self.state_msg, self.phase = "error", ev[1], "error"
            log.error("모델을 불러오지 못했습니다: %s", ev[1])
            say(f"\n[오류] 모델을 불러오지 못했습니다:\n  {ev[1]}\n", err=True)
            self.broadcast(dict(type="status", state="error", message=ev[1]))
        else:
            s = self.sessions.get(ev[1])
            if s is not None:
                try:
                    s.on_worker(ev)
                except Exception:
                    log.exception("워커 이벤트 처리 실패 (%s)", kind)

    # ─ 검증
    def default_tgt(self, src):
        for a, b in self.meta["pairs"]:
            if a == src: return b
        return None

    def check_mode(self, src, mode, tgt, delay):
        if src not in self.meta["langs"]: raise BadConfig(f"원문 언어가 올바르지 않습니다: {src!r}")
        if mode not in ("asr", "translate"): raise BadConfig(f"mode 는 asr|translate: {mode!r}")
        if mode == "translate" and [src, tgt] not in [list(p) for p in self.meta["pairs"]]:
            raise BadConfig(f"학습되지 않은 번역 방향입니다: {src}→{tgt}")
        if not isinstance(delay, int) or delay not in self.meta["delays"]:
            raise BadConfig(f"지연 δ 가 올바르지 않습니다: {delay!r}")

    def validate_start(self, m: dict) -> dict:
        src = m.get("src_lang", "Korean"); mode = m.get("mode", "asr")
        tgt = m.get("tgt_lang") if mode == "translate" else None
        if mode == "translate" and not tgt: tgt = self.default_tgt(src)
        dflt = self.meta["defaults"]
        delay = m.get("delay")
        if delay is None:
            delay = dflt["asr_delay"] if mode == "asr" else dflt["translate_delay"].get(f"{src}>{tgt}", 10)
        try:
            delay = int(delay)
        except (TypeError, ValueError):
            raise BadConfig(f"지연 δ 가 올바르지 않습니다: {delay!r}")
        self.check_mode(src, mode, tgt, delay)
        source = m.get("source", "mic")
        if source not in ("mic", "file"): raise BadConfig("source 는 mic|file")
        rate = int(_num(m.get("sample_rate", 48000), "sample_rate", 8000, 96000, int))
        fmt = m.get("format", "f32le")
        if fmt not in ("f32le", "s16le"): raise BadConfig("format 은 f32le|s16le")
        o = dict(DEFAULT_OPTIONS); raw = m.get("options") or {}
        if not isinstance(raw, dict): raise BadConfig("options 는 객체여야 합니다")
        if "segmentation" in raw:
            if raw["segmentation"] not in ("vad", "none"): raise BadConfig("segmentation 은 vad|none")
            o["segmentation"] = raw["segmentation"]
        for k, lo, hi, typ in (("pre_roll_ms", 160, 960, int), ("end_pause_ms", 300, 2000, int), ("hard_max_s", 10, 30, float)):
            if k in raw: o[k] = _num(raw[k], k, lo, hi, typ)
        tmin = max(0.5, 0.08 * (delay + 2))
        if "tail_s" in raw: o["tail_s"] = _num(raw["tail_s"], f"tail_s(δ{delay})", round(tmin, 3), 1.5, float)
        if "gain" in raw:
            if raw["gain"] not in ("auto", "off"): raise BadConfig("gain 은 auto|off")
            o["gain"] = raw["gain"]
        for k in ("lead_in", "final_redecode", "debug_tokens", "save_audio"):
            if k in raw: o[k] = bool(raw[k])
        if "speed" in raw: o["speed"] = _num(raw["speed"], "speed", 0, 1000, float)
        if "file_name" in raw: o["file_name"] = str(raw["file_name"])[:200]
        if "file_duration_s" in raw: o["file_duration_s"] = _num(raw["file_duration_s"], "file_duration_s", 0, 1e6, float)
        if o["segmentation"] == "none":                  # 평가 방식: 파일 전체 + 1.0 s 0, 이득·lead-in 없음
            if source != "file": raise BadConfig("분할 없음(평가 방식)은 파일 재생에서만 쓸 수 있습니다")
            if float(o.get("file_duration_s") or 0) > 30.0: raise BadConfig("분할 없음(평가 방식)은 30 초 이하 파일만 됩니다")
            o.update(gain="off", lead_in=False, tail_s=1.0)
        try:
            seg_params(o).check(delay)
        except ParamError as e:
            raise BadConfig(str(e))
        return dict(src_lang=src, mode=mode, tgt_lang=tgt, delay=delay, source=source, sample_rate=rate, format=fmt, options=o)

    # ─ 웹소켓
    def _origin_ok(self, req) -> bool:
        o = req.headers.get("Origin")
        if not o: return True                     # CLI 클라이언트
        try:
            u = urlsplit(o)
        except ValueError:
            return False
        return u.hostname in ("127.0.0.1", "localhost", "::1") or u.netloc == req.host

    async def h_ws(self, req):
        if not self._origin_ok(req):
            raise web.HTTPForbidden(text="origin")
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20, compress=False)
        await ws.prepare(req)
        conn = Conn(ws, req.remote); self.conns.add(conn)
        conn.send(self.hello(conn))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    self.on_binary(conn, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    self.on_text(conn, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.conns.discard(conn)
            for s in list(self.sessions.values()):
                if s.conn is conn: s.abort("disconnected", notify=False)
            await conn.close()
        return ws

    def session_of(self, conn, states=("active",)):
        for s in self.sessions.values():
            if s.conn is conn and s.state in states: return s
        return None

    def on_binary(self, conn, data):
        s = self.session_of(conn)
        if s is None:
            if not conn.bin_warned:
                conn.bin_warned = True
                conn.send(warn("binary_outside_session", "세션 밖에서 받은 오디오는 무시합니다", bytes=len(data)))
            return
        s.push_bytes(data)

    def on_text(self, conn, data):
        try:
            m = json.loads(data)
            if not isinstance(m, dict): raise ValueError
        except ValueError:
            conn.send(err("bad_config", "JSON 객체가 아닙니다")); return
        t = m.get("type")
        try:
            if t == "start": self.on_start(conn, m)
            elif t == "update":
                s = self.session_of(conn)
                if s is None: conn.send(err("bad_config", "진행 중인 세션이 없습니다"))
                else: s.update(m)
            elif t == "stop":
                s = self.session_of(conn)
                if s: s.stop()
            elif t in ("abort", "reset"):
                for s in list(self.sessions.values()):
                    if s.conn is conn: s.abort("aborted")
            elif t == "ping":
                conn.send(dict(type="pong", t=m.get("t"), server_ms=round(time.time() * 1000)))
            else:
                conn.send(err("bad_config", f"알 수 없는 메시지: {t!r}"))
        except BadConfig as e:
            conn.send(err("bad_config", str(e)))
        except Exception as e:
            log.exception("메시지 처리 실패")
            conn.send(err("internal", f"{type(e).__name__}: {e}"))

    def on_start(self, conn, m):
        if self.closing.is_set():
            conn.send(err("model_not_ready", "서버가 종료 중입니다")); return
        if self.state != "ready":
            conn.send(err("model_not_ready", "모델이 아직 준비되지 않았습니다" if self.state == "loading" else f"모델 오류: {self.state_msg}"))
            return
        cfg = self.validate_start(m)
        old = self.session_of(conn)
        if old: old.stop()                              # 같은 소켓: stop 후 start
        if self.active and self.active.conn is not conn and self.active.state == "active":
            if not m.get("takeover"):
                conn.send(err("busy", "다른 탭에서 사용 중입니다", busy=True)); return
            prev = self.active
            prev.send(err("preempted", "다른 탭이 세션을 가져갔습니다", fatal=True)); prev.abort("preempted")
        conn.bin_warned = False
        s = Session(self, conn, cfg)
        self.sessions[s.sid] = s; self.active = s
        s.begin()
        log.info("세션 시작 %s %s", s.sid, {k: cfg[k] for k in ("src_lang", "mode", "tgt_lang", "delay", "source", "sample_rate")})


# ── 진입점 ──────────────────────────────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="VAPKT 실시간 전사·번역 서버 (MLX)")
    ap.add_argument("--model", default=os.environ.get("VAPKT_APP_MODEL"), help="s4 체크포인트 폴더 (<dir> 또는 <dir>/final)")
    ap.add_argument("--port", type=int, default=None, help="기본: 8796–8805 중 첫 빈 포트 (8765·8790 제외)")
    ap.add_argument("--new-instance", action="store_true", help="이미 실행 중인 VAPKT 가 있어도 새로 띄운다 (모델 사본이 하나 더 생긴다)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--kv-capacity", type=int, default=2048)
    ap.add_argument("--engine", choices=("mlx", "fake"), default="mlx")
    ap.add_argument("--precision", choices=("auto", "fp32", "fp16"), default="auto")
    ap.add_argument("--encoder-precision", choices=("auto", "fp32", "fp16"), default="auto")
    ap.add_argument("--no-speculative", action="store_true")
    ap.add_argument("--draft", choices=("auto", "q4", "off"), default="auto",
                    help="q4 자기 초안 검증 디코드: auto = 시작 때 재서 이득일 때만 (결과 id 는 같다), q4 = 늘, off = 끔")
    ap.add_argument("--save-audio", action="store_true", help="세션 입력 오디오를 ~/.vapkt-app/sessions/<id>.wav 로 저장")
    ap.add_argument("--fake-round-ms", type=float, default=0.0, help="fake 엔진의 라운드당 지연(ms)")
    ap.add_argument("--fake-spec", default="", help='fake 엔진 옵션, 예: "round_ms=120,hangul=1,asr_trans=1"')
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args(argv)


def setup_logging(debug: bool):
    os.makedirs(os.path.join(HOME, "logs"), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s", "%H:%M:%S")
    root = logging.getLogger(); root.setLevel(logging.DEBUG if debug else logging.INFO)
    sh = logging.StreamHandler(sys.stderr); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.FileHandler(os.path.join(HOME, "logs", f"server-{_dt.datetime.now():%Y%m%d-%H%M%S}.log"), encoding="utf-8")
    fh.setFormatter(fmt); root.addHandler(fh)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


async def amain(args):
    errs, warns = env_checks(args.engine)
    for w in warns: log.warning("%s", w)
    if errs:
        for e in errs: print(f"[오류] {e}", file=sys.stderr)
        return 2
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("%s 로 엽니다 — 브라우저는 http 의 localhost 가 아니면 마이크를 막습니다", args.host)
    loop = asyncio.get_running_loop()
    if args.port is None and not args.new_instance:         # 이미 떠 있으면 그 주소를 연다 (모델 사본을 하나 더 만들지 않는다)
        hit = await loop.run_in_executor(None, find_running, args.host, candidate_ports(None))
        if hit is not None:
            return await reuse_running(args, *hit)
    sock, port = bind_port(args.host, args.port)              # 엔진 적재(runner.setup) 전에: 포트가 없으면 적재도 시작하지 않는다
    srv = Server(args)
    srv.port = port
    runner = web.AppRunner(srv.make_app(), access_log=None, handle_signals=False)
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):   # SIGHUP: 터미널 창을 닫을 때 (pty 가 닫히면 두 번 올 수 있다)
        loop.add_signal_handler(sig, make_signal_handler(srv, stop, force_on_repeat=sig == signal.SIGINT))
    try:
        await runner.setup()
        await web.SockSite(runner, sock).start()
        host = "127.0.0.1" if args.host in ("0.0.0.0", "localhost") else args.host
        url = f"http://{host}:{port}/"
        say(f"VAPKT 실시간 전사·번역: {url}  (엔진 {args.engine}, 종료 Ctrl-C)")
        if not args.no_browser:
            loop.run_in_executor(None, webbrowser.open, url)
        await stop.wait()
        say("종료 중…")
    finally:
        srv.closing.set()
        try:
            await runner.cleanup()                        # 세션 로그·오디오를 닫고 워커를 멈춘다 (tty 가 죽어도 건너뛰지 않는다)
        except Exception:
            log.exception("종료 정리 실패")
    if srv.worker.thread.is_alive():
        # 적재·워밍업 중이던 워커(daemon) — 인터프리터 정리가 MLX/Metal 안의 스레드와 겹치지 않게 바로 끝낸다
        log.info("워커가 아직 적재 중 — 바로 끝냅니다")
        logging.shutdown()
        os._exit(0)
    return 0


def make_signal_handler(srv: "Server", stop: asyncio.Event, force_on_repeat=True, hard_exit=os._exit):
    """첫 신호: 종료를 시작한다. 종료 중 다시 온 Ctrl-C(force_on_repeat): 기다리지 않고 바로 끝낸다 (130).
    SIGHUP·SIGTERM 은 반복돼도 정리를 끝까지 한다 (터미널 창을 닫으면 SIGHUP 이 두 번 온다)."""
    def on_signal():
        if srv.closing.is_set():
            if force_on_repeat:
                log.warning("종료 신호를 다시 받아 바로 끝냅니다")
                hard_exit(130)
            return
        srv.closing.set(); stop.set()
    return on_signal


async def reuse_running(args, port: int, health: dict) -> int:
    """이미 실행 중인 VAPKT 를 쓴다: 설정이 같으면 그 주소를 열고 0, 명령줄 옵션이 다르면 알리고 1."""
    url = f"http://127.0.0.1:{port}/"
    bad = option_conflicts(args, health)
    if bad:
        say(f"[오류] VAPKT 가 다른 설정으로 이미 실행 중입니다: {url}\n  다른 옵션: {', '.join(bad)}\n"
            f"  그 서버를 끝내거나(그 터미널에서 Ctrl-C) --new-instance 로 따로 띄우세요 (모델 사본이 하나 더 생깁니다)", err=True)
        return 1
    say(f"이미 실행 중: {url}  (상태 {health.get('state')}; 새로 띄우려면 --new-instance)")
    if not args.no_browser:
        await asyncio.get_running_loop().run_in_executor(None, webbrowser.open, url)
    return 0


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.debug)
    sys.exit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
