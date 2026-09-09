#!/usr/bin/env python
"""실시간 음성인식 웹 서버(aiohttp + WebSocket). 브라우저가 마이크 PCM(Int16 16 kHz) 을 80 ms 단위로 보내면 청크마다 전사 이벤트를 돌려준다.
  실행(컨테이너, GPU):  CUDA_VISIBLE_DEVICES=4 python experiments/live/server.py --model /soundai/Model/VAPASR/hf-C2/final --port 8765
  로컬 브라우저:       ssh -L 8765:<컨테이너 IP>:8765 mxc  →  http://localhost:8765   (마이크 권한은 localhost 에서 허용된다)
프로토콜: 클라이언트가 먼저 JSON {"type":"start","lang":"Korean","delay":4,"next_bias":0} 을 보내고, 이어서 바이너리 프레임(Int16 LE PCM 16 kHz, 길이 자유)을 보낸다.
         서버는 청크마다 {"type":"chunk","k":…,"t":초,"tokens":[…],"text":전체 전사,"tick_ms":…,"forced":…} 을, {"type":"stop"} 을 받으면 flush 후 {"type":"final",…} 을 보낸다.
delay(δ) 는 세션 시작 시 정한다(기본 4). 세션 중 바꾸려면 stop → start."""
import os, sys, json, time, argparse, asyncio, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.environ.get("VAPASR_LIVE_MODEL", "/soundai/Model/VAPASR/hf-C2/final")); ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--device", default="cuda"); ap.add_argument("--default-delay", type=int, default=4); ap.add_argument("--sync", action="store_true", help="GPU 작업을 이벤트 루프 스레드에서 직접(스레드 풀 없이)"); ap.add_argument("--fp32", action="store_true", help="thinker 를 fp32 로(기본 bf16: 디코드 2 배 빠름)"); a = ap.parse_args()
from aiohttp import web, WSMsgType             # uvicorn 은 websockets/wsproto 가 없으면 WS 업그레이드에 404 를 준다(env 에 미설치) → aiohttp 자체 WS 서버 사용
from vapasr.hf.infer import load_model
from vapasr.hf.live import LiveSession, SR

t0 = time.time(); model, tok = load_model(a.model, device=a.device, dtype=None if a.fp32 else torch.bfloat16); print(f"model ready ({time.time()-t0:.0f}s, thinker {'fp32' if a.fp32 else 'bf16'})", flush=True)
lock = threading.Lock()                       # GPU 는 세션 하나씩(동시 접속은 순서대로)
_w = LiveSession(model, tok, lang="Korean", delay=a.default_delay); _w.feed(np.zeros(SR, np.float32)); _w.finish(); del _w; print("warmup done", flush=True)   # 첫 세션의 커널 컴파일·autotune(14 s) 을 미리
async def gpu(loop, fn, *args):
    if a.sync: return fn(*args)
    with lock: return await loop.run_in_executor(None, fn, *args)
HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8").read()
MODEL_NAME = os.path.basename(os.path.dirname(a.model.rstrip("/"))) + "/" + os.path.basename(a.model.rstrip("/"))

async def index(req): return web.Response(text=HTML.replace("__DEFAULT_DELAY__", str(a.default_delay)).replace("__MODEL__", MODEL_NAME), content_type="text/html")

async def ws_handler(req):
    sock = web.WebSocketResponse(max_msg_size=0, heartbeat=20); await sock.prepare(req); sess = None; loop = asyncio.get_event_loop(); n_samples = 0; t_start = None
    def ev_json(e, kind="chunk"):
        return json.dumps(dict(type=kind, k=e.k, t=round(e.k * 0.08, 2), tokens=tok.convert_ids_to_tokens(e.ids), text=e.text, tick_ms=round(e.tick_ms, 1), enc_ms=round(e.enc_ms, 1), dec_ms=round(e.dec_ms, 1), forced=e.forced), ensure_ascii=False)
    try:
        async for msg in sock:
            if msg.type == WSMsgType.BINARY:
                if sess is None: continue
                pcm = np.frombuffer(msg.data, dtype="<i2").astype(np.float32) / 32768.0; n_samples += len(pcm)
                events = await gpu(loop, sess.feed, pcm)
                for e in events: await sock.send_str(ev_json(e))
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "start":
                    lang = m.get("lang", "Korean"); delay = int(m.get("delay", a.default_delay)); nb = float(m.get("next_bias", 0.0))
                    sess = await gpu(loop, lambda: LiveSession(model, tok, lang=lang, delay=delay, next_bias=nb))
                    n_samples = 0; t_start = time.time(); await sock.send_str(json.dumps(dict(type="started", lang=lang, delay=delay, next_bias=nb)))
                elif m.get("type") == "stop" and sess is not None:
                    events = await gpu(loop, sess.finish)
                    for e in events[:-1]: await sock.send_str(ev_json(e))
                    await sock.send_str(json.dumps(dict(type="final", text=sess.text(), k=sess.k, audio_s=round(n_samples / SR, 2), wall_s=round(time.time() - t_start, 2) if t_start else None), ensure_ascii=False)); sess = None
            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE): break
    except Exception as e:
        try: await sock.send_str(json.dumps(dict(type="error", message=f"{type(e).__name__}: {e}")))
        except Exception: pass
        raise
    return sock

app = web.Application(); app.add_routes([web.get("/", index), web.get("/ws", ws_handler)])
if __name__ == "__main__": print(f"serving http://{a.host}:{a.port}", flush=True); web.run_app(app, host=a.host, port=a.port, print=None)
