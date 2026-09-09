#!/usr/bin/env python
"""실시간 음성인식 웹 서버(FastAPI + WebSocket). 브라우저가 마이크 PCM(Int16 16 kHz) 을 80 ms 단위로 보내면 청크마다 전사 이벤트를 돌려준다.
  실행(컨테이너, GPU):  CUDA_VISIBLE_DEVICES=4 python experiments/live/server.py --model /soundai/Model/VAPASR/hf-C2/final --port 8765
  로컬 브라우저:       ssh -L 8765:<컨테이너 IP>:8765 mxc  →  http://localhost:8765   (마이크 권한은 localhost 에서 허용된다)
프로토콜: 클라이언트가 먼저 JSON {"type":"start","lang":"Korean","delay":4,"next_bias":0} 을 보내고, 이어서 바이너리 프레임(Int16 LE PCM 16 kHz, 길이 자유)을 보낸다.
         서버는 청크마다 {"type":"chunk","k":…,"t":초,"tokens":[…],"text":전체 전사,"tick_ms":…,"forced":…} 을, {"type":"stop"} 을 받으면 flush 후 {"type":"final",…} 을 보낸다.
delay(δ) 는 세션 시작 시 정한다(기본 4). 세션 중 바꾸려면 stop → start."""
import os, sys, json, time, argparse, asyncio, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.environ.get("VAPASR_LIVE_MODEL", "/soundai/Model/VAPASR/hf-C2/final")); ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--device", default="cuda"); ap.add_argument("--default-delay", type=int, default=4); a = ap.parse_args()
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from vapasr.hf.infer import load_model
from vapasr.hf.live import LiveSession, SR

t0 = time.time(); model, tok = load_model(a.model, device=a.device); print(f"model ready ({time.time()-t0:.0f}s) delays {model.config.delays if hasattr(model.config,'delays') else '?'}", flush=True)
lock = threading.Lock()                       # GPU 는 세션 하나씩(동시 접속은 순서대로)
app = FastAPI(); HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8").read()

@app.get("/")
async def index(): return HTMLResponse(HTML.replace("__DEFAULT_DELAY__", str(a.default_delay)).replace("__MODEL__", os.path.basename(os.path.dirname(a.model.rstrip("/"))) + "/" + os.path.basename(a.model.rstrip("/"))))

@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept(); sess = None; loop = asyncio.get_event_loop(); n_samples = 0; t_start = None
    def ev_json(e, kind="chunk"):
        return json.dumps(dict(type=kind, k=e.k, t=round(e.k * 0.08, 2), tokens=tok.convert_ids_to_tokens(e.ids), text=e.text, tick_ms=round(e.tick_ms, 1), forced=e.forced), ensure_ascii=False)
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect": break
            if msg.get("bytes") is not None:
                if sess is None: continue
                pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0; n_samples += len(pcm)
                with lock: events = await loop.run_in_executor(None, sess.feed, pcm)
                for e in events: await sock.send_text(ev_json(e))
            elif msg.get("text") is not None:
                m = json.loads(msg["text"])
                if m.get("type") == "start":
                    lang = m.get("lang", "Korean"); delay = int(m.get("delay", a.default_delay)); nb = float(m.get("next_bias", 0.0))
                    with lock: sess = await loop.run_in_executor(None, lambda: LiveSession(model, tok, lang=lang, delay=delay, next_bias=nb))
                    n_samples = 0; t_start = time.time(); await sock.send_text(json.dumps(dict(type="started", lang=lang, delay=delay, next_bias=nb)))
                elif m.get("type") == "stop" and sess is not None:
                    with lock: events = await loop.run_in_executor(None, sess.finish)
                    for e in events[:-1]: await sock.send_text(ev_json(e))
                    last = events[-1] if events else None
                    await sock.send_text(json.dumps(dict(type="final", text=sess.text(), k=sess.k, audio_s=round(n_samples / SR, 2), wall_s=round(time.time() - t_start, 2) if t_start else None), ensure_ascii=False)); sess = None
    except WebSocketDisconnect: pass
    except Exception as e: 
        try: await sock.send_text(json.dumps(dict(type="error", message=f"{type(e).__name__}: {e}")))
        except Exception: pass
        raise

if __name__ == "__main__": uvicorn.run(app, host=a.host, port=a.port, log_level="info")
