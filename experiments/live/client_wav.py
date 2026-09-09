#!/usr/bin/env python
"""마이크 대신 wav/flac 파일을 실시간 속도로 서버에 흘려 넣어 동작을 확인한다(서버와 같은 프로토콜).
  python experiments/live/client_wav.py --url ws://localhost:8765/ws --wav <file> --lang Korean --delay 4 [--speed 1.0]"""
import argparse, asyncio, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="ws://localhost:8765/ws"); ap.add_argument("--wav", required=True); ap.add_argument("--lang", default="Korean"); ap.add_argument("--delay", type=int, default=4)
ap.add_argument("--speed", type=float, default=1.0, help="1.0 = 실시간, 0 = 최대 속도"); a = ap.parse_args()
from vapasr.hf.infer import load_audio
wav = load_audio(a.wav); pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
async def main():
    import aiohttp
    async with aiohttp.ClientSession() as s, s.ws_connect(a.url, max_msg_size=0) as ws:
        await ws.send_str(json.dumps(dict(type="start", lang=a.lang, delay=a.delay))); t0 = time.time(); last = ""
        async def reader():
            nonlocal last
            async for m in ws:
                d = json.loads(m.data)
                if d["type"] == "chunk":
                    if d["text"] != last: print(f"[{time.time()-t0:6.2f}s k={d['k']:4d} {d['tick_ms']:5.1f}ms] {d['text']}", flush=True); last = d["text"]
                elif d["type"] == "final": print(f"FINAL ({d['audio_s']}s 오디오 / {d['wall_s']}s 실시간): {d['text']}", flush=True); return
                elif d["type"] == "error": print("ERROR", d["message"]); return
        r = asyncio.create_task(reader())
        for i in range(0, len(pcm), 1280):
            await ws.send_bytes(pcm[i: i + 1280].tobytes())
            if a.speed > 0: await asyncio.sleep(0.08 / a.speed)
        await ws.send_str(json.dumps(dict(type="stop"))); await r
asyncio.run(main())
