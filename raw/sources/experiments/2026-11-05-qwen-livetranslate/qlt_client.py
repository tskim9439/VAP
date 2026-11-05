#!/usr/bin/env python
"""Qwen3.8-LiveTranslate 실시간 API 클라이언트(2026-11-05, 사용자 "Qwen 3.8 live translate 도 동일하게 평가해보고, TAXI 에 대해서도 번역 성능을 평가해보고 싶은데
… 절대 mxc 서버에 올리지말고 로컬에서만 진행해야해"). 맥에서만 돈다. 오디오를 실시간 속도로 100 ms 씩 WebSocket 에 흘리고, 받은 서버 이벤트를 전부
받은 시각과 함께 원본 로그로 남긴다(채점은 qlt_ami.py·qlt_taxi.py 가 이 로그만 읽는다 — API 를 다시 부르지 않고 다시 채점할 수 있다).

  python qlt_client.py --jobs <jobs.jsonl> --out <로그 폴더> [--concurrency 8] [--speed 1.0] [--endpoint qwencloud-sg] [--limit N] [--dry-run]
  jobs.jsonl 한 줄 = {"id": "ami-sdm:ES2004a", "audio": "<.npy(float16 mono) | .wav>", "target": "de", "max_s": null}
  API 키: 저장소 .env.local 의 DASHSCOPE_API_KEY(출력하지 않는다). 엔드포인트는 키를 만든 곳에 맞춘다(--endpoint, 또는 QLT_WS_URL 로 직접).

문서(관찰 2026-11-05): https://docs.qwencloud.com/developer-guides/speech/realtime-translation ,
  …/api-reference/speech-translation/livetranslate-realtime/{client-events,server-events}
  - 입력 16 kHz 16-bit 모노 PCM base64(input_audio_buffer.append), 끝은 session.finish → session.finished 뒤 연결을 닫는다.
  - session.update: output_modalities ["text"](말소리 출력 끔), audio.input.turn_detection speaker_detection(기본, threshold 0.5 고정), translation.language = 목표 언어.
  - 받는 것: input_audio_buffer.speech_started(audio_start_ms·item_id·speaker_id) · speech_stopped(audio_end_ms), conversation.item.input_audio_transcription.completed
    (transcript·language — 원문은 늘 켜짐), conversation.item.created(previous_item_id = 원문 item → 번역 item), response.text.delta·done, response.done(usage).
로그(<out>/<id>.events.jsonl, 한 줄 = {"t": 스트림 시작부터 벽시계 s, "a": 그때까지 보낸 오디오 s, "seg": 연결 번호, "ev": 서버 이벤트}).
  우리 쪽 기록은 ev.type 이 "_" 로 시작한다: _start(설정), _seg(연결 시작 — 그때까지 보낸 오디오 s), _stream(오디오를 흘리기 시작한 순간 — 받은 시각을 오디오
  시각으로 바꾸는 0 점), _rtt(WebSocket ping 왕복 s), _closed(예상 밖 끊김), _end(요약). 끝나면 <id>.done.json.
끊김: session.finished 전에 연결이 끊기면 보낸 데까지 이어서 새 연결(seg + 1, 최대 --max-reconnect)로 간다 — 화자 번호는 연결마다 새로 매겨지므로 채점은
  seg 를 화자 이름에 붙여 섞지 않는다(서비스 한계로 그대로 둔다). 끝난 job(<id>.done.json)은 건너뛴다. 덜 끝난 로그는 <id>.events.jsonl.partN 으로 옮기고 처음부터 다시 한다.
"""
import argparse, asyncio, base64, json, os, re, sys, time

ENDPOINTS = {
    "qwencloud-sg": "wss://ap-southeast-1.maas.qwencloudapi.com/api-ws/v1/realtime",
    "qwencloud-cn": "wss://cn-beijing.maas.qwencloudapi.com/api-ws/v1/realtime",
    "modelstudio-intl": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
    "modelstudio-cn": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
}
SR = 16000; CHUNK = 1600                                                                          # 100 ms(문서 예제와 같은 크기)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))


def api_key() -> str:
    """.env.local 의 DASHSCOPE_API_KEY(환경 변수가 있으면 그것) — 값은 출력하지 않는다."""
    k = os.environ.get("DASHSCOPE_API_KEY")
    if not k and os.path.exists(os.path.join(REPO, ".env.local")):
        for line in open(os.path.join(REPO, ".env.local"), encoding="utf-8"):
            m = re.match(r"\s*(?:export\s+)?DASHSCOPE_API_KEY\s*=\s*(.*)\s*$", line)
            if m: k = m.group(1).strip().strip("'\"")
    if not k: sys.exit(".env.local 에 DASHSCOPE_API_KEY 가 없다")
    return k


def load_pcm(path: str, max_s=None) -> bytes:
    """오디오 → 16 kHz int16 little-endian 바이트. .npy 는 우리 평가 mono 캐시(float16/32), .wav 는 16 kHz 모노만."""
    import numpy as np
    if path.endswith(".npy"): x = np.load(path, mmap_mode="r").astype(np.float32)
    else:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=False); assert sr == SR and x.ndim == 1, f"{path}: {sr} Hz {x.shape} — 16 kHz 모노만"
    if max_s: x = x[: int(max_s * SR)]
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def safe(i: str) -> str: return re.sub(r"[^A-Za-z0-9._-]+", "_", i)


async def run_job(job, a, url, key, sem):
    import websockets
    from websockets.exceptions import ConnectionClosed
    base = os.path.join(a.out, safe(job["id"])); done_p = base + ".done.json"; log_p = base + ".events.jsonl"
    if os.path.exists(done_p): print(f"  {job['id']}: 이미 끝", flush=True); return json.load(open(done_p))
    async with sem:
        if os.path.exists(log_p):                                                                 # 덜 끝난 로그는 지우지 않고 옮긴다
            n = 1
            while os.path.exists(f"{log_p}.part{n}"): n += 1
            os.rename(log_p, f"{log_p}.part{n}")
        pcm = load_pcm(job["audio"], job.get("max_s")); n_ch = (len(pcm) // 2 + CHUNK - 1) // CHUNK; dur = len(pcm) / 2 / SR
        fo = open(log_p, "a", buffering=1, encoding="utf-8"); t0 = time.monotonic(); sent = 0; seg = 0; usage = {}; errors = []; finished = False
        W = lambda ev: fo.write(json.dumps(dict(t=round(time.monotonic() - t0, 4), a=round(sent * CHUNK / SR, 3), seg=seg, ev=ev), ensure_ascii=False) + "\n")
        W(dict(type="_start", id=job["id"], audio=os.path.abspath(job["audio"]), target=job["target"], duration_s=round(dur, 3), url=url, speed=a.speed,
               chunk_ms=CHUNK * 1000 // SR, model=a.model, started=time.strftime("%Y-%m-%d %H:%M:%S")))
        print(f"  {job['id']}: 시작 {dur / 60:.1f} 분 → {job['target']}", flush=True)
        while not finished and seg <= a.max_reconnect:
            try:
                async with websockets.connect(f"{url}?model={a.model}", additional_headers={"Authorization": f"Bearer {key}"}, max_size=None,
                                              open_timeout=30, ping_interval=None) as ws:
                    W(dict(type="_seg", seg=seg, offset_s=round(sent * CHUNK / SR, 3)))                 # 새 연결의 audio_*_ms 는 이 오디오 위치부터 다시 센다
                    await ws.send(json.dumps(dict(type="session.update", session=dict(
                        output_modalities=["text"],
                        audio=dict(input=dict(format=dict(type="pcm", sample_rate=SR), turn_detection=dict(type="speaker_detection", threshold=0.5))),
                        translation=dict(language=job["target"])))))
                    ready = asyncio.Event(); fin = asyncio.Event()

                    async def recv():
                        nonlocal usage
                        async for m in ws:
                            if isinstance(m, bytes): continue
                            ev = json.loads(m); W(ev); ty = ev.get("type", "")
                            if ty == "session.updated": ready.set()
                            elif ty == "error": errors.append(ev.get("error")); print(f"  !! {job['id']} 오류: {json.dumps(ev.get('error'), ensure_ascii=False)[:300]}", flush=True)
                            elif ty == "response.done":
                                for k_, v_ in ((ev.get("response") or {}).get("usage") or {}).items():
                                    if isinstance(v_, (int, float)): usage[k_] = usage.get(k_, 0) + v_
                            elif ty == "session.finished": fin.set(); return

                    async def rtt():
                        while True:
                            await asyncio.sleep(10)
                            try: q = time.monotonic(); await asyncio.wait_for(await ws.ping(), 10); W(dict(type="_rtt", s=round(time.monotonic() - q, 4)))
                            except Exception: return

                    rt = asyncio.create_task(recv()); pt = asyncio.create_task(rtt())
                    try:
                        try: await asyncio.wait_for(ready.wait(), 20)
                        except asyncio.TimeoutError:
                            raise RuntimeError(f"session.updated 를 받지 못했다(오류 {errors[-1:] if errors else '없음'})")
                        tz = time.monotonic() - sent * CHUNK / SR / a.speed                          # 이어 가는 연결도 벽시계 = 오디오 위치가 되게
                        W(dict(type="_stream", seg=seg, offset_s=round(sent * CHUNK / SR, 3), speed=a.speed))   # 받은 시각 → 오디오 시각 환산의 0 점(연결 준비 시간 제외)
                        while sent < n_ch:
                            if rt.done(): break
                            wait = tz + sent * CHUNK / SR / a.speed - time.monotonic()
                            if wait > 0: await asyncio.sleep(wait)
                            b = pcm[sent * CHUNK * 2: (sent + 1) * CHUNK * 2]
                            await ws.send(json.dumps(dict(type="input_audio_buffer.append", audio=base64.b64encode(b).decode())))
                            sent += 1
                        if sent >= n_ch and not rt.done():
                            await ws.send(json.dumps(dict(type="session.finish")))
                            try: await asyncio.wait_for(fin.wait(), a.finish_timeout)
                            except asyncio.TimeoutError: W(dict(type="_closed", reason=f"session.finished 를 {a.finish_timeout} s 안에 받지 못함"))
                            finished = True
                        if rt.done() and not rt.cancelled() and rt.exception(): raise rt.exception()
                        if not finished: raise ConnectionClosed(None, None)
                    finally:                                                                         # 남은 태스크를 거둔다(끊김 때 "never retrieved" 경고 없이)
                        for t_ in (pt, rt):
                            if not t_.done(): t_.cancel()
                        await asyncio.gather(pt, rt, return_exceptions=True)
            except Exception as e:                                                                # 끊김·서버 오류 — 보낸 데까지 이어 새 연결
                if finished: break
                W(dict(type="_closed", seg=seg, error=f"{type(e).__name__}: {str(e)[:300]}")); print(f"  !! {job['id']}: 연결 {seg} 끊김 {type(e).__name__} {str(e)[:200]} (오디오 {sent * CHUNK / SR:.0f} s)", flush=True)
                if any((er or {}).get("type") == "invalid_request_error" for er in errors) and sent == 0: break   # 설정 오류는 다시 해도 같다
                seg += 1; await asyncio.sleep(3)
        rtts = []
        fo.close()
        for l in open(log_p, encoding="utf-8"):
            ev = json.loads(l)["ev"]
            if ev.get("type") == "_rtt": rtts.append(ev["s"])
        rtts.sort(); summ = dict(id=job["id"], target=job["target"], duration_s=round(dur, 3), sent_s=round(sent * CHUNK / SR, 3), finished=finished, segments=seg + 1,
                                  wall_s=round(time.monotonic() - t0, 1), usage=usage, errors=errors[:20], rtt_median_s=rtts[len(rtts) // 2] if rtts else None, n_rtt=len(rtts))
        with open(log_p, "a", encoding="utf-8") as f2: f2.write(json.dumps(dict(t=summ["wall_s"], a=summ["sent_s"], seg=seg, ev=dict(type="_end", **summ)), ensure_ascii=False) + "\n")
        if finished and sent >= n_ch: json.dump(summ, open(done_p, "w"), indent=1, ensure_ascii=False)
        print(f"  {job['id']}: {'끝' if finished else '실패'} · 연결 {seg + 1} · 벽시계 {summ['wall_s'] / 60:.1f} 분 · usage {usage} · RTT 중앙 {summ['rtt_median_s']}", flush=True)
        return summ


async def main_async(a):
    jobs = [json.loads(l) for l in open(a.jobs, encoding="utf-8") if l.strip()]
    if a.only: jobs = [j for j in jobs if re.search(a.only, j["id"])]
    if a.limit: jobs = jobs[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    tot = 0.0
    for j in jobs:
        import numpy as np
        if j["audio"].endswith(".npy"): n = np.load(j["audio"], mmap_mode="r").shape[0]
        else:
            import soundfile as sf
            n = sf.info(j["audio"]).frames
        d = n / SR if not j.get("max_s") else min(n / SR, j["max_s"]); tot += d
    print(f"job {len(jobs)} · 오디오 {tot / 3600:.2f} h · 입력 토큰 어림 {tot * 7 / 1e6:.3f} M(초당 7) · 동시 {a.concurrency} · 속도 {a.speed}× → 벽시계 어림 "
          f"{tot / a.speed / max(1, min(a.concurrency, len(jobs))) / 3600:.2f} h", flush=True)
    if a.dry_run: return
    url = os.environ.get("QLT_WS_URL") or ENDPOINTS[a.endpoint]; key = api_key(); sem = asyncio.Semaphore(a.concurrency)
    res = await asyncio.gather(*(run_job(j, a, url, key, sem) for j in jobs), return_exceptions=True)
    bad = [(j["id"], r) for j, r in zip(jobs, res) if isinstance(r, Exception) or not r.get("finished")]
    for i, r in bad: print(f"  실패 {i}: {r if isinstance(r, Exception) else r.get('errors')}", flush=True)
    print(f"DONE jobs={len(jobs)} failed={len(bad)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.8-livetranslate-flash-realtime")
    ap.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="qwencloud-sg", help="API 키를 만든 곳(QLT_WS_URL 이 있으면 그것)")
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--speed", type=float, default=1.0, help="1.0 = 실시간(지연을 재려면 1.0)")
    ap.add_argument("--max-reconnect", type=int, default=5); ap.add_argument("--finish-timeout", type=float, default=180.0)
    ap.add_argument("--only", default=None, help="id 정규식"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    assert a.speed > 0
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
