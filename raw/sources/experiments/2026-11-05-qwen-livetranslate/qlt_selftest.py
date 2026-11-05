#!/usr/bin/env python
"""API 없이 Qwen 평가 경로를 점검한다(맥 로컬, 2026-11-05).

  mock   : python qlt_selftest.py mock [--port 8765] [--drop-after S]
           가짜 WebSocket 서버(qlt_client.py 를 QLT_WS_URL=ws://127.0.0.1:<port> 로 돌려 배선 — 속도·로그·session.finish·끊김 뒤 이어 가기).
           3 s 마다 item 하나(화자 0/1 번갈아)를 만들어 speech_started·stopped·원문·번역 조각을 보낸다. --drop-after S: 연결마다 오디오 S 초 뒤 끊는다.
  oracle : python qlt_selftest.py oracle-ami --data <ami-bench-v1> --tok <토크나이저> --out <로그 폴더> [--subset eval] [--corpora …]
           python qlt_selftest.py oracle-taxi --sessions <…/sessions.jsonl> --out <로그 폴더>
           참조 발화를 그대로 Qwen 이벤트 로그 꼴로 쓴다(화자·구간·원문 = 정답, 번역 = 참조 번역, 받은 시각 = 발화 끝 + 1 s).
           qlt_ami.py·qlt_taxi.py score 를 이 로그로 돌리면 cp·agn 이 거의 0, DER 이 거의 0, BLEU 가 거의 100 이어야 한다 — 변환·채점 경로 확인.
           (AMI 글자·TAXI 글자는 T5 결과 폴더에만 쓴다.)"""
import argparse, asyncio, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "experiments")]


async def mock(a):
    import websockets
    from websockets.asyncio.server import serve

    async def handler(ws):
        recv_s = 0.0; nid = [0]; tgt = "en"; spk = 0; last = 0.0; pend = []
        async def send(**ev): await ws.send(json.dumps(ev))
        try:
            async for m in ws:
                ev = json.loads(m); ty = ev.get("type")
                if ty == "session.update":
                    tgt = ev["session"]["translation"]["language"]; await send(type="session.created", session={}); await send(type="session.updated", session=ev["session"])
                elif ty == "input_audio_buffer.append":
                    import base64
                    recv_s += len(base64.b64decode(ev["audio"])) / 2 / 16000
                    if a.drop_after and recv_s >= a.drop_after: await ws.close(code=1011, reason="mock drop"); return
                    if recv_s - last >= 3.0:
                        nid[0] += 1; it = f"item_{nid[0]}"; tr = f"item_tr_{nid[0]}"; s0, s1 = int(last * 1000) + 200, int(recv_s * 1000) - 300
                        await send(type="input_audio_buffer.speech_started", audio_start_ms=s0, item_id=it, speaker_id=spk)
                        await send(type="input_audio_buffer.speech_stopped", audio_end_ms=s1, item_id=it)
                        await send(type="conversation.item.input_audio_transcription.completed", item_id=it, transcript=f"hello number {nid[0]} from speaker {spk}", language="en", emotion="neutral")
                        await send(type="conversation.item.created", previous_item_id=it, item=dict(id=tr, role="assistant", type="message"))
                        for w in ["hallo", "nummer", str(nid[0])]: await send(type="response.text.delta", item_id=tr, delta=w + " ")
                        await send(type="response.text.done", item_id=tr, text=f"hallo nummer {nid[0]}")
                        await send(type="response.done", response=dict(usage=dict(total_tokens=30, input_tokens=21, output_tokens=9)))
                        spk = 1 - spk; last = recv_s
                elif ty == "session.finish":
                    await send(type="session.finished"); return
        except websockets.exceptions.ConnectionClosed:
            return

    async with serve(handler, "127.0.0.1", a.port, max_size=None):
        print(f"mock ws://127.0.0.1:{a.port}", flush=True); await asyncio.Future()


def W(fo, t, ev, seg=0): fo.write(json.dumps(dict(t=round(t, 3), a=round(t, 3), seg=seg, ev=ev), ensure_ascii=False) + "\n")


def oracle_ami(a):
    from transformers import AutoTokenizer
    from cst_stage3_eval import windows_of
    from qlt_ami import safe
    tok = AutoTokenizer.from_pretrained(a.tok)
    ns = argparse.Namespace(data=a.data, split=None, corpora=a.corpora, deltas=[10], limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0, max_speakers=4,
                            window_draws=0, declare=4, xl_order="lang", subset=a.subset)
    sets, items = windows_of(ns, tok); os.makedirs(a.out, exist_ok=True)
    for c, i in items:
        ds = sets[c]; cid = ds.items[i][0]; d = ds.dlgs[cid]; sp = {s: k for k, s in enumerate(sorted({u.speaker for u in d.utterances}))}; b = os.path.join(a.out, safe(cid))
        with open(b + ".events.jsonl", "w", encoding="utf-8") as fo:
            W(fo, 0, dict(type="_start", id=cid, target="de", model="oracle", speed=1.0, chunk_ms=100))
            for n, u in enumerate(sorted(d.utterances, key=lambda u: u.start)):
                it = f"item_{n}"; W(fo, u.start, dict(type="input_audio_buffer.speech_started", audio_start_ms=int(u.start * 1000), item_id=it, speaker_id=sp[u.speaker]))
                W(fo, u.end, dict(type="input_audio_buffer.speech_stopped", audio_end_ms=int(u.end * 1000), item_id=it))
                txt = tok.decode([tid for tid, _ in u.tokens]).strip() if u.tokens else (u.raw or "")   # 다듬은 대화는 나뉜 조각의 text 가 비고 단어는 토큰에만 있다
                W(fo, u.end + 1.0, dict(type="conversation.item.input_audio_transcription.completed", item_id=it, transcript=txt, language="en"))
            W(fo, d.duration_s, dict(type="_end", segments=1, usage={}))
        json.dump(dict(id=cid, finished=True, oracle=True), open(b + ".done.json", "w"))
        print(f"  {cid}: 발화 {len(d.utterances)}", flush=True)


def oracle_taxi(a):
    from qlt_taxi import safe, CODE
    os.makedirs(a.out, exist_ok=True)
    for l in open(a.sessions, encoding="utf-8"):
        r = json.loads(l); sp = {s: k for k, s in enumerate(sorted(r["speakers"]))}
        for lg in sorted({s["lang"] for s in r["speakers"].values()}):
            b = os.path.join(a.out, safe(f"{r['session_id']}#to-{CODE[lg]}"))
            with open(b + ".events.jsonl", "w", encoding="utf-8") as fo:
                W(fo, 0, dict(type="_start", id=r["session_id"], target=CODE[lg], model="oracle"))
                for n, t in enumerate(sorted(r["turns"], key=lambda t: t["start_s"])):
                    it, tr = f"item_{n}", f"item_tr_{n}"
                    W(fo, t["start_s"], dict(type="input_audio_buffer.speech_started", audio_start_ms=int(t["start_s"] * 1000), item_id=it, speaker_id=sp[t["speaker"]]))
                    W(fo, t["end_s"], dict(type="input_audio_buffer.speech_stopped", audio_end_ms=int(t["end_s"] * 1000), item_id=it))
                    W(fo, t["end_s"] + 0.5, dict(type="conversation.item.input_audio_transcription.completed", item_id=it, transcript=t["transcript"], language=CODE[t["lang"]]))
                    W(fo, t["end_s"] + 0.5, dict(type="conversation.item.created", previous_item_id=it, item=dict(id=tr, role="assistant")))
                    txt = t["translation"] if t["lang"] != lg else t["transcript"]                  # 같은 언어 목표면 원문을 되풀이한다고 둔다(방향 고르기가 걸러야 한다)
                    W(fo, t["end_s"] + 1.0, dict(type="response.text.delta", item_id=tr, delta=txt + " "))
                    W(fo, t["end_s"] + 1.0, dict(type="response.text.done", item_id=tr, text=txt))
                W(fo, r["duration_s"], dict(type="_end", segments=1, usage={}))
            json.dump(dict(id=b, finished=True, oracle=True), open(b + ".done.json", "w"))
    print("oracle-taxi 끝", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("mock"); q.add_argument("--port", type=int, default=8765); q.add_argument("--drop-after", type=float, default=0.0)
    q = sub.add_parser("oracle-ami"); q.add_argument("--data", required=True); q.add_argument("--tok", required=True); q.add_argument("--out", required=True)
    q.add_argument("--subset", default="eval"); q.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    q = sub.add_parser("oracle-taxi"); q.add_argument("--sessions", required=True); q.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "mock": asyncio.run(mock(a))
    elif a.cmd == "oracle-ami": oracle_ami(a)
    else: oracle_taxi(a)


if __name__ == "__main__":
    main()
