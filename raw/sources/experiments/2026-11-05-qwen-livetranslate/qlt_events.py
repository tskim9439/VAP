"""Qwen3.8-LiveTranslate 원본 로그(qlt_client.py 의 <id>.events.jsonl) 읽기 — qlt_ami.py·qlt_taxi.py 가 같이 쓴다(맥 로컬 전용, 2026-11-05).

parse(path) → dict(
  items:  원문 item 마다 {id, seg, speaker("s<seg>_<speaker_id>"), start, end(오디오 s — audio_start_ms·audio_end_ms + 그 연결의 오디오 시작 위치), transcript, language,
          t_started·t_stopped·t_done(받은 시각을 오디오 시각으로 — 그 연결의 _stream 순간 = 그때 보낸 오디오 위치, 벽시계 경과 × 속도), tr_items[번역 item id]},
  번역 조각 시각도 같은 오디오 시각이다(연결 준비·재연결 공백이 지연에 섞이지 않는다 — 네트워크 왕복은 섞인다).
  trans:  번역 item 마다 {id, src(원문 item id), seg, deltas[(받은 시각 s, 조각 글자)], text(done 이 있으면 그것)},
  meta:   _start 설정 · _end 요약 · rtt[] · closed[]).
화자 번호는 연결(seg)마다 새로 매겨지므로 "s<seg>_<번호>" 로 둔다(끊김이 없으면 seg 는 0 하나)."""
import json


def parse(path):
    items, trans, order, meta = {}, {}, [], dict(rtt=[], closed=[])
    off = {}                                                                                      # 연결마다 audio_*_ms 의 0 = 그 연결을 열 때까지 보낸 오디오(_seg)
    strm = {}                                                                                     # 연결마다 (_stream 벽시계, 오디오 위치, 속도) — 받은 시각 t 를 오디오 시각으로 바꾼다
    for line in open(path, encoding="utf-8"):
        try: r = json.loads(line)
        except ValueError: continue
        ev, t, seg = r["ev"], r["t"], r.get("seg", 0); ty = ev.get("type", "")
        if ty == "_stream": strm[seg] = (t, ev["offset_s"], ev.get("speed", 1.0))
        if seg in strm: t = round(strm[seg][1] + (t - strm[seg][0]) * strm[seg][2], 4)          # 받은 시각을 오디오 시각으로(그 연결이 오디오를 흘리기 시작한 순간 기준)
        if ty == "_seg": off[ev["seg"]] = ev["offset_s"]
        if ty in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped") and seg not in off: off[seg] = 0.0   # _seg 전 로그(호환)
        o = off.get(seg, 0.0)
        if ty == "_start": meta["start"] = ev
        elif ty == "_end": meta["end"] = ev
        elif ty == "_rtt": meta["rtt"].append(ev["s"])
        elif ty == "_closed": meta["closed"].append(ev)
        elif ty == "input_audio_buffer.speech_started":
            k = (seg, ev["item_id"]); it = items.setdefault(k, dict(id=ev["item_id"], seg=seg, tr_items=[]))
            if k not in order: order.append(k)
            spk = ev.get("speaker_id"); it.update(speaker=f"s{seg}_{spk if spk is not None else 'x'}", start=o + ev.get("audio_start_ms", 0) / 1000.0, t_started=t)
        elif ty == "input_audio_buffer.speech_stopped":
            k = (seg, ev["item_id"]); it = items.setdefault(k, dict(id=ev["item_id"], seg=seg, tr_items=[]))
            if k not in order: order.append(k)
            it.update(end=o + ev.get("audio_end_ms", 0) / 1000.0, t_stopped=t)
        elif ty == "conversation.item.input_audio_transcription.completed":
            k = (seg, ev["item_id"]); it = items.setdefault(k, dict(id=ev["item_id"], seg=seg, tr_items=[]))
            if k not in order: order.append(k)
            it.update(transcript=ev.get("transcript", ""), language=ev.get("language"), t_done=t)
        elif ty == "conversation.item.created":
            iid = (ev.get("item") or {}).get("id"); src = ev.get("previous_item_id")
            if iid and (ev.get("item") or {}).get("role") == "assistant":
                trans.setdefault((seg, iid), dict(id=iid, seg=seg, src=src, deltas=[], text=None))["src"] = src
                if (seg, src) in items: items[(seg, src)]["tr_items"].append(iid)
        elif ty in ("response.text.delta", "response.audio_transcript.delta"):
            x = trans.setdefault((seg, ev.get("item_id")), dict(id=ev.get("item_id"), seg=seg, src=None, deltas=[], text=None)); x["deltas"].append((t, ev.get("delta", "")))
        elif ty in ("response.text.done", "response.audio_transcript.done"):
            x = trans.setdefault((seg, ev.get("item_id")), dict(id=ev.get("item_id"), seg=seg, src=None, deltas=[], text=None))
            x["text"] = ev.get("text", ev.get("transcript")); x["t_done"] = t
    out = []
    for k in order:                                                                               # 시작·끝이 빠진 item 은 받은 시각으로 메운다(근사 — 기록)
        it = items[k]; it.setdefault("speaker", f"s{it['seg']}_x"); it.setdefault("transcript", ""); it.setdefault("language", None)
        if "start" not in it: it["start"] = it.get("t_started", it.get("t_done", 0.0)); it["approx"] = True
        if "end" not in it: it["end"] = max(it["start"], it.get("t_stopped", it.get("t_done", it["start"]))); it["approx"] = True
        out.append(it)
    return dict(items=out, trans=list(trans.values()), meta=meta)


def timed_words(text, t0, t1):
    """item 원문 단어를 [t0, t1] 안에 고르게 놓는다(단어 시각이 API 에 없다 — 근사, 마스크·시각순 정렬에만 쓴다)."""
    w = text.split()
    if not w: return []
    step = max(0.0, t1 - t0) / len(w)
    return [(x, round(t0 + step * (i + 0.5), 3)) for i, x in enumerate(w)]


def emitted_words(x):
    """번역 item 조각(deltas) → [(단어, 받은 시각 s)]: 단어는 뒤에 공백이 온(또는 item 이 끝난) 순간에 확정된 것으로 본다.
    done 의 text 가 조각 합과 다르면 done 을 믿고 마지막 받은 시각을 남은 단어에 준다(기록용 차이는 호출 쪽이 센다)."""
    buf, out = "", []
    for t, d in x["deltas"]:
        buf += d
        parts = buf.split(" ")
        for p in parts[:-1]:
            if p.strip(): out.append((p.strip(), t))
        buf = parts[-1]
    tl = x.get("t_done", x["deltas"][-1][0] if x["deltas"] else 0.0)
    if buf.strip(): out += [(p, tl) for p in buf.split()]
    final = x.get("text")
    if final is not None and " ".join(final.split()) != " ".join(w for w, _ in out):
        fw = final.split(); out = [(w, out[i][1] if i < len(out) else tl) for i, w in enumerate(fw)]
    return out
