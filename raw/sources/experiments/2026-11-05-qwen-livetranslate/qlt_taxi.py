#!/usr/bin/env python
"""Qwen3.8-LiveTranslate 를 TAXI(실제 독↔영 전화, cstbench-v0.1 taxi-natural)에서 우리와 같은 채점기로 잰다(맥 로컬 전용, 2026-11-05).
사용자 판단(2026-11-05): "추론만 하는거고 공개적으로 배포하는 것이 아니니 상관없을거 같은데?" — TAXI 오디오를 API 추론에 보낸다. 결과(전사·번역 글자)는
T5 에만 두고 저장소·mxc 에는 집계만 둔다(TAXI 재배포 금지).

  jobs : python qlt_taxi.py jobs --sessions <taxi-natural/sessions.jsonl> --out <jobs.jsonl>
         → 세션마다 두 job(목표 de · 목표 en — API 는 세션 목표 언어가 하나라 두 방향을 따로 돌린다). 오디오 = 세션 mono 혼합(mix.wav, 우리 평가 입력과 같다).
  score: python qlt_taxi.py score --sessions <…> --logs <qlt_client 로그> --out <결과> [--asr-pass en|de]
         → <out>/hyp-d{qwen,qwenref}.jsonl · asr-d{…}.jsonl 을 쓰고 experiments/cst_stage4_taxi.py score 를 그대로 부른다
           (cstbench: BLEU·chrF·StreamLAAL·끝 지연·빈 턴, 받아쓰기 WER·슬롯 순열 WER·cp/agn/귀속 몫 — 우리 모델 TAXI 표와 같은 셈, 이름의 d 자리만 qwen).
방향 고르기: 목표 X 실행의 번역 item 중 원문 item 의 언어가 X 가 아닌 것만 남긴다.
  qwen    = 원문 언어를 Qwen 자신이 감지한 language 로(참조 없음 — 본 수치).
  qwenref = 원문 item 구간과 가장 많이 겹치는 참조 턴의 언어로(정답 방향 — 방향 고르기 오류를 뺀 상한, 참고).
번역 단어 시각 t = 그 단어를 받은 벽시계(스트림 시작부터 s, 실시간 속도라 오디오 시각과 같다 — 네트워크 왕복이 섞인다, RTT 는 따로 적는다).
받아쓰기(asr): --asr-pass 실행의 원문 item. 슬롯 언어 = 그 화자(speaker_id) item 들의 감지 언어 다수결(세션 두 언어 밖이면 남은 언어). 단어 시각은 item 구간에 고르게.
읽기만 하고 아무것도 지우지 않는다(결과 파일은 다시 쓸 때 새로 만든다)."""
import argparse, collections, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path[:0] = [HERE, REPO]
CODE = {"English": "en", "German": "de"}; NAME = {v: k for k, v in CODE.items()}


def safe(i):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", i)


def jobs(a):
    root = os.path.dirname(os.path.abspath(a.sessions)); n = 0
    with open(a.out, "w") as fo:
        for l in open(a.sessions, encoding="utf-8"):
            r = json.loads(l); langs = sorted({s["lang"] for s in r["speakers"].values()})
            for lg in langs:
                fo.write(json.dumps(dict(id=f"{r['session_id']}#to-{CODE[lg]}", audio=os.path.join(root, r["mix"]), target=CODE[lg])) + "\n"); n += 1
    print(f"{n} job → {a.out}")


def lang_name(code):
    return NAME.get((code or "").split("-")[0].lower()[:2]) if code else None


def score(a):
    from qlt_events import parse, timed_words, emitted_words
    recs = [json.loads(l) for l in open(a.sessions, encoding="utf-8")]; os.makedirs(a.out, exist_ok=True)
    H = {k: [] for k in ("qwen", "qwenref")}; A = []; st = collections.Counter(); rtt = []; usage = collections.Counter(); missing = []
    for r in recs:
        sid = r["session_id"]; langs = sorted({s["lang"] for s in r["speakers"].values()}); turns = r["turns"]
        logs = {}
        for lg in langs:
            b = os.path.join(a.logs, safe(f"{sid}#to-{CODE[lg]}"))
            if not os.path.exists(b + ".done.json"): missing.append(f"{sid}#to-{CODE[lg]}"); continue
            logs[lg] = parse(b + ".events.jsonl")
        if len(logs) < len(langs): continue
        for lg, P in logs.items():                                                                # 번역: 목표 lg 실행에서 원문이 lg 가 아닌 item
            rtt += P["meta"]["rtt"]; usage.update({k: v for k, v in ((P["meta"].get("end") or {}).get("usage") or {}).items() if isinstance(v, (int, float))})
            src = {(x["seg"], x["id"]): x for x in P["items"]}
            for x in P["trans"]:
                s = src.get((x["seg"], x["src"])); ws = emitted_words(x)
                if not ws: continue
                if s is None: st["번역 원문 item 없음"] += 1; continue
                det = lang_name(s.get("language"))
                ov = collections.Counter()
                for t in turns: ov[t["lang"]] += max(0.0, min(t["end_s"], s["end"]) - max(t["start_s"], s["start"]))
                ref = ov.most_common(1)[0][0] if ov and ov.most_common(1)[0][1] > 0 else None
                st[f"번역 item 원문 감지 {det}·참조 {ref} → 목표 {lg}"] += 1
                for k, keep in (("qwen", det is not None and det != lg), ("qwenref", ref is not None and ref != lg)):
                    if keep: H[k] += [dict(session=sid, lang=lg, t=round(t, 3), text=w) for w, t in ws]
        P = logs[NAME[a.asr_pass]] if NAME[a.asr_pass] in logs else next(iter(logs.values()))     # 받아쓰기: 한 실행의 원문
        vote = collections.defaultdict(collections.Counter)
        for x in P["items"]:
            if x["transcript"].strip(): vote[x["speaker"]][lang_name(x.get("language"))] += len(x["transcript"].split())
        slot = {}
        for spk, c in vote.items():
            top = [lg for lg, _ in c.most_common() if lg in langs]
            slot[spk] = top[0] if top else None
        for spk in slot:
            if slot[spk] is None: slot[spk] = next((lg for lg in langs if lg not in slot.values()), langs[0]); st["받아쓰기 화자 언어 밖 → 남은 언어"] += 1
        st[f"받아쓰기 화자 수 {len(slot)}"] += 1
        for x in P["items"]:
            for w, t in timed_words(x["transcript"], x["start"], x["end"]): A.append(dict(session=sid, lang=slot.get(x["speaker"], langs[0]), t=t, text=w))
    for k, rows in H.items():
        for name, data in ((f"hyp-d{k}.jsonl", rows), (f"asr-d{k}.jsonl", A)):
            tmp = os.path.join(a.out, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fo:
                for x in data: fo.write(json.dumps(x, ensure_ascii=False) + "\n")
            os.replace(tmp, os.path.join(a.out, name))
    rt = sorted(rtt); info = dict(sessions=len(recs), missing=missing, asr_pass=a.asr_pass, counts=dict(st), rtt_median_s=rt[len(rt) // 2] if rt else None, usage=dict(usage),
                                  scored=time.strftime("%F %T"), logs=os.path.abspath(a.logs), sessions_file=os.path.abspath(a.sessions))
    json.dump(info, open(os.path.join(a.out, "qlt-taxi-info.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)
    if missing: print(f"  !! 끝나지 않은 job {len(missing)} — 그 세션은 뺐다(채점은 남은 세션만 — 우리 수치와 견주려면 다 채운다)", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, "experiments/cst_stage4_taxi.py"), "score", "--sessions", a.sessions, "--out", a.out, "--turn-metrics"],
                   check=False, env=dict(os.environ, PYTHONPATH=REPO))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("jobs"); q.add_argument("--sessions", required=True); q.add_argument("--out", required=True)
    q = sub.add_parser("score"); q.add_argument("--sessions", required=True); q.add_argument("--logs", required=True); q.add_argument("--out", required=True)
    q.add_argument("--asr-pass", choices=["en", "de"], default="en", help="받아쓰기를 가져올 실행(목표 언어)")
    a = ap.parse_args(); jobs(a) if a.cmd == "jobs" else score(a)


if __name__ == "__main__":
    main()
