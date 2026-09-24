#!/usr/bin/env python3
"""Semantic commit 뷰어 데이터 묶음 — semcommit_eval.py 스트림 jsonl(모델·설정별 방출·trace)에서 스트림을 골라, 오디오(평가와 같은 조립·패딩, MP3 base64)·참조 단어·후보 등급·
모델 설정별 가설 단어 시각/<SEM_END> 위치 분류/턴 종료(평가 행 event_ids 의 턴 토큰: <EOT>, v0.2 <TURN_END>)·p(SEM)/p(턴) trace 를 한 JSON 으로 만든다. 보고서 요약(설정별 지표·PC 대조·라벨 통계·블라인드 점검)도 같이 싣는다.

  python experiments/semcommit_viewer_bundle.py --tokenizer runs/v0.2-r1/final \\
      --set English=words-ls-test.jsonl,labels-ls-test.jsonl --set Korean=words-ks-eval.jsonl,labels-ks-eval.jsonl \\
      --run r1=eval/v0.2-r1/r1-d4 --run r2=eval/v0.2-r2/r2-d4 --run base=eval/v0.2-r1/base-d4 --run r1d2=eval/v0.2-r1/r1-d2 \\
      --pc eval/pc-ls-test.json --label-stats labels/v0.2 --spot spot/judged.json --n English=12 --n Korean=18 --out viewer.json \\
      --html experiments/semcommit_viewer.template.html viewer.html

--run NAME=PREFIX: PREFIX-<set>.streams.jsonl 과 PREFIX-<set>.json(보고서). set 이름 = words 파일의 words- 뒤(예: ls-test). 첫 --run 이 선택 기준(bias=0)이다.
"""
import argparse, base64, collections, glob, hashlib, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vapasr.hf.commit_metrics import (CHUNK_S, EOT_ID, LEGACY_TURN_END_ID, SEM_END_ID, emit_time, map_events, pc_punct_after, ref_chunk)

EVENT_IDS = {SEM_END_ID, EOT_ID, LEGACY_TURN_END_ID}                     # 텍스트가 아닌 이벤트 토큰(새 모델 <EOT>·v0.2 <TURN_END> 모두)

SR = 16000


def tok_fns(tok):
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}) or {}) | EVENT_IDS
    cache = {}
    def piece(t):
        if t not in cache: cache[t] = tok.convert_ids_to_tokens(int(t)) or ""
        return cache[t]
    return (lambda t: t not in special), (lambda t: piece(t)[:1] in ("Ġ", " ", "▁")), (lambda ids: tok.decode(ids, skip_special_tokens=True))


def hyp_word_chunks(emitted, is_text, word_start, decode):
    """commit_metrics.split_hyp 와 같은 단어 경계로 (단어, 마지막 토큰 청크) 목록."""
    words, cur, kl = [], [], None
    for k, t in emitted:
        k, t = int(k), int(t)
        if t in EVENT_IDS or not is_text(t): continue
        if cur and word_start(t): words.append((decode(cur).strip(), kl)); cur = []
        cur.append(t); kl = k
    if cur: words.append((decode(cur).strip(), kl))
    return words


def classify(p, cand, last, ev):
    if ev.get("mid_word"): return "mid_word"
    if p < 0: return "no_word"
    g = cand.get(p)
    if g in ("A", "B", "N"): return g
    return "end_nocand" if p == last else "inside"


def trace_by_chunk(tr, K):
    ps, pt = [0.0] * (K + 1), [0.0] * (K + 1)
    for k, a, b in zip(tr.get("k", []), tr.get("p_sem", []), tr.get("p_turn", [])):
        k = min(int(k), K); ps[k] = max(ps[k], float(a)); pt[k] = max(pt[k], float(b))
    return [round(x * 1000) for x in ps], [round(x * 1000) for x in pt]


def encode_mp3(x):
    import numpy as np
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-codec:a", "libmp3lame", "-b:a", "32k", "-f", "mp3", "-"],
                       input=np.asarray(x, np.float32).tobytes(), capture_output=True, check=True)
    return "data:audio/mpeg;base64," + base64.b64encode(p.stdout).decode()


def report_rows(path):
    d = json.load(open(path)); out = []
    for name, c in d["configs"].items():
        o = c["overall"]; t = o["text"]; w = o["timing"]["late2"]; tu = o.get("turn") or {}; tl = tu.get("late2") or {}; a = o.get("asr") or {}
        asr = {m: round(v["rate"] * 100, 2) for m, v in a.items() if isinstance(v, dict) and "rate" in v}
        out.append(dict(config=name, mode=c.get("mode"), value=c.get("value"), n_hyp=t["n_hyp"], P=w.get("precision"), P_exB=w.get("precision_excl_B"), R=w.get("recall"), F1=w.get("f1"),
                        lat50=(w.get("latency_s") or {}).get("p50"), pcr=t.get("pcr"), B_rate=t.get("B_commit_rate"), N_rate=t.get("hardneg_commit_rate"),
                        B_share=(t["ambiguous"] / t["n_hyp"]) if t.get("n_hyp") else None,
                        turn_P=tl.get("precision"), turn_R=tl.get("recall"), turn_early=tu.get("early_turn"), n_streams=o["streams"], asr=asr,
                        premature=t.get("premature_by_category")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--set", action="append", required=True, help="Lang=words.jsonl,labels.jsonl")
    ap.add_argument("--run", action="append", required=True, help="name=prefix (첫 번째가 선택 기준)"); ap.add_argument("--n", action="append", default=[], help="Lang=N")
    ap.add_argument("--main-config", default="bias=0"); ap.add_argument("--pc", default=None); ap.add_argument("--label-stats", default=None); ap.add_argument("--spot", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--html", nargs=2, metavar=("TEMPLATE", "OUT"), default=None, help="템플릿의 __DATA__ 에 묶음을 넣은 단일 HTML")
    a = ap.parse_args()
    from vapasr.hf import load_tokenizer
    from vapasr.data.streams import assemble_stream
    tok = load_tokenizer(a.tokenizer); is_text, word_start, decode = tok_fns(tok)
    runs = [tuple(r.split("=", 1)) for r in a.run]; nsel = {k: int(v) for k, v in (x.split("=") for x in a.n)}
    spot = json.load(open(a.spot)) if a.spot else {}
    spot_by = collections.defaultdict(list)
    for n, s in spot.items(): spot_by[s["stream"]].append(dict(n=int(n), after_word=s["after_word"], grade=s["grade"], verdicts=s["verdicts"], notes=s["notes"]))
    bundle = dict(runs=[r for r, _ in runs], main_config=a.main_config, chunk_s=CHUNK_S, sets={}, streams=[], summary={})
    for spec in a.set:
        lang, files = spec.split("="); wf, lf = files.split(","); set_name = Path(wf).stem.replace("words-", "")
        W = {r["id"]: r for r in map(json.loads, open(wf))}; L = {r["id"]: r for r in map(json.loads, open(lf))}
        rows = collections.defaultdict(dict)                                                # id → run → config → row
        for rn, pref in runs:
            for line in open(f"{pref}-{set_name}.streams.jsonl"):
                r = json.loads(line); rows[r["id"]].setdefault(rn, {})[r["config"]["name"]] = r
        bundle["sets"][set_name] = dict(lang=lang, reports={rn: report_rows(f"{pref}-{set_name}.json") for rn, pref in runs if os.path.exists(f"{pref}-{set_name}.json")})
        # 스트림별 요약 + 태그(첫 run 의 main config 기준, 둘째 run 도 참고)
        items = []
        for sid, w in W.items():
            if sid not in rows or runs[0][0] not in rows[sid]: continue
            ws = sorted(w["words"], key=lambda x: int(x["i"])); last = len(ws) - 1
            cands = L.get(sid, {}).get("candidates", []); cand = {int(c["after_word"]): c["grade"] for c in cands}
            pa = pc_punct_after(ws, w["pnc_text"]) if w.get("pnc_text") else {}
            models = {}
            for rn, _ in runs:
                models[rn] = {}
                for cfg, r in rows[sid].get(rn, {}).items():
                    K = int(r["K"]); d = int(r["delta"]); hw = hyp_word_chunks(r["emitted"], is_text, word_start, decode)
                    if [x for x, _ in hw] != r["hyp"]["words"]: raise SystemExit(f"가설 단어 재구성 불일치 {sid} {rn} {cfg}")
                    sem_ev = [e for e in r["hyp"]["events"] if int(e["id"]) == SEM_END_ID]
                    pos = map_events([x["text"] for x in ws], r["hyp"]["words"], sem_ev, [ref_chunk(float(x["end_time"]), d, K) for x in ws], K)
                    seen, sem = set(), []
                    for e, p in zip(sem_ev, pos):                                           # 중복 규칙 = text_position_metrics(단어 조각·첫 단어 전은 seen 에 안 넣는다)
                        c = classify(p, cand, last, e)
                        if c in ("A", "B", "N", "end_nocand", "inside"):
                            if p in seen: c = "dup"
                            seen.add(p)
                        sem.append([int(e["k"]), int(e["after"]), int(p), c, pa.get(p, "")])
                    turn_tid = (r.get("event_ids") or [SEM_END_ID, None])[1]
                    turn = [int(k) for k, t in r["emitted"] if turn_tid is not None and int(t) == int(turn_tid)]
                    asr = r["metrics"]["asr"]; err = {m: [v["errors"], v["n_ref"]] for m, v in asr.items() if isinstance(v, dict) and "errors" in v}
                    m = dict(K=K, delta=d, words=[[x, k] for x, k in hw], sem=sem, turn=turn, err=err, text=r["hyp"]["text"])
                    if cfg == a.main_config and r.get("trace"): m["p_sem"], m["p_turn"] = trace_by_chunk(r["trace"], K)
                    models[rn][cfg] = m
            main = models[runs[0][0]].get(a.main_config) or {}
            t_last = float(ws[-1]["end_time"]) if ws else 0.0
            tags = set()
            for rn in [x for x, _ in runs][:2]:
                mm = models.get(rn, {}).get(a.main_config)
                if not mm: continue
                cls = {s[3] for s in mm["sem"]}
                for c, tg in (("A", "A 적중"), ("B", "B 위 commit"), ("N", "N 위 commit"), ("inside", "후보 밖 commit"), ("end_nocand", "후보 없는 발화 끝 commit")):
                    if c in cls: tags.add(tg)
                if any(emit_time(k, mm["K"]) < t_last - 1e-6 for k in mm["turn"]): tags.add("TURN 조기")
                hitA = {s[2] for s in mm["sem"] if s[3] == "A"}
                if any(g == "A" and p not in hitA for p, g in cand.items()): tags.add("A 놓침")
            if any(g == "A" and p != last for p, g in cand.items()): tags.add("중간 A")
            if sid in spot_by: tags.add("블라인드 점검 표본")
            items.append(dict(sid=sid, w=w, ws=ws, cands=cands, pa=pa, models=models, tags=sorted(tags), h=hashlib.md5(sid.encode()).hexdigest()))
        # 태그 덮기 선택: 우선순위 순으로 태그마다 몫만큼, 이미 고른 것 제외, 태그 많은 순 → 해시 순
        order = ["중간 A", "후보 없는 발화 끝 commit", "N 위 commit", "TURN 조기", "B 위 commit", "후보 밖 commit", "A 적중", "A 놓침", "블라인드 점검 표본"]
        N = nsel.get(lang, 12); quota = max(1, N // len(order) + 1); chosen = []
        for tg in order:
            cand_items = sorted([it for it in items if tg in it["tags"] and it not in chosen], key=lambda it: (-len(it["tags"]), it["h"]))
            chosen += cand_items[:quota]
            if len(chosen) >= N: break
        chosen = chosen[:N]
        for it in sorted(chosen, key=lambda it: it["sid"]):
            w, ws = it["w"], it["ws"]; K = max(m["K"] for mm in it["models"].values() for m in mm.values())
            row = dict(w); row["duration_s"] = max(float(w["duration_s"]), K * CHUNK_S)
            audio = encode_mp3(assemble_stream(row))
            bundle["streams"].append(dict(
                id=it["sid"], set=set_name, lang=lang, duration_s=round(float(w["duration_s"]), 3), K=K, tags=it["tags"], audio=audio, ref_text=w.get("text"), pnc_text=w.get("pnc_text"),
                words=[[x["text"], round(float(x["end_time"]), 3), x.get("tags") or [], it["pa"].get(int(x["i"]), "")] for x in ws],
                candidates=[dict(after_word=int(c["after_word"]), grade=c["grade"], why=c.get("why"), B=c.get("B"), C=c.get("C"), fu=c.get("future_unobserved")) for c in it["cands"]],
                models=it["models"], spot=spot_by.get(it["sid"], [])))
        print(f"{set_name}: 후보 스트림 {len(items)} → 선택 {len(chosen)}", {tg: sum(tg in it["tags"] for it in chosen) for tg in order}, flush=True)
    if a.pc: bundle["summary"]["pc"] = {Path(k).name.replace(".streams.jsonl", ""): v for k, v in json.load(open(a.pc)).items()}
    if a.label_stats:
        bundle["summary"]["labels"] = {}
        for f in sorted(glob.glob(os.path.join(a.label_stats, "labels-*.stats.json"))):
            d = json.load(open(f)); lang = next(iter(d["by_lang"])); b = d["by_lang"][lang]
            bundle["summary"]["labels"][Path(f).name[7:-11]] = dict(lang=lang, **{k: b.get(k) for k in ("candidates", "A", "B", "N", "future_unobserved_A")},
                                                                     kappa={k: v["kappa"] for k, v in d.get("agreement", {}).items()}, C_type=d.get("C_type", {}).get(lang))
    if spot:
        cnt = collections.Counter()
        for s in spot.values():
            v = list(s["verdicts"].values()); cons = v[0] if len(set(v)) == 1 else "SPLIT"; cnt[(s["set"], s["grade"], cons)] += 1
        bundle["summary"]["spot"] = [dict(set=k[0], grade=k[1], verdict=k[2], n=n) for k, n in sorted(cnt.items())]
    json.dump(bundle, open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
    if a.html:
        t = open(a.html[0]).read(); assert t.count("__DATA__") == 1, "템플릿에 __DATA__ 가 하나여야 한다"
        open(a.html[1], "w").write(t.replace("__DATA__", json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("\ufffd", "\\ufffd")))   # 불완전 바이트 조각(U+FFFD)은 JSON 이스케이프로
        print("→", a.html[1], f"{os.path.getsize(a.html[1]) / 1e6:.2f} MB")
    print("→", a.out, f"{os.path.getsize(a.out) / 1e6:.2f} MB", len(bundle["streams"]), "streams")


if __name__ == "__main__":
    main()
