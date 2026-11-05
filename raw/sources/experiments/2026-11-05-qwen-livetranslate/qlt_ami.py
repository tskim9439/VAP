#!/usr/bin/env python
"""Qwen3.8-LiveTranslate 를 우리 AMI 회의 통째 평가와 같은 조건으로 잰다(맥 로컬 전용, 2026-11-05 — 사용자 "Qwen 3.8 live translate 도 동일하게 평가해보고 …
절대 mxc 서버에 올리지말고 로컬에서만 진행해야해").

  jobs : python qlt_ami.py jobs --data <ami-bench-v1> --out <jobs.jsonl> [--subset eval] [--target de]
         → qlt_client.py 입력(회의 × IHM-mix·SDM, 오디오 = 우리 평가가 디코드하는 mono 혼합 캐시 <data>/mono/<코퍼스>/<코퍼스>_<회의>.npy)
  score: python qlt_ami.py score --data <ami-bench-v1> --tok <우리 체크포인트(토크나이저만)> --logs <qlt_client 로그> --out <결과> [--subset eval] [--deltas 10]

같은 것: 회의(ami-bench-v1 공식 eval — 우리 판단용 -subeval 과 같은 16 회의), 오디오, 참조(cst_stage3_eval.window_ref δ — 전사 없는 구간 마스크까지),
  채점(experiments/baseline_ami_ext.py 의 외부 기준 경로 그대로 — score_ext 단어마다 cp·agn, meeteval_cpwer.py 화자 통째 표준 cpWER).
  <out>(우리 정규화)·<out>-wn(Whisper 정규화, 단어마다 — agn 표준)에 cst_stage3_eval 과 같은 꼴의 predictions-rank0.jsonl·summary.json 을 쓰고,
  <out>-wn/meeteval-whisper.{txt,json}(외부 AMI 기준 줄 포함)을 쓴다.
Qwen 쪽 변환(qlt_events): 원문 item(speech_started 의 speaker_id · audio_start_ms ~ speech_stopped 의 audio_end_ms) → 화자 "s<연결>_<번호>",
  단어 시각은 item 구간 안에 고르게(근사 — VibeVoice·Parakeet 외부 기준과 같은 처리, 마스크에만 쓴다).
LLM DER 자리(llm_der): item 구간 × 화자로 만든 80 ms 프레임 활동 대 참조 slot_activity(episode), 전사 없는 구간 프레임은 뺀다(우리 LLM DER 와 같은 마스크·셈).
  혼동은 최적 화자 짝(헝가리안 — der_counts 의 순열 최대와 같은 값, 화자가 많아도 된다).
지연: 원문 완료를 받은 시각 − item 끝(오디오 s), 단어 지연 어림(받은 시각 − 고르게 놓은 단어 시각), WebSocket 왕복(RTT) — 벽시계라 네트워크가 섞인다.
결과 파일에는 AMI 글자가 들어간다(AMI 는 CC BY 4.0). 읽기만 하고 아무것도 지우지 않는다(결과 파일은 다시 쓸 때 새로 만든다)."""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "experiments")]
CHUNK_S = 0.08


def safe(i):
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", i)


def jobs(a):
    man = json.load(open(os.path.join(a.data, "manifest.json"))); n = 0
    with open(a.out, "w") as fo:
        for c in a.corpora.split(","):
            for m in man[a.subset]:
                p = os.path.join(a.data, "mono", c, f"{c}_{m}.npy"); assert os.path.exists(p), f"오디오 없음: {p}"
                fo.write(json.dumps(dict(id=f"{c}:{m}", audio=os.path.abspath(p), target=a.target)) + "\n"); n += 1
    print(f"{n} job → {a.out}")


def der_counts_any(pred, ref, mask):
    """der_counts(vapasr.hf.diar)와 같은 셈 — 혼동의 최적 짝만 헝가리안으로(가설 화자 수가 많아도 된다). pred (K, Nh)·ref (K, Nr) 0/1."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    p, r = pred[mask], ref[mask]; n_ref, n_hyp = r.sum(1), p.sum(1)
    miss = np.clip(n_ref - n_hyp, 0, None).sum(); fa = np.clip(n_hyp - n_ref, 0, None).sum(); both = np.minimum(n_ref, n_hyp).sum()
    ov = r.T @ p; ri, hi = linear_sum_assignment(-ov); best = ov[ri, hi].sum()
    return dict(ref=float(n_ref.sum()), miss=float(miss), fa=float(fa), conf_best=float(both - best), conf_arrival=float(both - best))


def score(a):
    import numpy as np
    from transformers import AutoTokenizer
    from cst_stage3_eval import windows_of, window_ref, summarize
    from baseline_ami_ext import hyp_by_speaker, score_ext, NO_EOT
    from vapasr.data.cst_eval import set_en_norm, pool
    from vapasr.data.cst_dialogue import slot_activity
    from qlt_events import parse, timed_words
    tok = AutoTokenizer.from_pretrained(a.tok); EOT = tok.convert_tokens_to_ids("<EOT>")
    ns = argparse.Namespace(data=a.data, split=None, corpora=a.corpora, deltas=a.deltas, limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0, max_speakers=4,
                            window_draws=0, declare=4, xl_order="lang", subset=a.subset)
    sets, items = windows_of(ns, tok)
    man = json.load(open(os.path.join(a.data, "manifest.json"))); SUB = {**{m: "dev" for m in man["dev"]}, **{m: "eval" for m in man["eval"]}}
    parsed, lat = {}, {}
    for c, i in items:
        cid = sets[c].items[i][0]; lp = os.path.join(a.logs, safe(cid) + ".events.jsonl"); dp = os.path.join(a.logs, safe(cid) + ".done.json")
        if not os.path.exists(dp): print(f"  !! {cid}: 끝난 로그 없음 — 뺀다", flush=True); continue
        P = parse(lp); parsed[cid] = P
        its = [x for x in P["items"] if x["transcript"].strip()]
        lat[cid] = dict(item_done=[x["t_done"] - x["end"] for x in its if "t_done" in x and not x.get("approx")],
                        word=[x["t_done"] - t for x in its if "t_done" in x for _, t in timed_words(x["transcript"], x["start"], x["end"])],
                        rtt=P["meta"]["rtt"], segments=(P["meta"].get("end") or {}).get("segments"), approx=sum(bool(x.get("approx")) for x in P["items"]),
                        usage=(P["meta"].get("end") or {}).get("usage"))
    info = dict(system="qwen3.8-livetranslate", **{k: v for k, v in (next(iter(parsed.values()))["meta"].get("start") or {}).items() if k in ("model", "url", "speed", "chunk_ms", "target")}) if parsed else {}
    for norm, out in (("ours", a.out), ("whisper", a.out + "-wn")):
        set_en_norm(norm); os.makedirs(out, exist_ok=True); tmp = os.path.join(out, "predictions-rank0.jsonl.tmp"); n = 0
        json.dump(dict(info, data=os.path.abspath(a.data), tok=os.path.abspath(a.tok), logs=os.path.abspath(a.logs), corpora=a.corpora, deltas=a.deltas, subset=a.subset,
                       en_norm=norm, world_size=1, scored=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(out, "config-rank0.json"), "w"), indent=1)
        with open(tmp, "w", encoding="utf-8") as fo:
            for c, i in items:
                ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]
                if cid not in parsed: continue
                P = parsed[cid]; K = int(round(L / CHUNK_S))
                hw = [(x["speaker"], w, t) for x in P["items"] for w, t in timed_words(x["transcript"], x["start"], x["end"])]
                spk = list(dict.fromkeys(x["speaker"] for x in P["items"] if x["end"] > x["start"]))
                pa = np.zeros((K, max(1, len(spk))), np.float32)
                for x in P["items"]:
                    if x["end"] > x["start"]: pa[max(0, int(x["start"] / CHUNK_S)): min(K, int(math.ceil(x["end"] / CHUNK_S))), spk.index(x["speaker"])] = 1
                fm = np.ones(K, bool)
                for s_, e_ in ds._bad_spans(d):
                    if e_ > 0 and s_ < L: fm[max(0, int(s_ / CHUNK_S)): min(K, int(np.ceil(e_ / CHUNK_S)))] = False
                for delta in a.deltas:
                    ref_w, _, mask, sp_order = window_ref(ds, i, delta, EOT, tok)
                    keep = [x for x in hw if int(x[2] / CHUNK_S) + delta not in mask]; hyp = hyp_by_speaker(keep); sc = score_ext(ref_w, hyp, d.lang)
                    segs = []                                                                           # item 단위(마스크 뒤) — qlt_tc.py 의 tcpWER·tcORC-WER 가설 구간
                    for x in P["items"]:
                        kw = [w for w, t in timed_words(x["transcript"], x["start"], x["end"]) if int(t / CHUNK_S) + delta not in mask]
                        if kw: segs.append([x["speaker"], round(x["start"], 3), round(max(x["end"], x["start"] + 0.01), 3), " ".join(kw)])
                    ra = np.asarray(slot_activity(ds.eps[cid], sp_order, t0w, L, K, n_slots=len(sp_order)), np.float32).reshape(K, len(sp_order))
                    row = dict(id=f"{cid}@{t0w}", corpus=c, lang=d.lang, delta=delta, L=round(L, 3), K=K, speakers=sp_order, n_spk=len(sp_order),
                               masked_chunks=sum(0 <= k < K for k in mask), forced=0, flush_rounds=0, emitted=[],
                               ext=dict(info, n_hyp_spk=len(hyp), dropped_masked=len(hw) - len(keep), segments=lat[cid]["segments"], approx_items=lat[cid]["approx"],
                                        usage=lat[cid]["usage"]),
                               hyp_words=[[s, w, round(t, 3)] for s, w, t in hw], hyp_segs=segs, ref_text=[" ".join(w for w, _ in r) for r in ref_w],
                               hyp_text=[" ".join(w for w, _ in h) for h in hyp], **sc, eot=dict(NO_EOT), llm_der=der_counts_any(pa, ra, fm))
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
        os.replace(tmp, os.path.join(out, "predictions-rank0.jsonl")); summarize(argparse.Namespace(out=out))
        print(f"  {norm}: {n} 행 → {out}", flush=True)
    # meeteval 표준 cpWER(화자 통째, Whisper) — 외부 AMI 기준 줄까지(meeteval_cpwer.py 그대로)
    import subprocess
    with open(os.path.join(a.out + "-wn", "meeteval-whisper.txt"), "w") as fo:
        subprocess.run([sys.executable, os.path.join(REPO, "experiments/meeteval_cpwer.py"), "--pred", a.out, "--norm", "whisper", "--manifest", os.path.join(a.data, "manifest.json"),
                        "--json", os.path.join(a.out + "-wn", "meeteval-whisper.json")], stdout=fo, stderr=subprocess.STDOUT, check=False,
                       env=dict(os.environ, PYTHONPATH=REPO))
    mt = json.load(open(os.path.join(a.out + "-wn", "meeteval-whisper.json")))
    rows = {k: [json.loads(l) for l in open(os.path.join(o, "predictions-rank0.jsonl"), encoding="utf-8")] for k, o in (("ours", a.out), ("wn", a.out + "-wn"))}
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
    lines = [f"Qwen3.8-LiveTranslate AMI 회의 통째 — {a.subset}, δ{a.deltas}(참조 마스크) · 표준 = Whisper + meeteval(우리 표와 같은 틀) · {time.strftime('%F %T')}"]
    for c in a.corpora.split(","):
        for dl in a.deltas:
            r1 = [r for r in rows["ours"] if r["corpus"] == c and r["delta"] == dl]; r2 = [r for r in rows["wn"] if r["corpus"] == c and r["delta"] == dl]
            if not r1: continue
            p1, p2 = pool(r1), pool(r2); m = mt.get(f"{os.path.basename(a.out.rstrip('/'))}/{c}/{a.subset}/d{dl}", {})
            dz = [r["llm_der"] for r in r1]; n_ = max(1.0, sum(x["ref"] for x in dz)); mi, fa, cf = (100 * sum(x[k] for x in dz) / n_ for k in ("miss", "fa", "conf_best"))
            cids = [r["id"].split("@")[0] for r in r1]; L1 = [v for cid in cids for v in lat[cid]["item_done"]]; L2 = [v for cid in cids for v in lat[cid]["word"]]
            RT = [v for cid in cids for v in lat[cid]["rtt"]]
            lines.append(f"  {c:11s} 회의 {len(r1)} · cpWER 표준 {m.get('cpwer', float('nan')):6.2f} · agn(Whisper) {p2['err_agn']:6.2f} · cp(우리) {p1['err_cp']:6.2f} · agn(우리) {p1['err_agn']:6.2f}"
                         f" · DER {mi + fa + cf:6.2f}(놓침 {mi:.2f} 헛 {fa:.2f} 혼동 {cf:.2f}) · 가설 화자 중앙 {q([r['ext']['n_hyp_spk'] for r in r1], .5)}"
                         f" · 원문 완료 지연 중앙 {q(L1, .5):.2f} s(p90 {q(L1, .9):.2f}) · 단어 지연 평균 {sum(L2) / max(1, len(L2)):.2f} s · RTT 중앙 {q(RT, .5) if RT else float('nan'):.3f} s"
                         f" · 끊김 연결 {sum((lat[cid]['segments'] or 1) - 1 for cid in cids)}")
    rep = "\n".join(lines); print(rep, flush=True); open(os.path.join(a.out, "qlt-ami-report.txt"), "w").write(rep + "\n")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("jobs"); q.add_argument("--data", required=True); q.add_argument("--out", required=True); q.add_argument("--subset", choices=["eval", "dev"], default="eval")
    q.add_argument("--corpora", default="ami-ihmmix,ami-sdm"); q.add_argument("--target", default="de", help="번역 목표 언어(AMI 는 받아쓰기·화자만 채점)")
    q = sub.add_parser("score"); q.add_argument("--data", required=True); q.add_argument("--tok", required=True); q.add_argument("--logs", required=True); q.add_argument("--out", required=True)
    q.add_argument("--subset", choices=["eval", "dev"], default="eval"); q.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10])
    a = ap.parse_args(); jobs(a) if a.cmd == "jobs" else score(a)


if __name__ == "__main__":
    main()
