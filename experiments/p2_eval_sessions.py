#!/usr/bin/env python
"""Phase 2 평가 v2 — 세션 전체를 고정 격자 창으로 잘라 free-running 디코드하고 네 축(인식·화자 귀속·턴·지연)으로 채점한다([[output-phase2-eval-plan]], [[decision-phase2-eval-plan-v2]]).
  python experiments/p2_eval_sessions.py --model <ckpt> --data <phase2 dir> --sets nikl2020:heldout,turnbench:heldout,chime6:heldout,aihub71631:seen --out <dir> [--max-minutes 60] [--delay-onset 2]
창: 각 세션을 0 초부터 --window 초(기본 30) 격자로 전부(선택 없음). 마지막 조각은 --tail-min 초 이상일 때만. lane 은 창 안에서만 유효하므로 화자 귀속 축은 창 단위 매핑(cp 배정) 으로 채점하고 세션으로 합산한다.
참조: 보정/원 발화(speaker·start·end·text=TN). 정렬 토큰(<data>/align-asr-tn-v1/<corpus>) 이 있으면 지연 축과 ONSET/EOT 참조 청크(학습 직렬화 규약)에 쓰고, 없으면 proxy 토큰. quarantine 발화(flags) 는 텍스트 채점에서 빼고 그 구간을 덮는 가설도 뺀다.
TurnBench: 턴 축(ONSET/EOT/턴 교대/backchannel) 은 meta.consensus_utts(3 트랙 합의) 만 참조로 쓴다.
산출: <out>/report.json(셋→세션→창), <out>/seglst/<corpus>/<conv>.{ref,hyp}.json(절대 시각; meeteval 재채점용), <out>/windows/<name>.json/.wav(뷰어; 셋당 --viewer-windows 개)."""
import os, sys, json, time, argparse, collections, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--sets", required=True, help="<corpus>:<tag>[,...]"); ap.add_argument("--out", required=True)
ap.add_argument("--window", type=float, default=30.0); ap.add_argument("--tail-min", type=float, default=10.0); ap.add_argument("--max-minutes", type=float, default=60.0, help="셋당 평가 오디오 상한(분); 세션 순서대로 채운다"); ap.add_argument("--max-sessions", type=int, default=0)
ap.add_argument("--delay", type=int, default=4); ap.add_argument("--delay-onset", type=int, default=0); ap.add_argument("--R", type=int, default=6); ap.add_argument("--mono-cache", default=None); ap.add_argument("--gpu", default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-constrain", action="store_true"); ap.add_argument("--onset-thr", type=float, default=0.35); ap.add_argument("--act-close-chunks", type=int, default=6); ap.add_argument("--act-thr", type=float, default=0.3); ap.add_argument("--runaway-cap", type=int, default=None)
ap.add_argument("--viewer-windows", type=int, default=6); ap.add_argument("--meeteval", action="store_true", help="창 단위 meeteval cpWER/ORC-WER 대조 열")
ap.add_argument("--split-file", default=None, help="세션 split JSON: 각 코퍼스를 heldout 세션 목록으로 제한(7 코퍼스 벤치마크 세션)")
a = ap.parse_args()
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, soundfile as sf
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.hf.lane_state import LaneParser, decode_p2, token_latency
from vapasr.hf import lane_metrics as M
from vapasr.data.dialogue import Dialogue, CHUNK_S
from vapasr.data.dialogue_dataset import DialogueWindowDataset
from vapasr.data.dialogue_interleave import serialize
from vapasr.data.dialogue_mix import crop_audio
from vapasr.data.textnorm import score_en, score_ko
def log(*s): print(*s, flush=True)

t0_ = time.time(); model = VapAsrForStreamingASR.from_pretrained(a.model); tok = load_tokenizer(a.model); model.cuda().eval()
reg = dict(model.config.phase2_registry); assert reg and model.config.lanes > 0, "Phase 2 모델이 아님"; parser = LaneParser(reg, R=model.config.lanes)
log(f"모델 {a.model} 준비 {time.time()-t0_:.0f}s · lanes {model.config.lanes}")
NAMES = {v: k for k, v in reg.items()}; NAMES[model.next_audio] = "<NEXT_AUDIO>"; NAMES[model.empty_audio] = "<EMPTY_AUDIO>"
for d_ in ("windows", "seglst"): os.makedirs(os.path.join(a.out, d_), exist_ok=True)
def clean(x): return x.replace("�", "▯")
def norm(text, lang): return score_ko(text, True) if lang == "Korean" else score_en(text)

def proxy_tokens(d: Dialogue) -> int:
    n = 0
    for u in d.utterances:
        if not u.text or u.tokens: continue
        enc = tok(" " + u.text, add_special_tokens=False)["input_ids"]
        if enc: u.tokens = [(t, u.start + (u.end - u.start) * (i + 1) / len(enc)) for i, t in enumerate(enc)]; n += 1
    return n

def load_set(corpus):
    refined = os.path.join(a.data, f"{corpus}.refined.dialogues.jsonl"); plain = os.path.join(a.data, f"{corpus}.dialogues.jsonl"); path = refined if os.path.exists(refined) else plain; assert os.path.exists(path), f"{path} 없음"
    align_dir = os.path.join(a.data, "align-asr-tn-v1", corpus); align_dir = align_dir if os.path.isdir(align_dir) else None
    ds = DialogueWindowDataset([path], tok, align_dir=align_dir, R=a.R, delays=(a.delay,), delay_onset=a.delay_onset, seed=a.seed, mono_cache_dir=a.mono_cache, max_items=1)   # 창 목록은 안 쓰고 dlgs·mono·episode 도구만
    if a.split_file:
        held = set(json.load(open(a.split_file))["corpora"].get(corpus, {}).get("heldout", []))
        if held: ds.dlgs = {k: v for k, v in ds.dlgs.items() if k in held}
    tokens_from = "align" if align_dir and any(u.tokens for d in ds.dlgs.values() for u in d.utterances) else "proxy"
    if tokens_from == "proxy":
        for d in ds.dlgs.values(): proxy_tokens(d)
    return ds, dict(path=os.path.basename(path), align=tokens_from, dialogues=len(ds.dlgs))

def words_of(tokens):
    out, acc, last = [], [], None
    for i, (t, pos) in enumerate(tokens):
        acc.append(t); last = pos; nxt = tok.decode([tokens[i + 1][0]]) if i + 1 < len(tokens) else " "; txt = tok.decode(acc)
        if nxt.startswith(" ") and "�" not in txt: out.append(dict(text=clean(txt).strip(), pos=round(float(pos), 3), n=len(acc))); acc = []
    if acc: out.append(dict(text=clean(tok.decode(acc)).strip(), pos=round(float(last), 3), n=len(acc)))
    return [w for w in out if w["text"]]

def eval_window(ds, d: Dialogue, t0: float, L: float, corpus: str, tag: str, keep_viewer: bool, name: str):
    t1 = t0 + L; K = int(round(L / CHUNK_S)); lang = d.lang; unit = "cer" if lang == "Korean" else "wer"
    wav = np.asarray(crop_audio(ds.mono(d.conv_id), t0, t1), dtype=np.float32)
    ts = time.time(); out = decode_p2(model, tok, wav, lang=lang, delay=a.delay, runaway_cap=a.runaway_cap, constrain=not a.no_constrain, act_close_chunks=a.act_close_chunks, act_thr=a.act_thr, onset_thr=a.onset_thr); dt = time.time() - ts
    segs, pst = parser.parse(out["emits"], out.get("act_closed"))
    # ── 참조 발화(창 상대 시각). quarantine(flags) 발화는 텍스트 채점 제외 + 그 구간을 덮는 가설 제외
    ref_all = [u for u in d.utterances if u.end > t0 and u.start < t1]
    bad = [(max(u.start, t0) - t0, min(u.end, t1) - t0) for u in ref_all if u.flags or not u.text]
    ref = [dict(speaker=u.speaker, start=max(u.start, t0) - t0, end=min(u.end, t1) - t0, text=norm(u.text, lang), utt_id=u.utt_id) for u in ref_all if not u.flags and u.text]
    cons = set(d.meta.get("consensus_utts") or []); ref_turn = [r for r in ref if r["utt_id"] in cons] if cons else ref
    hyp = []
    for sg in segs:
        st = sg.start if sg.start is not None else ((sg.tokens[0][1] + 1) * CHUNK_S if sg.tokens else 0.0); en = sg.end if sg.end is not None else L
        if bad and sum(M.overlap(st, en, b0, b1) for b0, b1 in bad) > 0.5 * max(1e-3, en - st): continue
        hyp.append(dict(lane=sg.lane, start=round(st, 3), end=round(en, 3), text=norm(tok.decode([t_ for t_, _ in sg.tokens], skip_special_tokens=True), lang), k_on=sg.k_on, k_eot=sg.k_eot, implicit=sg.implicit, k_act_close=sg.k_act_close,
                        words=[dict(text=w["text"], k=int(w["pos"]), n=w["n"]) for w in words_of(sg.tokens)], raw=clean(tok.decode([t_ for t_, _ in sg.tokens])).strip()))
    # ── A·B
    A = M.utterance_assigned(ref, hyp, unit); P = M.pooled(ref, hyp, unit); C = M.cp_error(ref, hyp, unit); mapping = C["mapping"]
    D = M.lane_der(ref, hyp, mapping, horizon=L); SW = M.lane_switches(ref, hyp)
    # ── C: 참조 ONSET/EOT 청크는 학습 직렬화 규약(episode·δ_onset·EOT 후보 규칙)으로, 합의 발화만
    eps, _ = ds.window_episodes(d, t0, L); eps = [e for e in eps if e.lane]
    if bad: eps = [e for e in eps if sum(M.overlap(e.start, e.end, b0, b1) for b0, b1 in bad) <= 0.5 * max(1e-3, e.end - e.start)]     # 가설과 같은 규칙: quarantine 구간의 참조 이벤트도 제외
    if cons: eps = [e for e in eps if any(uid in cons for uid in e.utt_ids)]
    chunks, _ = serialize(eps, (K - 0.5) * CHUNK_S, ds.sp, delay_text=a.delay, delay_onset=a.delay_onset); ref_on, ref_eot = [], []
    for k, em in chunks:
        for e in em:
            if e.kind == "onset": ref_on.append((k + 1) * CHUNK_S)
            if e.kind == "eot" and k < K: ref_eot.append((k + 1) * CHUNK_S)
    hyp_on = [(sg.k_on + 1) * CHUNK_S for sg in segs if sg.k_on is not None and not sg.implicit]; hyp_eot = [(sg.k_eot + 1) * CHUNK_S for sg in segs if sg.k_eot is not None]; hyp_eot_lanes = [sg.lane for sg in segs if sg.k_eot is not None]
    EV = dict(onset=M.event_metrics(hyp_on, ref_on), eot=M.event_metrics(hyp_eot, ref_eot)); TC = M.turn_change_metrics(ref_turn, hyp); EIT = M.eot_in_turn(ref_turn, hyp_eot, mapping, hyp_eot_lanes)
    spk = sorted({r["speaker"] for r in ref}); col = {s: i for i, s in enumerate(spk)}; ref_act = np.zeros((K, max(1, len(spk))), dtype=np.float32)
    for r in ref:
        for k in range(max(0, int(r["start"] / CHUNK_S)), min(K, int(math.ceil(r["end"] / CHUNK_S)))): ref_act[k, col[r["speaker"]]] = 1
    hyp_act = out["act"][:K, : a.R] if out["act"].size else np.zeros((K, a.R), dtype=np.float32)
    AF = M.activity_f1(ref_act, hyp_act, {l: col[s] for l, s in mapping.items() if s in col})
    lat, n_lat = token_latency(sorted([(t_, k) for sg in segs for t_, k in sg.tokens], key=lambda x: x[1]), sorted([(t_, tt - t0) for u in ref_all if not u.flags for t_, tt in (u.tokens or []) if t0 <= tt <= t1], key=lambda x: x[1]))
    n_ref_tok = sum(len([1 for _, tt in (u.tokens or []) if t0 <= tt <= t1]) for u in ref_all if not u.flags)
    m = dict(unit=unit, n_ref=A["n_ref"], A=dict(ua=A, pooled=P), B=dict(cp=dict(rate=C["rate"], errors=C["errors"], mapping={str(k): v for k, v in mapping.items()}, unmatched_lanes=C["unmatched_lanes"], unmatched_speakers=C["unmatched_speakers"]), der=D, switches=SW["switches"], n_segments=SW["n_segments"]),
             C=dict(events=EV, turn=TC, eot_in_turn=EIT, activity=AF, n_ref_turn_utts=len(ref_turn)), D=dict(latency_mean=(float(np.mean(lat)) if lat else None), latency_p50=(float(np.median(lat)) if lat else None), matched=n_lat, n_ref_tokens=n_ref_tok),
             decode=dict(forced=out["forced"], rounds=out["rounds"], **pst, n_emits=len(out["emits"]), sec=round(dt, 1), ms_per_chunk=round(float(np.mean(out["ticks_ms"])), 1)), n_hyp=len(hyp), n_bad_ref=len(bad))
    if a.meeteval:
        try:
            from meeteval.wer import cpwer, orcwer; from meeteval.io.seglst import SegLST
            R_ = SegLST([dict(session_id=name, speaker=r["speaker"], start_time=r["start"], end_time=r["end"], words=r["text"]) for r in ref]); H_ = SegLST([dict(session_id=name, speaker=f"l{h['lane']}", start_time=h["start"], end_time=h["end"], words=h["text"]) for h in hyp])
            c_ = cpwer(R_, H_)[name]; o_ = orcwer(R_, H_)[name]; m["meeteval"] = dict(cpwer=dict(rate=c_.error_rate, errors=c_.errors, length=c_.length), orcwer=dict(rate=o_.error_rate, errors=o_.errors, length=o_.length))
        except Exception as e: m["meeteval"] = dict(error=f"{type(e).__name__}: {e}")
    seg_ref = [dict(session_id=d.conv_id, speaker=r["speaker"], start_time=round(r["start"] + t0, 3), end_time=round(r["end"] + t0, 3), words=r["text"]) for r in ref]
    seg_hyp = [dict(session_id=d.conv_id, speaker=f"w{int(t0)}_l{h['lane']}", start_time=round(h["start"] + t0, 3), end_time=round(h["end"] + t0, 3), words=h["text"]) for h in hyp]
    if keep_viewer:
        sf.write(os.path.join(a.out, "windows", name + ".wav"), wav, 16000)
        js = dict(name=name, corpus=corpus, tag=tag, conv=d.conv_id, lang=lang, t0=t0, L=L, K=K, delay=a.delay, delay_onset=a.delay_onset, R=a.R, speakers=spk,
                  ref=dict(episodes=[dict(ep=e.ep_id, speaker=e.speaker, lane=e.lane, start=round(e.start, 3), end=round(e.end, 3), outcome=e.outcome, p_end=e.p_end, words=words_of(e.tokens), text=clean(tok.decode([t_ for t_, _ in e.tokens])).strip()) for e in sorted(eps, key=lambda e: e.start)],
                           utts=ref, activity=ref_act.astype(int).tolist(), speakers=spk, consensus=bool(cons)),
                  hyp=dict(constrain=not a.no_constrain, act_closed=out.get("act_closed", []), segments=hyp, activity=np.round(hyp_act, 2).tolist(),
                           stream=[dict(k=k, text=(clean(tok.decode([t_])) if t_ not in NAMES else NAMES[t_]), kind=("lane" if t_ in parser.lane_of else "onset" if t_ == parser.onset else "eot" if t_ == parser.eot else "text"), lane=parser.lane_of.get(t_), p=p_) for (k, t_), p_ in zip(out["emits"], out.get("probs", [None] * len(out["emits"])))]),
                  metrics=m)
        json.dump(js, open(os.path.join(a.out, "windows", name + ".json"), "w"), ensure_ascii=False)
    return m, seg_ref, seg_hyp

def agg_windows(rows):
    """창 행 → 합산 지표(오류 합 / 참조 길이 합 등)."""
    def s(path):
        tot = 0
        for r in rows:
            v = r
            for k in path: v = v.get(k) if isinstance(v, dict) else None
            tot += (v or 0)
        return tot
    n = s(("n_ref",)); out = dict(windows=len(rows), n_ref=n, unit=rows[0]["unit"] if rows else None)
    out["A_ua"] = (s(("A", "ua", "S")) + s(("A", "ua", "D")) + s(("A", "ua", "I"))) / n if n else None; out["A_ua_SDI"] = [s(("A", "ua", "S")) / n if n else None, s(("A", "ua", "D")) / n if n else None, s(("A", "ua", "I")) / n if n else None]
    out["A_pooled"] = (s(("A", "pooled", "S")) + s(("A", "pooled", "D")) + s(("A", "pooled", "I"))) / n if n else None
    out["B_cp"] = s(("B", "cp", "errors")) / n if n else None; out["B_attr_cost"] = (out["B_cp"] - out["A_ua"]) if out["B_cp"] is not None and out["A_ua"] is not None else None
    rt = sum((r["B"]["der"]["ref_time"] or 0) for r in rows); out["B_der"] = (sum((r["B"]["der"][k] or 0) * (r["B"]["der"]["ref_time"] or 0) for r in rows for k in ("der",)) / rt) if rt else None
    out["B_der_parts"] = {k: (sum((r["B"]["der"][k] or 0) * (r["B"]["der"]["ref_time"] or 0) for r in rows) / rt if rt else None) for k in ("miss", "fa", "conf")}
    ns = s(("B", "n_segments")); out["B_switch_rate"] = (s(("B", "switches")) / ns) if ns else None
    for ev in ("onset", "eot"):
        for tol in ("0.2", "0.4", "0.8"):
            h = sum(r["C"]["events"][ev][tol]["hit"] for r in rows); nh = sum(r["C"]["events"][ev][tol]["hyp"] for r in rows); nr = sum(r["C"]["events"][ev][tol]["ref"] for r in rows)
            out[f"C_{ev}_{tol}"] = dict(p=(h / nh if nh else None), r=(h / nr if nr else None), hit=h, hyp=nh, ref=nr)
    th = sum(r["C"]["turn"]["hit"] for r in rows); tn = sum(r["C"]["turn"]["hyp"] for r in rows); tr = sum(r["C"]["turn"]["ref"] for r in rows); p = th / tn if tn else None; rc = th / tr if tr else None
    out["C_turn"] = dict(p=p, r=rc, f1=((2 * p * rc / (p + rc)) if p and rc else None), hit=th, hyp=tn, ref=tr, latency_median=(float(np.median([r["C"]["turn"]["latency_median"] for r in rows if r["C"]["turn"]["latency_median"] is not None])) if any(r["C"]["turn"]["latency_median"] is not None for r in rows) else None))
    sp = sum(r["C"]["eot_in_turn"]["ref_speech_s"] for r in rows); out["C_eot_in_turn_per_min"] = (s(("C", "eot_in_turn", "count")) / (sp / 60)) if sp else None
    tp = s(("C", "activity", "tp")); fp = s(("C", "activity", "fp")); fn = s(("C", "activity", "fn")); out["C_activity_f1"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    lat = [r["D"]["latency_mean"] for r in rows if r["D"]["latency_mean"] is not None]; out["D_latency_mean"] = float(np.mean(lat)) if lat else None; out["D_token_match"] = (s(("D", "matched")) / s(("D", "n_ref_tokens"))) if s(("D", "n_ref_tokens")) else None
    out["decode"] = dict(forced=s(("decode", "forced")), stray_eot=s(("decode", "stray_eot")), implicit=s(("decode", "implicit")), unclosed=s(("decode", "unclosed")), ms_per_chunk=(float(np.mean([r["decode"]["ms_per_chunk"] for r in rows])) if rows else None))
    if rows and "meeteval" in rows[0] and "error" not in rows[0]["meeteval"]:
        le = sum(r["meeteval"]["cpwer"]["length"] for r in rows if "cpwer" in r.get("meeteval", {})); out["meeteval_cpwer"] = (sum(r["meeteval"]["cpwer"]["errors"] for r in rows if "cpwer" in r.get("meeteval", {})) / le) if le else None; out["meeteval_orcwer"] = (sum(r["meeteval"]["orcwer"]["errors"] for r in rows if "orcwer" in r.get("meeteval", {})) / le) if le else None
    return out

report = dict(model=a.model, args=vars(a), sets={})
for spec in a.sets.split(","):
    corpus, tag = spec.split(":"); tset = time.time(); ds, meta = load_set(corpus); log(f"[{corpus}] {meta}")
    sessions = {}; minutes = 0.0; n_view = 0; all_rows = []
    for cid in sorted(ds.dlgs):
        if a.max_sessions and len(sessions) >= a.max_sessions: break
        if minutes >= a.max_minutes: break
        d = ds.dlgs[cid]; rows = []; seg_ref, seg_hyp = [], []; t0 = 0.0
        while t0 + a.tail_min <= d.duration_s and minutes < a.max_minutes:
            L = min(a.window, d.duration_s - t0); name = f"{corpus}-{cid.split(':')[-1].replace('/', '_')}@{int(t0)}"
            try: m, sr, sh = eval_window(ds, d, t0, L, corpus, tag, n_view < a.viewer_windows, name)
            except Exception as e: log(f"  ! {name} 실패: {type(e).__name__}: {e}"); t0 += a.window; continue
            if n_view < a.viewer_windows: n_view += 1
            m["t0"] = t0; m["L"] = L; rows.append(m); seg_ref += sr; seg_hyp += sh; minutes += L / 60; t0 += a.window
        if not rows: continue
        sessions[cid] = dict(duration_s=d.duration_s, summary=agg_windows(rows), windows=rows); all_rows += rows
        sd = os.path.join(a.out, "seglst", corpus); os.makedirs(sd, exist_ok=True); base = os.path.join(sd, cid.split(":")[-1].replace("/", "_"))
        json.dump(seg_ref, open(base + ".ref.json", "w"), ensure_ascii=False); json.dump(seg_hyp, open(base + ".hyp.json", "w"), ensure_ascii=False)
        s_ = sessions[cid]["summary"]; log(f"  {cid} {d.duration_s/60:.1f} min · 창 {len(rows)} · A_ua {s_['A_ua']} pooled {s_['A_pooled']} · cp {s_['B_cp']} DER {s_['B_der']} switch {s_['B_switch_rate']} · onset0.4 {s_['C_onset_0.4']['p']}/{s_['C_onset_0.4']['r']} eot0.4 {s_['C_eot_0.4']['p']}/{s_['C_eot_0.4']['r']} turnF1 {s_['C_turn']['f1']} eot_in_turn/min {s_['C_eot_in_turn_per_min']} · lat {s_['D_latency_mean']}")
    summ = agg_windows(all_rows) if all_rows else {}
    if sessions:
        vals = [s["summary"]["A_ua"] for s in sessions.values()]; w = [s["summary"]["n_ref"] for s in sessions.values()]; summ["A_ua_ci95"] = M.bootstrap_ci(vals, w)
        vals = [s["summary"]["B_cp"] for s in sessions.values()]; summ["B_cp_ci95"] = M.bootstrap_ci(vals, w)
    report["sets"][corpus] = dict(tag=tag, meta=meta, minutes=round(minutes, 1), n_sessions=len(sessions), summary=summ, sessions=sessions, sec=round(time.time() - tset))
    log(f"[{corpus}] 요약 {json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict) or k in ('C_turn',)}, ensure_ascii=False)}")
json.dump(report, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1); log("→", os.path.join(a.out, "report.json"))
