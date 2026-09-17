#!/usr/bin/env python
"""Phase 2 D1 lane 디코드 평가 — 창 단위 free-running 디코드(lane_state.decode_p2) → LaneParser → 참조(창 시퀀스)와 대조. 정본 §10 의 1차 지표.
  python experiments/p2_eval_lanes.py --model /soundai/Model/VAPASR/p2-D1/final --data <phase2 dir> --sets aihub71631:seen:8,ami:seen:6,nikl2020:heldout:10,chime6:heldout:6 --out <dir> --gpu 0
--sets: <corpus>:<tag>:<창 수>. 대화 파일은 <data>/<corpus>.refined.dialogues.jsonl → 없으면 <data>/<corpus>.dialogues.jsonl (정렬 토큰이 없으면 proxy 토큰: 단어 시각이 있으면 단어 끝, 없으면 발화 구간 균등 배치).
지표(창별·코퍼스별 집계): lane 별 텍스트 CER/WER(참조 lane 과 같은 lane 의 가설을 대조)·화자 무관 pooled CER/WER, ONSET/EOT 시각 매칭(±tol) 정밀도/재현율, 가설 구간의 lane 정확도(겹치는 참조 episode 의 lane 과 일치),
활동 헤드 청크 정확도·F1, 토큰 지연(일치 토큰의 방출 청크 끝 − 참조 시각), 디코드 이상(강제 NEXT·stray EOT·미닫힘). 산출: <out>/report.json, <out>/windows/<corpus>-<i>.json/.wav (뷰어용)."""
import os, sys, json, time, random, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--sets", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--delay", type=int, default=4); ap.add_argument("--R", type=int, default=6); ap.add_argument("--window", type=float, nargs=2, default=(20.0, 40.0)); ap.add_argument("--hop", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tol", type=float, default=0.4, help="ONSET/EOT 매칭 허용 오차(초)"); ap.add_argument("--mono-cache", default=None); ap.add_argument("--gpu", default=None)
ap.add_argument("--min-speakers", type=int, default=2); ap.add_argument("--max-tries", type=int, default=40); ap.add_argument("--runaway-cap", type=int, default=None)
a = ap.parse_args()
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, soundfile as sf
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.hf.lane_state import LaneParser, decode_p2, cer, wer, token_latency, match_events, edit_distance
from vapasr.data.dialogue import Dialogue, CHUNK_S
from vapasr.data.dialogue_dataset import DialogueWindowDataset
from vapasr.data.dialogue_interleave import serialize, lane_activity
from vapasr.data.dialogue_mix import crop_audio
from vapasr.data.textnorm import score_en, score_ko
def log(*s): print(*s, flush=True)

t0 = time.time(); model = VapAsrForStreamingASR.from_pretrained(a.model); tok = load_tokenizer(a.model); model.cuda().eval()
reg = dict(model.config.phase2_registry); assert reg and model.config.lanes > 0, "Phase 2 모델이 아님"; parser = LaneParser(reg, R=model.config.lanes)
log(f"모델 {a.model} 준비 {time.time()-t0:.0f}s · lanes {model.config.lanes} · registry {reg}")
NAMES = {v: k for k, v in reg.items()}; NAMES[model.next_audio] = "<NEXT_AUDIO>"; NAMES[model.empty_audio] = "<EMPTY_AUDIO>"
os.makedirs(os.path.join(a.out, "windows"), exist_ok=True)

def proxy_tokens(d: Dialogue) -> int:
    n = 0
    for u in d.utterances:
        if not u.text or u.tokens: continue
        enc = tok(" " + u.text, add_special_tokens=False)["input_ids"]
        if not enc: continue
        if u.word_timing:
            ends = [w[2] for w in u.word_timing if w[2] >= u.start - 1e-3]
            u.tokens = [(t, min(u.end, ends[min(len(ends) - 1, int(i * len(ends) / len(enc)))])) for i, t in enumerate(enc)] if ends else [(t, u.start + (u.end - u.start) * (i + 1) / len(enc)) for i, t in enumerate(enc)]
        else: u.tokens = [(t, u.start + (u.end - u.start) * (i + 1) / len(enc)) for i, t in enumerate(enc)]
        n += 1
    return n

def clean(x): return x.replace("�", "▯")
def norm(text, lang): return score_ko(text, True) if lang == "Korean" else score_en(text)
def words_of(tokens):
    """[(tid, pos)] → 단어 묶음 [(text, pos)] — 다음 조각이 공백으로 시작하고 누적 디코딩이 깨지지 않으면 경계."""
    out, acc, last = [], [], None
    for i, (t, pos) in enumerate(tokens):
        acc.append(t); last = pos; nxt = tok.decode([tokens[i + 1][0]]) if i + 1 < len(tokens) else " "; txt = tok.decode(acc)
        if nxt.startswith(" ") and "�" not in txt: out.append(dict(text=clean(txt).strip(), pos=round(float(pos), 3), n=len(acc))); acc = []
    if acc: out.append(dict(text=clean(tok.decode(acc)).strip(), pos=round(float(last), 3), n=len(acc)))
    return [w for w in out if w["text"]]

def load_set(corpus):
    refined = os.path.join(a.data, f"{corpus}.refined.dialogues.jsonl"); plain = os.path.join(a.data, f"{corpus}.dialogues.jsonl")
    path = refined if os.path.exists(refined) else plain; assert os.path.exists(path), f"{path} 없음"
    dl = [Dialogue.from_json(l) for l in open(path, encoding="utf-8") if l.strip()]
    n_tok = sum(1 for d in dl for u in d.utterances if u.tokens); prox = 0
    if n_tok == 0:
        for d in dl: prox += proxy_tokens(d)
        tmp = os.path.join(a.out, f"_{corpus}.proxy.dialogues.jsonl")
        with open(tmp, "w", encoding="utf-8") as f:
            for d in dl: f.write(d.to_json() + "\n")
        path = tmp
    ds = DialogueWindowDataset([path], tok, R=a.R, window_s=tuple(a.window), hop_s=a.hop, delays=(a.delay,), seed=a.seed, mono_cache_dir=a.mono_cache)
    return ds, dict(path=os.path.basename(path), dialogues=len(dl), tokens_from=("align" if n_tok else "proxy"), proxied_utts=prox, windows=len(ds), stats=ds.stats)

def eval_window(ds, i, corpus, tag, idx_out):
    s = ds.sequence(i, a.delay); d = ds.dlgs[s["cid"]]; K = s["K"]; L = s["L"]; lang = d.lang
    eps, info = ds.window_episodes(d, s["t0"], L); eps = [e for e in eps if e.lane]
    if len({e.speaker for e in eps}) < a.min_speakers or s.get("masked_chunks"): return None
    wav = np.asarray(crop_audio(ds.mono(s["cid"]), s["t0"], s["t0"] + L), dtype=np.float32)
    t = time.time(); out = decode_p2(model, tok, wav, lang=lang, delay=a.delay, runaway_cap=a.runaway_cap); dt = time.time() - t
    segs, pst = parser.parse(out["emits"])
    # ── 참조: lane 별 텍스트·ONSET/EOT 청크(직렬화 규약과 동일)·활동
    chunks, _ = serialize(eps, (K - 0.5) * CHUNK_S, ds.sp, delay_text=a.delay, delay_onset=0)
    ref_on, ref_eot = {}, {}
    for k, em in chunks:
        for e in em:
            if e.kind == "onset": ref_on[e.ep] = k
            if e.kind == "eot": ref_eot[e.ep] = k
    ref_lane_tokens = collections.defaultdict(list)
    for e in sorted(eps, key=lambda e: e.start): ref_lane_tokens[e.lane] += [(t_, tt) for t_, tt in e.tokens]
    hyp_lane_tokens = collections.defaultdict(list)
    for sg in segs: hyp_lane_tokens[sg.lane] += sg.tokens
    lanes = sorted(set(ref_lane_tokens) | set(hyp_lane_tokens))
    def text_of(tokens): return norm(tok.decode([t_ for t_, _ in tokens], skip_special_tokens=True), lang)
    err_fn = cer if lang == "Korean" else wer; unit = "cer" if lang == "Korean" else "wer"
    per_lane = {}; ed_sum = 0; len_sum = 0
    for ln in lanes:
        r = text_of(ref_lane_tokens.get(ln, [])); h = text_of(hyp_lane_tokens.get(ln, []))
        ru = list(r.replace(" ", "")) if unit == "cer" else r.split(); hu = list(h.replace(" ", "")) if unit == "cer" else h.split()
        ed = edit_distance(hu, ru); ed_sum += ed; len_sum += len(ru); per_lane[ln] = dict(ref=r, hyp=h, ed=ed, n_ref=len(ru), rate=(ed / len(ru) if ru else None))
    ref_all = text_of(sorted([(t_, tt) for e in eps for t_, tt in e.tokens], key=lambda x: x[1])); hyp_all = text_of(sorted([(t_, k) for sg in segs for t_, k in sg.tokens], key=lambda x: x[1]))
    pooled = err_fn(hyp_all, ref_all)
    # ── 이벤트(청크 단위 → 초), lane 정확도, 활동, 지연
    tol = a.tol; on_h = [(sg.k_on + 1) * CHUNK_S for sg in segs if sg.k_on is not None and not sg.implicit]; on_r = [(k + 1) * CHUNK_S for k in ref_on.values()]
    eot_h = [(sg.k_eot + 1) * CHUNK_S for sg in segs if sg.k_eot is not None]; eot_r = [(k + 1) * CHUNK_S for k in ref_eot.values() if k < K]
    on_m = match_events(on_h, on_r, tol); eot_m = match_events(eot_h, eot_r, tol)
    lane_ok = lane_bad = spurious = 0
    for sg in segs:
        st_, en_ = sg.start if sg.start is not None else (sg.tokens[0][1] + 1) * CHUNK_S if sg.tokens else 0.0, sg.end if sg.end is not None else L
        best, bo = None, 0.0
        for e in eps:
            ov = min(en_, e.end) - max(st_, e.start)
            if ov > bo: best, bo = e, ov
        if best is None: spurious += 1
        elif best.lane == sg.lane: lane_ok += 1
        else: lane_bad += 1
    ref_act = np.array(lane_activity(eps, K, a.R), dtype=np.float32); hyp_act = out["act"][:K, : a.R] if out["act"].size else np.zeros_like(ref_act)
    pred = (hyp_act > 0.5).astype(np.float32); act_acc = float((pred == ref_act).mean()); tp = float(((pred == 1) & (ref_act == 1)).sum()); fp = float(((pred == 1) & (ref_act == 0)).sum()); fn = float(((pred == 0) & (ref_act == 1)).sum())
    act_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    lat, n_lat = token_latency(sorted([(t_, k) for sg in segs for t_, k in sg.tokens], key=lambda x: x[1]), sorted([(t_, tt) for e in eps for t_, tt in e.tokens], key=lambda x: x[1]))
    m = dict(unit=unit, lane_rate=(ed_sum / len_sum if len_sum else None), pooled_rate=pooled, n_ref_units=len_sum,
             onset=dict(hit=on_m[0], hyp=on_m[1], ref=on_m[2]), eot=dict(hit=eot_m[0], hyp=eot_m[1], ref=eot_m[2]), lane=dict(ok=lane_ok, bad=lane_bad, spurious=spurious),
             act_acc=act_acc, act_f1=act_f1, latency=dict(mean=(float(np.mean(lat)) if lat else None), p50=(float(np.median(lat)) if lat else None), n=n_lat, n_ref_tokens=sum(len(e.tokens) for e in eps)),
             decode=dict(forced=out["forced"], rounds=out["rounds"], **pst, n_emits=len(out["emits"]), sec=round(dt, 1), ms_per_chunk=round(float(np.mean(out["ticks_ms"])), 1)))
    # ── 뷰어용 JSON
    name = f"{corpus}-{idx_out}"; sf.write(os.path.join(a.out, "windows", name + ".wav"), wav, 16000)
    lane_names = {v: k for k, v in reg.items()}
    js = dict(name=name, corpus=corpus, tag=tag, conv=s["cid"], lang=lang, t0=s["t0"], L=L, K=K, delay=a.delay, R=a.R, speakers=sorted({e.speaker for e in eps}),
              ref=dict(episodes=[dict(ep=e.ep_id, speaker=e.speaker, lane=e.lane, start=round(e.start, 3), end=round(e.end, 3), outcome=e.outcome, p_end=e.p_end, k_on=ref_on.get(e.ep_id), k_eot=ref_eot.get(e.ep_id),
                                      words=words_of(e.tokens), text=clean(tok.decode([t_ for t_, _ in e.tokens])).strip()) for e in sorted(eps, key=lambda e: e.start)],
                       activity=ref_act.astype(int).tolist()),
              hyp=dict(segments=[dict(lane=sg.lane, k_on=sg.k_on, k_eot=sg.k_eot, start=(round(sg.start, 3) if sg.start is not None else None), end=(round(sg.end, 3) if sg.end is not None else None), implicit=sg.implicit, unclosed=sg.k_eot is None,
                                      words=[dict(text=w["text"], k=int(w["pos"]), n=w["n"]) for w in words_of(sg.tokens)], text=clean(tok.decode([t_ for t_, _ in sg.tokens])).strip()) for sg in segs],
                       activity=np.round(hyp_act, 2).tolist(), stream=[dict(k=k, text=(clean(tok.decode([t_])) if t_ not in NAMES else NAMES[t_]), kind=("lane" if t_ in parser.lane_of else "onset" if t_ == parser.onset else "eot" if t_ == parser.eot else "text"), lane=parser.lane_of.get(t_)) for k, t_ in out["emits"]]),
              per_lane={str(k): v for k, v in per_lane.items()}, metrics=m)
    json.dump(js, open(os.path.join(a.out, "windows", name + ".json"), "w"), ensure_ascii=False)
    return dict(name=name, conv=s["cid"], t0=s["t0"], L=L, speakers=len(js["speakers"]), episodes=len(eps), **{k: v for k, v in m.items()})

def agg(rows):
    def mean(xs): xs = [x for x in xs if x is not None]; return round(float(np.mean(xs)), 4) if xs else None
    ed = sum(r["lane_rate"] * r["n_ref_units"] for r in rows if r["lane_rate"] is not None); n = sum(r["n_ref_units"] for r in rows if r["lane_rate"] is not None)
    on = [sum(r["onset"][k] for r in rows) for k in ("hit", "hyp", "ref")]; eo = [sum(r["eot"][k] for r in rows) for k in ("hit", "hyp", "ref")]; ln = [sum(r["lane"][k] for r in rows) for k in ("ok", "bad", "spurious")]
    return dict(windows=len(rows), unit=rows[0]["unit"] if rows else None, lane_rate=(round(ed / n, 4) if n else None), pooled_rate=mean([r["pooled_rate"] for r in rows]),
                onset_p=(round(on[0] / on[1], 3) if on[1] else None), onset_r=(round(on[0] / on[2], 3) if on[2] else None), eot_p=(round(eo[0] / eo[1], 3) if eo[1] else None), eot_r=(round(eo[0] / eo[2], 3) if eo[2] else None),
                lane_acc=(round(ln[0] / (ln[0] + ln[1]), 3) if ln[0] + ln[1] else None), spurious_segments=ln[2], act_acc=mean([r["act_acc"] for r in rows]), act_f1=mean([r["act_f1"] for r in rows]),
                latency_mean=mean([r["latency"]["mean"] for r in rows]), latency_p50=mean([r["latency"]["p50"] for r in rows]), latency_match=(round(sum(r["latency"]["n"] for r in rows) / max(1, sum(r["latency"]["n_ref_tokens"] for r in rows)), 3)),
                forced=sum(r["decode"]["forced"] for r in rows), stray_eot=sum(r["decode"]["stray_eot"] for r in rows), implicit=sum(r["decode"]["implicit"] for r in rows), unclosed=sum(r["decode"]["unclosed"] for r in rows), ms_per_chunk=mean([r["decode"]["ms_per_chunk"] for r in rows]))

report = dict(model=a.model, delay=a.delay, tol=a.tol, sets={}, args=vars(a))
for spec in a.sets.split(","):
    corpus, tag, n = spec.split(":"); n = int(n); t = time.time()
    ds, meta = load_set(corpus); log(f"[{corpus}] {meta}")
    rng = random.Random(a.seed); order = list(range(len(ds))); rng.shuffle(order); rows = []; tries = 0
    for i in order:
        if len(rows) >= n or tries >= a.max_tries * n: break
        tries += 1
        try: r = eval_window(ds, i, corpus, tag, len(rows))
        except Exception as e: log(f"  ! 창 {i} 실패: {type(e).__name__}: {e}"); continue
        if r is None: continue
        rows.append(r); log(f"  {r['name']} {r['conv']}@{r['t0']} L{r['L']:.0f} spk{r['speakers']} {r['unit']} lane {r['lane_rate']} pooled {r['pooled_rate']} onset {r['onset']} eot {r['eot']} lane {r['lane']} act {r['act_acc']:.3f} lat {r['latency']['mean']} dec {r['decode']}")
    report["sets"][corpus] = dict(tag=tag, meta=meta, summary=agg(rows), windows=rows, sec=round(time.time() - t))
    log(f"[{corpus}] 요약 {json.dumps(report['sets'][corpus]['summary'], ensure_ascii=False)}")
json.dump(report, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1); log("→", os.path.join(a.out, "report.json"))
