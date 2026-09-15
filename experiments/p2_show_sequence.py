#!/usr/bin/env python
"""Phase 2 Q0 — 실제 샘플로 학습 시퀀스를 만들고 검증한다 (task-phase2-data-prep, 사용자 요청 2026-09-15 "각 DB 마다 실제 샘플로 시퀀스를 짜보고 검증").

  python experiments/p2_show_sequence.py --dialogues <dir>/ami.dialogues.jsonl [--align <dir>/align-asr-tn-v1/ami] --tokenizer $MXC_QWEN_ASR_DIR --out <dir>/seq/ami [--windows 3] [--delay 4]
토큰 시각: 정렬 parts 가 있으면 그것, 없으면 --proxy (word_timing 이 있는 NOTSOFAR 는 단어 시각, 그 외는 발화 구간에 균등 배치) — proxy 는 시퀀스 구조 검증용이며 학습용이 아니다.
검증(창마다): (1) 텍스트 라운드트립 — 화자별 lexical 토큰을 이으면 참조 토큰열과 같다 (2) never_free ≡ lazy_free (N≤R) (3) ONSET < 그 episode 의 lexical < EOT 순서·EOT 청크 = 규칙값
      (4) EOT soft 위치·가중치 (5) 활동 타깃 = 참조 구간 (6) 오디오 길이 = K·0.08 s (7) 태그 규칙(ONSET/EOT 앞 태그, lexical 은 변경 시) (8) lane 부족 0
출력: <out>/report.json, <out>/window-<i>.txt (블록 문자열 + 토큰 표), <out>/window-<i>.wav (mono 혼합 창)."""
import os, sys, json, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf
from vapasr.data.dialogue import Dialogue, build_episodes, chunk_of, CHUNK_S
from vapasr.data.lane_alloc import allocate, eot_chunk
from vapasr.data.eot_soft import assign_p_end
from vapasr.data.dialogue_interleave import serialize, flatten, lane_activity, render
from vapasr.data.dialogue_dataset import DialogueWindowDataset, load_align_parts
from vapasr.data.dialogue_tokens import add_phase2_specials, LANE_TOKENS

ap = argparse.ArgumentParser(); ap.add_argument("--dialogues", required=True); ap.add_argument("--align"); ap.add_argument("--tokenizer", default=os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"))
ap.add_argument("--out", required=True); ap.add_argument("--windows", type=int, default=3); ap.add_argument("--delay", type=int, default=4); ap.add_argument("--R", type=int, default=6)
ap.add_argument("--window-s", type=float, nargs=2, default=(20.0, 40.0)); ap.add_argument("--proxy", action="store_true", help="정렬 없이 proxy 시각으로 토큰을 배치"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer); ids = add_phase2_specials(tok); NAMES = {v: k for k, v in ids.items()}

def proxy_tokens(d: Dialogue):
    """정렬 대신: word_timing 이 있으면 단어 끝 시각을 BPE 에 배분, 없으면 발화 구간에 균등 배치."""
    n = 0
    for u in d.utterances:
        if not u.text or u.tokens: continue
        enc = tok(" " + u.text, add_special_tokens=False)["input_ids"]
        if not enc: continue
        if u.word_timing:
            words = [w for w in u.word_timing if w[2] >= u.start - 1e-3]; ends = [w[2] for w in words]
            u.tokens = [(t, min(u.end, ends[min(len(ends) - 1, int(i * len(ends) / len(enc)))])) for i, t in enumerate(enc)] if ends else [(t, u.start + (u.end - u.start) * (i + 1) / len(enc)) for i, t in enumerate(enc)]
        else: u.tokens = [(t, u.start + (u.end - u.start) * (i + 1) / len(enc)) for i, t in enumerate(enc)]
        n += 1
    return n

# ── 데이터 로드
dlg_path = a.dialogues; tmp_path = None
if a.proxy:
    tmp_path = os.path.join(a.out, "_proxy.dialogues.jsonl"); n = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in open(a.dialogues, encoding="utf-8"):
            d = Dialogue.from_json(line); n += proxy_tokens(d); f.write(d.to_json() + "\n")
    dlg_path = tmp_path; print(f"proxy tokens for {n} utterances")
ds = DialogueWindowDataset([dlg_path], tok, align_dir=None if a.proxy else a.align, R=a.R, window_s=tuple(a.window_s), hop_s=10.0, delays=(a.delay,), seed=a.seed)
print("dataset:", json.dumps(ds.stats)); assert len(ds) > 0, "창이 없다 — 전사 없는 음성/미정렬 발화가 창을 전부 제외했는지 stats 확인"
rep = dict(dialogues=ds.stats, windows=[], tokenizer=a.tokenizer, delay=a.delay, R=a.R, proxy=a.proxy)
rng = random.Random(a.seed); picks = list(range(len(ds))); rng.shuffle(picks); picks = picks[: a.windows]

def check_window(i):
    s = ds.sequence(i, a.delay); d = ds.dlgs[s["cid"]]; eps, info = ds.window_episodes(d, s["t0"], s["L"]); K = s["K"]; errs = []
    chunks, st = serialize(eps, (K - 0.5) * CHUNK_S, ds.sp, delay_text=a.delay, delay_onset=0)
    # (1) 텍스트 라운드트립: 화자별로 lexical 토큰을 이으면 episode 토큰열과 같다
    got = {};
    for k, em in chunks:
        for e in em:
            if e.kind == "text": got.setdefault(e.ep, []).append(e.tid)
    for ep in eps:
        if ep.lane is None: continue
        ref = [t for t, _ in ep.tokens]
        if got.get(ep.ep_id, []) != ref: errs.append(f"roundtrip ep{ep.ep_id} {len(got.get(ep.ep_id, []))}!={len(ref)}")
    # (2) allocator 동등성 (N≤R)
    if len({e.speaker for e in eps}) <= a.R:
        import copy; e2 = copy.deepcopy(eps)
        for e in e2: e.lane = None; e.generation = 0
        allocate(e2, R=a.R, policy="never_free", delay_text=a.delay)
        if [(e.lane, e.generation) for e in eps] != [(e.lane, e.generation) for e in e2]: errs.append("never_free != lazy_free")
    # (3) 순서·EOT 청크 (4) soft (7) 태그
    pos = {}; sel = None; tags_ok = True
    for k, em in chunks:
        for j, e in enumerate(em):
            if e.kind in ("onset", "text", "eot"): pos.setdefault(e.ep, {}).setdefault(e.kind, []).append(k)
            if e.kind in ("onset", "eot") and not (j > 0 and em[j - 1].kind == "tag" and em[j - 1].lane == e.lane): tags_ok = False
            if e.kind == "tag": sel = e.lane
            if e.kind == "text" and (j == 0 or em[j - 1].kind != "tag") and sel != e.lane: tags_ok = False
    if not tags_ok: errs.append("tag rule")
    for ep in eps:
        if ep.lane is None: continue
        p = pos.get(ep.ep_id, {}); on = p.get("onset", [None])[0]; tx = p.get("text", []); eo = p.get("eot", [None])[0]
        if on is None or eo is None: errs.append(f"missing onset/eot ep{ep.ep_id}"); continue
        if tx and not (on <= min(tx) and max(tx) <= eo): errs.append(f"order ep{ep.ep_id}")
        want = eot_chunk(ep, a.delay)
        if (want < K and eo != want) or (want >= K and eo < K): errs.append(f"eot chunk ep{ep.ep_id} {eo}!={want}")     # K 이상은 flush 라운드(청크 K, K+1, …)
    f = s; eot_id = ds.sp.eot
    for j, w in zip(f["soft_pos"], f["soft_w"]):
        if f["ids"][j] != eot_id or not (0.0 <= w <= 1.0) or f["soft_alt"][f["soft_pos"].index(j)] != f["ids"][j + 1]: errs.append("soft label")
    n_eot_masked = sum(1 for j, t in enumerate(f["ids"]) if t == eot_id and f["labels"][j] == -100)
    # (5) 활동 = 참조 구간
    act = np.array(s["activity"]); ref = np.zeros_like(act)
    for ep in eps:
        if ep.lane is None: continue
        for k in range(max(0, int(ep.start / CHUNK_S)), min(K, int(np.ceil(ep.end / CHUNK_S)))): ref[k, ep.lane - 1] = 1
    if not np.array_equal(act, ref): errs.append("activity")
    # (6) 오디오 길이 (8) lane 부족
    wav = ds[i]["wav"].numpy()
    if abs(len(wav) - int(round(s["L"] * 16000))) > 1: errs.append(f"audio len {len(wav)} vs {int(round(s['L']*16000))}")
    if info["alloc"].exhausted: errs.append(f"lane exhausted {info['alloc'].exhausted}")
    mk = set(f.get("masked_chunks", []))                                                   # (9) 마스크 청크의 label 은 전부 -100, soft 도 없음
    if any(f["labels"][j] != -100 for j in range(len(f["labels"])) if f["chunk_of"][j] in mk) or any(f["chunk_of"][j] in mk for j in f["soft_pos"]): errs.append("mask")
    # 출력
    name = os.path.join(a.out, f"window-{i}"); sf.write(name + ".wav", wav, 16000)
    names = {**NAMES, **{t: tok.decode([t]) for _, em in chunks for e in em for t in [e.tid] if e.kind == "text"}}
    with open(name + ".txt", "w", encoding="utf-8") as fo:
        fo.write(f"# {s['cid']} t0={s['t0']} L={s['L']} K={K} delay={a.delay} speakers={sorted({e.speaker for e in eps})} lanes={ {e.speaker: e.lane for e in eps} }\n")
        fo.write(f"# episodes={len(eps)} reassigned={info['alloc'].reassigned} outcomes={info['outcomes']} soft={len(f['soft_pos'])} eot_masked={n_eot_masked} tokens={st.text} tags={st.tags} overflow={st.overflow} masked_chunks={len(mk)}/{K}\n")
        fo.write(f"# checks: {'OK' if not errs else errs}\n\n")
        for ep in eps: fo.write(f"ep{ep.ep_id:03d} spk={ep.speaker} lane={ep.lane} g{ep.generation} {ep.start:7.2f}-{ep.end:7.2f} outcome={ep.outcome} p_end={ep.p_end} tokens={len(ep.tokens)} :: {tok.decode([t for t,_ in ep.tokens])[:80]}\n")
        fo.write("\n" + render(chunks, names, K) + "\n")
    # 시각화용 JSON: 디코딩된 토큰·시각·청크 방출
    def clean(x): return x.replace("\ufffd", "\u25af")                        # 바이트 BPE 조각 단독 디코딩의 U+FFFD → ▯
    def word_groups(ep, emitk):
        """토큰 → 단어 묶음: 다음 토큰이 공백으로 시작하고 누적 디코딩이 깨지지 않았을 때 경계. (text, 참조 끝 t, 방출 청크 k, 조각 수)"""
        out = []; acc = []; ks = []; ts = []
        for i, (t, tt) in enumerate(ep.tokens):
            acc.append(t); ts.append(tt); ks.append(emitk[i] if i < len(emitk) else None)
            nxt = tok.decode([ep.tokens[i + 1][0]]) if i + 1 < len(ep.tokens) else " "
            txt = tok.decode(acc)
            if nxt.startswith(" ") and "\ufffd" not in txt:
                out.append(dict(text=clean(txt).strip(), t=round(ts[-1], 3), k=ks[-1], n=len(acc))); acc, ks, ts = [], [], []
        if acc: out.append(dict(text=clean(tok.decode(acc)).strip(), t=round(ts[-1], 3), k=ks[-1], n=len(acc)))
        return out
    emitk = {}
    for k, em in chunks:
        for e in em:
            if e.kind == "text": emitk.setdefault(e.ep, []).append(k)
    js = dict(conv=s["cid"], corpus=d.corpus, lang=d.lang, t0=s["t0"], L=s["L"], K=K, delay=a.delay, R=a.R, lanes={e.speaker: e.lane for e in eps},
              episodes=[dict(ep=e.ep_id, speaker=e.speaker, lane=e.lane, gen=e.generation, start=round(e.start, 3), end=round(e.end, 3), outcome=e.outcome, p_end=e.p_end,
                             tokens=[dict(text=clean(tok.decode([t])), t=round(tt, 3)) for t, tt in e.tokens], words=word_groups(e, emitk.get(e.ep_id, [])), text=clean(tok.decode([t for t, _ in e.tokens])).strip()) for e in eps],
              chunks=[dict(k=k, emits=[dict(kind=e.kind, lane=e.lane, ep=e.ep, text=(clean(tok.decode([e.tid])) if e.kind == "text" else NAMES.get(e.tid, str(e.tid))), p_end=e.p_end) for e in em]) for k, em in chunks],
              masked_chunks=sorted(mk), stats=dict(text=st.text, tags=st.tags, onset=st.onset, eot=st.eot, overflow=st.overflow, soft=len(f["soft_pos"]), eot_masked=n_eot_masked, reassigned=info["alloc"].reassigned, outcomes=info["outcomes"], masked=len(mk)))
    json.dump(js, open(name + ".json", "w"), ensure_ascii=False)
    return dict(index=i, conv=s["cid"], t0=s["t0"], L=s["L"], K=K, speakers=len({e.speaker for e in eps}), episodes=len(eps), reassigned=info["alloc"].reassigned, exhausted=info["alloc"].exhausted,
                outcomes=info["outcomes"], soft=len(f["soft_pos"]), eot_masked=n_eot_masked, text_tokens=st.text, tags=st.tags, overflow=st.overflow, max_per_chunk=max(st.per_chunk_hist) if st.per_chunk_hist else 0, masked_chunks=len(mk), errors=errs)

for i in picks:
    r = check_window(i); rep["windows"].append(r); print(json.dumps({k: v for k, v in r.items() if k != "outcomes"}, ensure_ascii=False))
rep["all_ok"] = all(not w["errors"] for w in rep["windows"])
json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1); print("ALL_OK" if rep["all_ok"] else "ERRORS")
if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)
