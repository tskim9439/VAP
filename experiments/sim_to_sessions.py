#!/usr/bin/env python
"""합성 대화 셋 → cstbench 세션 형식(V3 C4 — TAXI 형식의 비-TAXI 전화 dev, 예 xl-tel-dev·xl-eval-v1). <data>/<corpus>.refined.dialogues.jsonl 과 mono 캐시를
TAXI 렌더(cstbench render)와 같은 키의 sessions.jsonl·mix wav 로 바꿔 cst_stage4_taxi.py run/score(번역 prefix·세션 통째·cp/agn/attr)와
cst_attr_turns(턴 결정 지표)를 TAXI 없이 쓰게 한다.

  python experiments/sim_to_sessions.py --data <sim 셋 폴더> --corpus sim_en-de-tel --tokenizer <Stage 3/4 체크포인트> --out <새 폴더> \\
      [--limit 160] [--targets <targets 폴더> | --no-trans] [--mono <mono 폴더>]

세션 = 대화 하나: session_id = conv_id, speakers {화자: {lang, role None}} — 교차 언어는 meta.spk_lang(CstDialogueDataset.xl_langs 와 같은 판정), 아니면 Dialogue.lang.
턴 = episode(dialogue.build_episodes — 같은 화자 발화를 gap < 0.25 s 로 병합, 학습 직렬화의 <EOT> 단위), 시각순: turn_id t<ep_id>·speaker·lang·start_s·end_s·
  gap_before_s(앞 턴 끝부터, 겹치면 음수)·transcript = tokenizer.decode(그 episode 정렬 토큰, 시각순).strip() — 학습 타깃 전사와 같은 토큰.
번역 참조: targets-<corpus>/(load_ast_targets 형식, 행 id = <conv>|<화자>|<utt_id 의 # 앞>)가 있으면 화자마다 상대 화자 언어 번역을 CstDialogueDataset._rows·trans_of 와
  같은 규칙(되살림 없음, 창 = 대화 전체)으로 턴에 붙인다 — 확정 시각을 행 조각 구간으로 자르고(clamp_to_spans) 그 시각이 든 episode 에, 행마다 decode 해 공백으로
  잇는다. translation_lang = 상대 언어. 번역 토큰이 없는 턴은 translation None(cstbench.evaluate 는 번역 참조가 있는 턴만 채점 — 번역이 전혀 없으면 BLEU 방향이
  비고 cst_stage4_taxi score 의 받아쓰기·cp/agn/attr 는 그대로 나온다). 같은 언어 대화는 번역 참조를 붙이지 않는다.
mix: <out>/mix/<conv>.wav — mono 캐시(<mono>/<corpus>/<conv>.npy float16, dialogue_dataset.mono_cache_path, 기본 <data>/mono)를 ±1 로 잘라 16 kHz PCM16 으로.
  duration_s = 샘플 수 / 16000. 캐시가 없는 대화는 건너뛰고 센다.
출력: <out>/sessions.jsonl(키 정렬), <out>/mix/*.wav, <out>/build_info.json(수·시간·턴 통계·인자). --out 은 없어야 한다(덮어쓰지 않는다). 아무것도 지우지 않는다."""
import argparse, collections, json, os, sys, time
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cst-bench", "src"))
SR = 16000


def turn_translations(d, eps, TR, tgt_of, tok):
    """episode 마다 번역 참조 → {ep_id: (텍스트 | None, 행 수, 번역 없는 행 수)}. tgt_of = {화자: 목표 언어}.
    CstDialogueDataset._rows(행 = (화자, utt_id 의 # 앞), 조각 = 토큰 있고 격리 아닌 발화)·trans_of(행이 episode 여럿이면 확정 시각이 든 episode 에)와 같은 규칙."""
    from vapasr.data.dialogue import clamp_to_spans
    piece = {u.utt_id: (u.utt_id.split("#")[0], u.start, u.end) for u in d.utterances if u.tokens and not u.flags}; rows = {}
    for ep in eps:
        for uid in ep.utt_ids:
            if uid not in piece: continue
            b, a, e = piece[uid]; r = rows.setdefault((ep.speaker, b), dict(a=a, e=e, eps=[], spans=[]))
            r["a"], r["e"] = min(r["a"], a), max(r["e"], e); r["spans"].append((a, e))
            if not r["eps"] or r["eps"][-1] != ep.ep_id: r["eps"].append(ep.ep_id)
    for r in rows.values(): r["spans"].sort()
    out = {}
    for ep in eps:
        parts, seen, miss = [], set(), 0
        for uid in ep.utt_ids:
            key = (ep.speaker, uid.split("#")[0]); r = rows.get(key)
            if r is None or key in seen: continue
            seen.add(key); tt = TR.get(f"{d.conv_id}|{key[0]}|{key[1]}", {}).get(tgt_of.get(ep.speaker))
            if tt is None: miss += 1; continue
            own = (lambda t: ep.ep_id) if len(r["eps"]) == 1 else (lambda t, r=r: next((j for j in r["eps"] if t <= eps[j].end + 1e-6), r["eps"][-1]))
            ids = [int(i) for i, t in zip(tt[0], tt[1]) if own(clamp_to_spans(float(t), r["spans"])) == ep.ep_id]
            if ids: parts.append(tok.decode(ids).strip())
        txt = " ".join(p for p in parts if p); out[ep.ep_id] = (txt or None, len(seen), miss)
    return out


def session_of(d, tok, TR=None):
    """Dialogue → (cstbench 세션 dict(mix·duration_s 제외), 턴 통계 Counter)."""
    from vapasr.data.dialogue import build_episodes
    from vapasr.data.cst_dialogue import CstDialogueDataset
    from cstbench.timeline import overlap_ratio
    sl = CstDialogueDataset.xl_langs(d); lang = {s: (sl[s] if sl else d.lang) for s in d.speakers}
    eps = build_episodes(d); st = collections.Counter(turns=len(eps))
    tgt_of = {s: next(lang[o] for o in d.speakers if o != s) for s in d.speakers} if sl else {}
    tr = turn_translations(d, eps, TR, tgt_of, tok) if TR and tgt_of else {}
    bad = {u.utt_id for u in d.utterances if u.flags or not u.tokens}; turns = []; prev = None
    for ep in eps:
        txt, nrow, miss = tr.get(ep.ep_id, (None, 0, 0))
        st["untranscribed"] += any(u in bad for u in ep.utt_ids); st["no_tokens"] += not ep.tokens
        if tgt_of: st["translated"] += txt is not None; st["translation_missing_rows"] += miss; st["translation_partial"] += txt is not None and miss > 0
        turns.append(dict(turn_id=f"t{ep.ep_id:03d}", speaker=ep.speaker, lang=lang[ep.speaker], start_s=round(ep.start, 4), end_s=round(ep.end, 4),
                          gap_before_s=None if prev is None else round(ep.start - prev.end, 4), transcript=tok.decode([int(tid) for tid, _ in ep.tokens]).strip() if ep.tokens else "",
                          translation=txt, translation_lang=tgt_of.get(ep.speaker) if txt is not None else None, trimmed_s=None))
        prev = ep
    placed = [SimpleNamespace(start_s=e.start, end_s=e.end, speaker=e.speaker) for e in eps]                       # cstbench.timeline.overlap_ratio 입력(Placed 의 시각·화자)
    rec = dict(session_id=d.conv_id, corpus=d.corpus, redistributable=False, speakers={s: dict(lang=lang[s], role=None) for s in d.speakers}, channels=list(d.speakers),
               sample_rate=SR, config="sim_to_sessions", seed=None, overlap_ratio=round(overlap_ratio(placed), 4), skipped=[], two_channel=None, turns=turns)
    return rec, st


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True, help="<corpus>.refined.dialogues.jsonl·mono/·targets-<corpus>/ 가 든 폴더")
    ap.add_argument("--corpus", required=True); ap.add_argument("--tokenizer", required=True, help="전사·번역 토큰을 decode 할 tokenizer(학습 체크포인트 폴더)")
    ap.add_argument("--out", required=True, help="새 폴더(있으면 멈춘다)"); ap.add_argument("--mono", default=None, help="mono 캐시 폴더(기본 <data>/mono)")
    ap.add_argument("--targets", default=None, help="번역 타깃 폴더(기본 <data>/targets-<corpus> — 있으면)"); ap.add_argument("--no-trans", action="store_true", help="번역 참조를 붙이지 않는다")
    ap.add_argument("--limit", type=int, default=0, help="파일 앞에서부터 이 수의 대화만(0 = 전부)"); a = ap.parse_args()
    convert(a)


def convert(a, tok=None):
    """main 본체(시험은 tok 을 넘긴다) → build_info dict."""
    import numpy as np
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.dialogue_dataset import mono_cache_path
    from cstbench.audio import write_pcm16
    if os.path.exists(a.out): sys.exit(f"--out 이 이미 있다: {a.out} (덮어쓰지 않는다)")
    src = os.path.join(a.data, f"{a.corpus}.refined.dialogues.jsonl"); mono = a.mono or os.path.join(a.data, "mono"); t0 = time.time()
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    td = None if a.no_trans else (a.targets or os.path.join(a.data, f"targets-{a.corpus}")); TR = None
    if td and os.path.isdir(td):
        from vapasr.data.ast_dataset import load_ast_targets
        TR, tr_st, _ = load_ast_targets([td]); print(f"번역 타깃 {dict(tr_st)} · {time.time() - t0:.0f}s", flush=True)
    elif a.targets: sys.exit(f"--targets 폴더가 없다: {a.targets}")
    os.makedirs(os.path.join(a.out, "mix")); recs, hours = [], 0.0
    st = collections.Counter({k: 0 for k in ("dialogues", "no_mono", "xl", "turns", "untranscribed", "no_tokens", "translated", "translation_partial", "translation_missing_rows")})
    for line in open(src, encoding="utf-8"):
        if a.limit and st["dialogues"] >= a.limit: break
        d = Dialogue.from_json(line); st["dialogues"] += 1; f = mono_cache_path(mono, d)
        if not os.path.exists(f): st["no_mono"] += 1; continue
        y = np.clip(np.asarray(np.load(f), dtype=np.float32), -1.0, 1.0); rec, ts = session_of(d, tok, TR); st.update(ts)
        name = d.conv_id.replace("/", "_").replace(":", "_"); write_pcm16(os.path.join(a.out, "mix", f"{name}.wav"), y, SR)
        rec.update(mix=f"mix/{name}.wav", duration_s=round(len(y) / SR, 3)); recs.append(rec); hours += len(y) / SR / 3600
        st["xl"] += len({v["lang"] for v in rec["speakers"].values()}) > 1
    with open(os.path.join(a.out, "sessions.jsonl"), "w", encoding="utf-8") as g:
        for r in recs: g.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    info = dict(source=os.path.abspath(src), corpus=a.corpus, mono=os.path.abspath(mono), targets=os.path.abspath(td) if TR is not None else None, tokenizer=a.tokenizer,
                sessions=len(recs), hours=round(hours, 3), stats=dict(st), limit=a.limit, sec=round(time.time() - t0, 1))
    json.dump(info, open(os.path.join(a.out, "build_info.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: info[k] for k in ("sessions", "hours", "stats")}, ensure_ascii=False), flush=True)
    return info


if __name__ == "__main__":
    main()
