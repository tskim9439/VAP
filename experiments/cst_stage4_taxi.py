#!/usr/bin/env python
"""Stage 4 TAXI 평가(S4) — CST-Bench-TAXI(실제 독↔영 전화 대화, cstbench render 의 세션 mono 혼합)를 교차 언어 prefix 로 스트리밍 디코드하고
번역 흐름을 cstbench 가설 형식으로 써서 cstbench 채점기로 잰다(wiki output-cst-taxi-offline-baselines-20261005, output-stage4-prep-plan-20261019 S4).

  python experiments/cst_stage4_taxi.py run --checkpoint <ck> --encoder <nemotron dir> --sessions <cstbench-v0.1/taxi-natural/sessions.jsonl> --out <dir> \\
      [--deltas 4,10] [--max-window 40] [--batch-size 16]
  python experiments/cst_stage4_taxi.py score --sessions <…/sessions.jsonl> --out <dir>

창: 세션이 --max-window(s)보다 길면 턴 사이 쉼 가운데에서 나눠 따로 디코드한다(학습 창 20–40 s). 창마다 prefix
  'language <SPK_A>{L_A} <SPK_B>{L_B}\\ntranslate <SPK_A>{L_B} <SPK_B>{L_A}<asr_text><DELAY_d>' — 두 언어를 이름순으로 두고 세션 id 해시로 반을 뒤집는다
  (CstDialogueDataset.slots 와 같은 규칙).
가설: 슬롯 k 의 번역 단어(방출 시각 = 청크 끝 (k+1)·80 ms + 창 시작) → hyp-d{δ}.jsonl {"session", "lang": 슬롯 k 의 목표 언어, "t", "text"} 한 단어씩.
  받아쓰기는 asr-d{δ}.jsonl(세션·슬롯 언어·단어·시각). --asr-only: translate 줄 없이(받아쓰기만 — 번역 형식과 받아쓰기를 가른다).
채점(score): cstbench.evaluate(세션·출력 언어마다 참조 번역에 단어 정렬로 재분절 → BLEU·chrF·StreamLAAL·end_offset·빈 턴) → report-d{δ}.json,
  받아쓰기 WER(세션·언어마다 이어 붙인 참조 전사 대 그 언어 슬롯 전사, cstbench.normalize) → asr-report-d{δ}.json.
TAXI 전사·번역은 재배포 금지라 hyp·asr 파일은 mxc 에만 두고 저장소에는 report 만 가져온다."""
import argparse, json, os, re, sys, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cst-bench", "src"))
from cst_stage3_eval import words_of  # noqa: E402


def slots_of(rec):
    """세션 → (화자 [A, B], 언어 [A, B]) — 언어 이름순, 세션 id 해시로 반을 뒤집는다."""
    sp = sorted(rec["speakers"], key=lambda s: rec["speakers"][s]["lang"])
    if zlib.crc32(rec["session_id"].encode()) % 2: sp = sp[::-1]
    return sp, [rec["speakers"][s]["lang"] for s in sp]


def windows_of(rec, max_s):
    """세션 → 창 [(t0, t1)]: 길면 턴 사이 쉼 가운데에서 나눈다(창 안 첫 턴부터 max_s 를 넘기 전까지)."""
    T = rec["duration_s"]
    if T <= max_s: return [(0.0, T)]
    turns = sorted(rec["turns"], key=lambda t: t["start_s"]); out = []; t0 = 0.0
    for a, b in zip(turns, turns[1:]):
        cut = (a["end_s"] + b["start_s"]) / 2
        if b["end_s"] - t0 > max_s and cut > t0 + 5.0: out.append((t0, cut)); t0 = cut
    out.append((t0, T))
    return out


def simple_words(s):
    """두 언어 공통 정규화(소문자·문장부호 제거) — cp 와 화자 무시 WER 을 같은 규칙으로 재서 귀속 몫(cp − agn)을 가른다."""
    return re.sub(r"[^\w' ]+", " ", s.lower()).split()


def attr_split(sessions, asr):
    """세션 단위 (cp 오류, 화자 무시 오류, 참조 단어 수) — 공통 정규화. cp = 언어 슬롯 순열 중 작은 쪽(2 언어 세션), 화자 무시 = 참조 턴 전부를 시각순으로 이은 것 대
    가설 단어 전부를 방출 시각순으로 이은 것. VibeVoice 대비 분해(output-vibevoice-stream-comparison-20261020)와 같은 정의."""
    from vapasr.data.cst_eval import ed
    by = {}
    for x in asr: by.setdefault(x["session"], []).append(x)
    e_cp = e_agn = n = 0
    for s in sessions:
        turns = sorted(s["turns"], key=lambda t: t["start_s"]); langs = sorted({t["lang"] for t in turns})
        R = [[w for t in turns if t["lang"] == lg for w in simple_words(t["transcript"])] for lg in langs]
        hw = sorted(by.get(s["session_id"], []), key=lambda x: x["t"])
        H = [[w for x in hw if x["lang"] == lg for w in simple_words(x["text"])] for lg in langs]
        same = sum(ed(R[k], H[k]) for k in range(len(langs)))
        e_cp += min(same, ed(R[0], H[1]) + ed(R[1], H[0])) if len(langs) == 2 else same
        e_agn += ed([w for t in turns for w in simple_words(t["transcript"])], [w for x in hw for w in simple_words(x["text"])]); n += sum(len(r) for r in R)
    return e_cp, e_agn, n


def run(a):
    import numpy as np, soundfile as sf, torch
    from vapasr.hf.infer import load_model
    from vapasr.hf.batch_decode import decode_batch, SwitchBias, word_init_mask, logits_size
    from vapasr.data.cst_dialogue import cst_head_text
    from vapasr.data.cst_eval import parse_emitted4
    from collections import Counter
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sb = SwitchBias.parse(a.switch_bias) if getattr(a, "switch_bias", None) else None              # V3 C3 상태 인지 교대 bias(교차 언어 창만) — 없으면 예전 디코드 그대로
    model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=getattr(torch, a.dtype)); sp = model.config.sp_ids
    wi = word_init_mask(tok, logits_size(model)) if sb else None; sbc = {d: (Counter(), Counter()) for d in a.deltas}
    from vapasr.data.cst_dialogue import SLOTS
    spk = tuple(sp[t] for t in SLOTS[: len(getattr(model.config, "slot_tokens", None) or SLOTS[:2])])                        # V2(4 슬롯)도 출력은 모든 슬롯 태그로 해석 — C·D 로 낸 말은 채점(A·B 언어 슬롯)에서 빠진다(삭제)
    EOT = sp["<EOT>"]; TRANS = int(tok.convert_tokens_to_ids("<TRANS>")); dev = next(model.parameters()).device
    pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    recs = [json.loads(l) for l in open(a.sessions, encoding="utf-8")]; root = os.path.dirname(os.path.abspath(a.sessions))
    jobs = [(r, w) for r in recs for w in windows_of(r, a.max_window)]
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    json.dump(dict(checkpoint=os.path.abspath(a.checkpoint), sessions=os.path.abspath(a.sessions), deltas=a.deltas, max_window=a.max_window, windows=len(jobs), asr_only=a.asr_only,
                   **({"switch_bias": sb.policy()} if sb else {}), slots={r["session_id"]: slots_of(r) for r in recs}), open(os.path.join(a.out, "config.json"), "w"), indent=1, ensure_ascii=False)
    fh = {d: open(os.path.join(a.out, f"hyp-d{d}.jsonl"), "w", encoding="utf-8") for d in a.deltas}
    fa = {d: open(os.path.join(a.out, f"asr-d{d}.jsonl"), "w", encoding="utf-8") for d in a.deltas}
    mix = {}
    with torch.inference_mode():
        for b0 in range(0, len(jobs), a.batch_size):
            batch = jobs[b0:b0 + a.batch_size]; feats = []
            for r, (w0, w1) in batch:
                if r["session_id"] not in mix:
                    x, sr = sf.read(os.path.join(root, r["mix"]), dtype="float32"); assert sr == 16000 and x.ndim == 1, f"{r['mix']}: {sr} Hz {x.shape}"
                    mix[r["session_id"]] = x
                x = mix[r["session_id"]]; K = int(round((w1 - w0) / 0.08)); seg = x[int(round(w0 * 16000)): int(round(w0 * 16000)) + K * 1280]
                seg = np.pad(seg, (0, K * 1280 - len(seg))); w = torch.from_numpy(np.ascontiguousarray(seg)).to(dev)
                feats.append(model.encode(w[None], torch.tensor([len(seg)], device=dev), torch.tensor([K], device=dev))[0])
            for d in a.deltas:
                prefixes = []
                for r, _ in batch:
                    _, langs = slots_of(r); tg = None if a.asr_only else [langs[1], langs[0]]
                    prefixes.append(pre + tok(cst_head_text(langs, tg), add_special_tokens=False)["input_ids"] + [int(tok.convert_tokens_to_ids(f"<DELAY_{d}>"))])
                modes = ["xl" if len(set(slots_of(r)[1])) > 1 else "same" for r, _ in batch] if sb else None   # 교차 언어 = 선언한 두 언어가 다름
                states = decode_batch(model, feats, prefixes, max_flush_rounds=a.max_flush, switch_bias=sb, row_modes=modes, word_init=wi)
                for s in states if sb else (): sbc[d][0].update(s.sb_flips); sbc[d][1].update(s.sb_points)
                for (r, (w0, w1)), s in zip(batch, states):
                    _, langs = slots_of(r); src, tr, _ = parse_emitted4(s.emitted, spk, TRANS, EOT)
                    for k in range(2):
                        for wd, t in words_of(tok, tr[k]): fh[d].write(json.dumps(dict(session=r["session_id"], lang=langs[1 - k], t=round(w0 + t, 3), text=wd), ensure_ascii=False) + "\n")
                        for wd, t in words_of(tok, src[k]): fa[d].write(json.dumps(dict(session=r["session_id"], lang=langs[k], t=round(w0 + t, 3), text=wd), ensure_ascii=False) + "\n")
            print(f"  {min(b0 + a.batch_size, len(jobs))}/{len(jobs)} 창 · {time.time() - t0:.0f}s", flush=True)
    for f in list(fh.values()) + list(fa.values()): f.close()
    if sb:                                                                                       # 바꾼 태그·결정 지점 수(δ별 합, 집계만 — 세션·글자 없음)
        res = dict(policy=sb.policy(), **{f"d{d}": dict(flips=dict(f), points=dict(p)) for d, (f, p) in sbc.items()})
        json.dump(res, open(os.path.join(a.out, "switch-bias.json"), "w"), indent=1); print("SWITCH_BIAS", json.dumps(res), flush=True)
    print("DONE", flush=True)


def score(a):
    import glob
    from cstbench.evaluate import evaluate, load_jsonl, normalize
    from vapasr.data.cst_eval import ed
    sessions = load_jsonl(a.sessions)
    for f in sorted(glob.glob(os.path.join(a.out, "hyp-d*.jsonl"))):
        d = f.rsplit("-d", 1)[1].split(".")[0]
        rep, _ = evaluate(sessions, load_jsonl(f), "t"); rep.update(hyp=f, sessions=a.sessions)
        json.dump(rep, open(os.path.join(a.out, f"report-d{d}.json"), "w"), indent=1, ensure_ascii=False)
        asr = load_jsonl(os.path.join(a.out, f"asr-d{d}.jsonl")); res = {}; langs = sorted({t["lang"] for s in sessions for t in s["turns"]})
        ref_of = lambda s, lg: " ".join(t["transcript"] for t in sorted(s["turns"], key=lambda t: t["start_s"]) if t["lang"] == lg)
        hyp_of = lambda s, lg: " ".join(x["text"] for x in sorted((x for x in asr if x["session"] == s["session_id"] and x["lang"] == lg), key=lambda x: x["t"]))
        for lang in langs:
            e = n = 0
            for s in sessions:
                ref = normalize(ref_of(s, lang), lang).split(); e += ed(ref, normalize(hyp_of(s, lang), lang).split()); n += len(ref)
            res[lang] = dict(wer=round(100 * e / max(1, n), 2), ref_words=n)
        if len(langs) == 2:                                                                     # 슬롯 순열을 허용한 WER(세션마다 작은 쪽) — 받아쓰기 자체와 슬롯·언어 어긋남을 가른다
            e_id = e_cp = n = 0
            for s in sessions:
                R = [normalize(ref_of(s, lg), lg).split() for lg in langs]; H = [hyp_of(s, lg) for lg in langs]
                same = sum(ed(R[k], normalize(H[k], langs[k]).split()) for k in range(2)); cross = sum(ed(R[k], normalize(H[1 - k], langs[k]).split()) for k in range(2))
                e_id += same; e_cp += min(same, cross); n += len(R[0]) + len(R[1])
            res["both"] = dict(wer_slot=round(100 * e_id / max(1, n), 2), wer_cp=round(100 * e_cp / max(1, n), 2), ref_words=n)
            c, g, m = attr_split(sessions, asr)                                                 # 공통 정규화로 받아쓰기(agn)와 귀속 몫을 가른다
            res["split"] = dict(cp=round(100 * c / max(1, m), 2), agn=round(100 * g / max(1, m), 2), attr=round(100 * (c - g) / max(1, m), 2), ref_words=m)
        json.dump(res, open(os.path.join(a.out, f"asr-report-d{d}.json"), "w"), indent=1, ensure_ascii=False)
        for k, x in rep["directions"].items():
            print(f"δ{d} {k}: 턴 {x.get('turns')} · BLEU {x.get('bleu')} · chrF {x.get('chrf')} · StreamLAAL 중앙 {(x.get('stream_laal_s') or {}).get('median')} s · "
                  f"끝 지연 중앙 {(x.get('end_offset_s') or {}).get('median')} s · 빈 턴 {x.get('empty_turns')}", flush=True)
        print(f"δ{d} 받아쓰기 WER: {res}", flush=True)
    if getattr(a, "turn_metrics", False):                                                       # 턴 결정 지표(cst_attr_turns, 기본 끔 — 끄면 위 산출·출력 그대로): asr-d*.jsonl 마다, 같은 폴더 eot-d*.jsonl 이 있으면 함께
        from cst_attr_turns import turn_metrics, brief
        for f in sorted(glob.glob(os.path.join(a.out, "asr-d*.jsonl"))):
            d = f.rsplit("-d", 1)[1].split(".")[0]; fe = os.path.join(a.out, f"eot-d{d}.jsonl")
            tm = turn_metrics(sessions, load_jsonl(f), load_jsonl(fe) if os.path.exists(fe) else None); tm.update(asr=f, eot_file=fe if os.path.exists(fe) else None, sessions_file=a.sessions)
            json.dump(tm, open(os.path.join(a.out, f"turn-metrics-d{d}.json"), "w"), indent=1, ensure_ascii=False)
            print(f"δ{d} 턴 결정: {brief(tm)}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--checkpoint", required=True); q.add_argument("--encoder", default=None); q.add_argument("--sessions", required=True)
    q.add_argument("--out", required=True); q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 10])
    q.add_argument("--max-window", type=float, default=40.0); q.add_argument("--batch-size", type=int, default=16); q.add_argument("--max-flush", type=int, default=None)
    q.add_argument("--dtype", default="float32"); q.add_argument("--asr-only", action="store_true", help="translate 줄 없는 받아쓰기 전용 prefix(Stage 3 형식, 언어만 슬롯별)")
    q.add_argument("--switch-bias", default=None, help="V3 C3 상태 인지 교대 bias 'eot=B,res=B,wi=B'(batch_decode.SwitchBias — 선언 두 언어가 다른 창만, 기본 끔). "
                                                       "config.json 에 정책, switch-bias.json 에 δ별 바꾼 태그·결정 지점 수. 값은 dev 에서만 고른다")
    q = sub.add_parser("score"); q.add_argument("--sessions", required=True); q.add_argument("--out", required=True); q.add_argument("--turn-metrics", action="store_true", help="asr-d*.jsonl 마다 턴 결정 지표(cst_attr_turns) → turn-metrics-d*.json(기본 끔)")
    a = ap.parse_args(); {"run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
