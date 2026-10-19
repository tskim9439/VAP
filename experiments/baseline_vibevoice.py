#!/usr/bin/env python
"""외부 기준선 VibeVoice-ASR(Microsoft, MIT) — Stage 3 회귀 평가와 같은 held-out 대화 창을 디코드해 같은 참조·단위로 cp·agn 을 잰다.

  python experiments/baseline_vibevoice.py run --mode stream|offline --model <ckpt> --src <VibeVoice 소스> [--lm-tokenizer <Qwen2.5-7B tokenizer>] \\
      --checkpoint <우리 체크포인트(참조 tokenizer)> --data <phase2> --split <v2.json> --out <dir> [--corpora …] [--limit-windows 200] [--rank r --world-size n]
  python experiments/baseline_vibevoice.py score --out <dir> [--compare 이름=stage3-eval 결과 폴더,…]

모델: stream = VibeVoice-ASR-Streaming-7B(model.streaming_generate, 체크포인트의 청크·lookahead), offline = VibeVoice-ASR(창 전체 한 번, JSON 출력 Start/End/Speaker/Content).
  둘 다 탐욕 디코딩, bf16, sdpa. 입력은 우리 평가와 같은 mono 혼합 창(16 kHz)을 24 kHz 로 올린 것(resample_poly 3/2).
참조: cst_stage3_eval.window_ref(δ 4·10) 그대로 — 전사 없는 발화(마스크)의 말은 참조에서 빠진다.
가설 단어 시각: offline 은 구간 [Start, End] 안에서 단어 순서대로 고르게, stream 은 그 단어가 나온 청크 구간 [cT, (c+1)T] 안에서 고르게 놓는다(근사).
  마스크: 참조가 버리는 범위와 같은 시간 범위 — 전사 없는 구간 [s, e] 에 대해 단어 끝 시각이 [s − δ·0.08, e + 0.32] 이면 버린다.
지표: cp = 가설 화자(몇 명이든)와 참조 두 슬롯의 최적 짝(남는 가설 화자 말은 삽입), agn = 화자 무시 시간순, 단위는 cst_eval.units(한국어 글자·영독 단어).
  마스크 없는 창만의 cp·agn 도 따로 낸다(시각 근사의 영향을 보는 대조).
산출: <out>/predictions-rank{r}.jsonl(창마다 원출력·참조·마스크), score.json. 아무것도 지우지 않는다."""
import argparse, collections, glob, itertools, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_vibevoice(a):
    import torch
    sys.path.insert(0, a.src)
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    kw = {"language_model_pretrained_name": a.lm_tokenizer} if a.mode == "offline" else {}
    proc = VibeVoiceASRProcessor.from_pretrained(a.model, **kw)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    frame = None
    if a.mode == "stream":
        cfg = json.load(open(os.path.join(a.model, "preprocessor_config.json")))
        fs = cfg["speech_tok_compress_ratio"] / cfg["target_sample_rate"]
        frame = dict(sample_rate=cfg["target_sample_rate"], chunk=cfg["chunk_frames"] * fs, lookahead=cfg["lookahead_frames"] * fs)
        assert proc.tokenizer.text_chunk_end_id is not None, "스트리밍 체크포인트가 아니다(<|text_chunk_end|> 없음)"
    return model, proc, frame


def run(a):
    import numpy as np, torch
    from scipy.signal import resample_poly
    from transformers import AutoTokenizer
    from cst_stage3_eval import windows_of, window_ref
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.checkpoint); EOT = tok.convert_tokens_to_ids("<EOT>")
    sets, items = windows_of(a, tok)
    items = [x for j, x in enumerate(items) if j % a.world_size == a.rank]
    os.makedirs(a.out, exist_ok=True)
    conf = dict(mode=a.mode, model=os.path.abspath(a.model), src=os.path.abspath(a.src), corpora=a.corpora, limit_windows=a.limit_windows, deltas=a.deltas,
                world_size=a.world_size, max_new_tokens=a.max_new_tokens, resample="resample_poly 16k→24k (3/2)", decoding="greedy bf16 sdpa")
    cp = os.path.join(a.out, f"config-rank{a.rank}.json")
    if os.path.exists(cp): assert json.load(open(cp)) == conf, "재개 설정 불일치 — 새 --out 을 쓴다"
    else: json.dump(conf, open(cp, "w"), indent=1)
    dest = os.path.join(a.out, f"predictions-rank{a.rank}.jsonl")
    done = {r["id"] for r in map(json.loads, open(dest, encoding="utf-8"))} if os.path.exists(dest) else set()
    model, proc, frame = load_vibevoice(a); t0 = time.time(); n = 0
    print(f"VibeVoice {a.mode} 적재 · 창 {len(items)}(완료 {len(done)})" + (f" · 청크 {frame['chunk']:.3f}s lookahead {frame['lookahead']:.3f}s" if frame else ""), flush=True)
    todo = [(c, i) for c, i in items if f"{sets[c].items[i][0]}@{sets[c].items[i][1]}" not in done]
    with open(dest, "a", buffering=1, encoding="utf-8") as fo, torch.inference_mode():
        for b0 in range(0, len(todo), a.batch_size if a.mode == "offline" else 1):
            batch = todo[b0:b0 + (a.batch_size if a.mode == "offline" else 1)]; wavs = []; meta = []
            for c, i in batch:
                ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]
                x16 = np.asarray(ds.mono(cid)[int(round(t0w * 16000)): int(round((t0w + L) * 16000))], dtype=np.float32)
                wavs.append(resample_poly(x16, 3, 2).astype(np.float32)); meta.append((c, i, cid, t0w, L, d))
            st = time.time()
            if a.mode == "stream":
                chunks = [txt for _, _, txt in model.streaming_generate(audio_tensor=torch.from_numpy(wavs[0]), tokenizer=proc.tokenizer, chunk_duration=frame["chunk"],
                                                                         text_audio_delay=frame["lookahead"], sample_rate=frame["sample_rate"],
                                                                         max_new_tokens_per_chunk=a.max_new_tokens, temperature=0.0)]
                outs = [dict(chunks=chunks, chunk_s=frame["chunk"], lookahead_s=frame["lookahead"])]
            else:
                inp = proc(audio=wavs, sampling_rate=None, return_tensors="pt", padding=True, add_generation_prompt=True)
                inp = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
                ids = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=proc.pad_id, eos_token_id=proc.tokenizer.eos_token_id)
                outs = []
                for j in range(len(wavs)):
                    g = ids[j, inp["input_ids"].shape[1]:]; eos = (g == proc.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos): g = g[:eos[0] + 1]
                    text = proc.decode(g, skip_special_tokens=True)
                    outs.append(dict(raw=text, segments=proc.post_process_transcription(text)))
            el = time.time() - st
            for (c, i, cid, t0w, L, d), o in zip(meta, outs):
                ds = sets[c]; refs = {}
                for delta in a.deltas:
                    ref_w, _, mask, sp_order = window_ref(ds, i, delta, EOT, tok)
                    refs[str(delta)] = dict(words=[[[w, round(t - t0w, 3)] for w, t in r] for r in ref_w[:2]], masked_chunks=len(mask))
                bad = [[round(max(0.0, s - t0w), 3), round(min(L, e - t0w), 3)] for s, e in ds._bad_spans(d) if e > t0w and s < t0w + L]
                fo.write(json.dumps(dict(id=f"{cid}@{t0w}", corpus=c, lang=d.lang, L=L, refs=refs, bad_spans=bad, speakers=sp_order, out=o, decode_s=round(el / len(meta), 3)),
                                    ensure_ascii=False) + "\n"); n += 1
            print(f"  {n}/{len(todo)} 창 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


SPK_RE = re.compile(r"(?:\[\s*)?(?:Speaker|SPEAKER|speaker|Spk|S)\s*[_ ]?(\d+)\s*(?:\])?\s*[:：]?")


def hyp_words(r):
    """→ [(화자, 단어, 창 기준 시각 s)]. offline: JSON 구간, stream: 청크 텍스트(화자 표지 정규식, 없으면 화자 0)."""
    o = r["out"]; out = []
    if "segments" in o:
        for sg in o["segments"]:
            ws = str(sg.get("text", "")).split()
            try: s, e = float(sg.get("start_time", 0)), float(sg.get("end_time", 0))
            except (TypeError, ValueError): s = e = 0.0
            e = max(e, s)
            for k, w in enumerate(ws): out.append((str(sg.get("speaker_id", 0)), w, s + (e - s) * (k + 1) / len(ws)))
        return out
    T = o["chunk_s"]; spk = "0"
    for c, txt in enumerate(o["chunks"]):
        parts = SPK_RE.split(txt); toks = []                                    # [앞 텍스트, 화자, 텍스트, 화자, 텍스트, …]
        for j, p in enumerate(parts):
            if j % 2 == 1: spk = p; continue
            toks += [(spk, w) for w in re.sub(r'["{}\[\],]|Content\s*:|Speaker\s*:', " ", p).split()]
        for k, (sp, w) in enumerate(toks): out.append((sp, w, c * T + T * (k + 1) / len(toks)))
    return out


def score_row(r, delta):
    from vapasr.data.cst_eval import units, ed
    lang = r["lang"]; ref = r["refs"][str(delta)]; tail = delta * 0.08
    U = lambda ws: [u for w in ws for u in units(w, lang)]
    keep = lambda t: not any(s - tail <= t <= e + 0.32 for s, e in r["bad_spans"])
    hw = [(sp, w, t) for sp, w, t in hyp_words(r) if keep(t)]
    R = [U([w for w, _ in slot]) for slot in ref["words"]]
    by = collections.defaultdict(list)
    for sp, w, _ in hw: by[sp].append(w)
    H = sorted((U(ws) for ws in by.values()), key=len, reverse=True)
    if len(H) > 6: H = H[:5] + [[u for h in H[5:] for u in h]]                  # 화자가 아주 많으면 작은 쪽을 하나로(계산량)
    n = max(2, len(H)); Rp = R + [[] for _ in range(n - 2)]; Hp = H + [[] for _ in range(n - len(H))]
    cp = min(sum(ed(Rp[k], Hp[p[k]]) for k in range(n)) for p in itertools.permutations(range(n)))
    ra = [u for w, _ in sorted([x for slot in ref["words"] for x in slot], key=lambda x: x[1]) for u in units(w, lang)]
    ha = [u for _, w, _ in sorted(hw, key=lambda x: x[2]) for u in units(w, lang)]
    return dict(n_ref=len(R[0]) + len(R[1]), n_hyp=sum(len(h) for h in H), cp=cp, agn=ed(ra, ha), n_hyp_spk=len(H), masked=ref["masked_chunks"] > 0)


def score(a):
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out, "predictions-rank*.jsonl"))) for l in open(p, encoding="utf-8")]
    cmp = {}
    for x in (a.compare or "").split(","):
        if x: name, p = x.split("=", 1); cmp[name] = [json.loads(l) for f in glob.glob(os.path.join(p, "predictions-rank*.jsonl")) for l in open(f, encoding="utf-8")]
    res = {}
    for delta in a.deltas:
        acc = collections.defaultdict(collections.Counter); ids = collections.defaultdict(set)
        for r in rows:
            s = score_row(r, delta)
            for g in ("all", "nomask" if not s["masked"] else None):
                if g is None: continue
                c = acc[(r["corpus"], g)]; c["n_ref"] += s["n_ref"]; c["n_hyp"] += s["n_hyp"]; c["cp"] += s["cp"]; c["agn"] += s["agn"]; c["win"] += 1; c["spk>2"] += s["n_hyp_spk"] > 2
                ids[(r["corpus"], g)].add(r["id"])
        for (c, g), v in sorted(acc.items()):
            e = dict(cp=round(100 * v["cp"] / max(1, v["n_ref"]), 2), agn=round(100 * v["agn"] / max(1, v["n_ref"]), 2), hyp_ref=round(v["n_hyp"] / max(1, v["n_ref"]), 3),
                     windows=v["win"], hyp_spk_gt2=v["spk>2"])
            for name, crow in cmp.items():                                      # 같은 창의 우리 모델(Stage 3 회귀 평가 원출력의 점수)
                cc = [x for x in crow if x["delta"] == delta and x["id"] in ids[(c, g)]]; nr = sum(x["n_ref"] for x in cc)
                if cc: e[name] = dict(cp=round(100 * sum(x["cp"] for x in cc) / max(1, nr), 2), agn=round(100 * sum(x["agn"] for x in cc) / max(1, nr), 2), windows=len(cc))
            res[f"{c}/d{delta}/{g}"] = e
    json.dump(res, open(os.path.join(a.out, "score.json"), "w"), ensure_ascii=False, indent=1)
    for k, e in res.items():
        print(f"{k:28s} VibeVoice cp {e['cp']:6.2f} agn {e['agn']:6.2f} h/r {e['hyp_ref']:.2f} 창 {e['windows']:3d} (가설 화자 >2: {e['hyp_spk_gt2']})"
              + "".join(f" | {n} cp {e[n]['cp']:6.2f} agn {e[n]['agn']:6.2f}" for n in cmp if n in e))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--mode", choices=["stream", "offline"], required=True); q.add_argument("--model", required=True); q.add_argument("--src", required=True)
    q.add_argument("--lm-tokenizer", default=None, help="offline: Qwen2.5-7B tokenizer 폴더(체크포인트에 tokenizer 가 없다)")
    q.add_argument("--checkpoint", required=True, help="참조 단어를 만들 우리 tokenizer"); q.add_argument("--data", required=True); q.add_argument("--split", required=True)
    q.add_argument("--out", required=True); q.add_argument("--corpora", default="aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc")
    q.add_argument("--limit-windows", type=int, default=200); q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 10])
    q.add_argument("--backchannel", default=(1.0, 3), type=lambda s: (float(s.split(",")[0]), int(s.split(",")[1])))
    q.add_argument("--batch-size", type=int, default=8, help="offline 만"); q.add_argument("--max-new-tokens", type=int, default=None, help="기본: offline 2048, stream 청크당 256(데모 기본)")
    q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1)
    q = sub.add_parser("score"); q.add_argument("--out", required=True); q.add_argument("--compare", default=None)
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 10])
    a = ap.parse_args()
    if a.cmd == "run":
        if a.mode == "offline": assert a.lm_tokenizer, "--lm-tokenizer 가 필요하다"
        if a.max_new_tokens is None: a.max_new_tokens = 2048 if a.mode == "offline" else 256
    {"run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
