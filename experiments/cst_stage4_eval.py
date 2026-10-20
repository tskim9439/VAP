#!/usr/bin/env python
"""Stage 4 평가(S4) — 2 인 대화 창을 번역 모드 prefix(translate 줄)로 스트리밍 디코드해 받아쓰기·화자 귀속·턴 끝 <EOT>·번역을 잰다
(wiki decision-stage4-prep-20261019, output-stage4-prep-plan-20261019 S4).

  python experiments/cst_stage4_eval.py run --checkpoint <ck> --data <dir> [--split <phase2>/splits/v2.json] --trans <번역 타깃>[,…] --out <dir> \\
      [--corpora …] [--deltas 4,6,10] [--limit-windows 200] [--tgts English,German,…] [--rank r --world-size n] [--oracle | --from-predictions DIR]
  python experiments/cst_stage4_eval.py summarize --out <dir>

창: --split 이 있으면 코퍼스마다 held-out 대화의 창(cst_stage3_eval.windows_of 와 같은 규칙·표본), 없으면 전부(held-out 재료로 따로 만든 합성 평가 셋).
목표: 같은 언어 대화는 CstDialogueDataset.tgt_of 의 목표 언어마다(영어 → 독·한), 교차 언어 대화(meta.spk_lang)는 'xl'(각 슬롯이 상대 언어). --tgts 로 거른다.
  슬롯 순서는 학습과 같다(같은 언어 = 먼저 말한 사람이 A, 교차 언어 = 창 id 해시).
참조 = 그 δ·목표의 학습 타깃(cst4_sequence)에서 전사 없는 음성 청크(mask)에 놓인 원소를 뺀 것. 전사·<EOT> 는 Stage 3 평가와 같고 번역은 흐름 2·3 의 토큰이다.
  가설도 mask 청크의 출력을 버린다. 번역은 참조 번역이 없는 행(판정 탈락·창 시작 전에 시작한 행)의 방출 청크(miss)에서 나온 것도 버린다.
지표:
  받아쓰기·귀속 — 같은 언어: cst_eval.score_window(slot·cp·agn·swap), 교차 언어: 슬롯이 언어로 고정이라 슬롯별 오류(src_err: 언어·편집 거리·참조 단위)
  <EOT> — Stage 3 와 같다(cst_eval.match_eot)
  번역 — 창·슬롯 하나가 한 분절이다. (원문 언어 → 목표 언어)별 코퍼스 BLEU·chrF(ast_eval.corpus_scores — 목표가 한국어면 BLEU 는 글자),
    지연 = 같은 비용 정렬로 일치한 단어의 (가설 − 학습 타깃) 방출 시각 중앙값(cst_eval.lag_matched), 빈 번역 비율.
    같은 언어 창은 가설 슬롯을 받아쓰기 cp 순열로 참조 슬롯에 맞춘다.
--oracle: 모델 없이 정답 시퀀스를 채점한다(번역 BLEU·chrF 100, 지연 0, 귀속 오류 0 이어야 한다).
출력: predictions-rank{r}.jsonl(창·δ·목표마다 원출력과 점수), summary.json."""
import argparse, collections, hashlib, json, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cst_stage3_eval import SEEN, words_of, win_kw, win_conf  # noqa: E402


def windows_of(a, tok, TR):
    """코퍼스별 평가 창 → (데이터셋, [(코퍼스, 창 index)])."""
    import numpy as np
    from vapasr.data.cst_dialogue import CstDialogueDataset
    split = json.load(open(a.split, encoding="utf-8")) if a.split else None; out = []; sets = {}
    for c in a.corpora.split(","):
        held = set(split["corpora"].get(c, {}).get("heldout", [])) if split else None
        if split is not None and not held: print(f"  {c}: held-out 없음 — 건너뜀", flush=True); continue
        ds = CstDialogueDataset([os.path.join(a.data, f"{c}.refined.dialogues.jsonl")], tok, delays=tuple(a.deltas), mono_cache_dir=os.path.join(a.data, "mono"), seed=0,
                                backchannel=a.backchannel, trans=TR, ast_frac=0.0, trans_lead_space=a.trans_lead_space, **win_kw(a))
        idx = [i for i, it in enumerate(ds.items) if held is None or it[0] in held]
        idx.sort(key=lambda i: (ds.items[i][2], ds.items[i][0], ds.items[i][1]))
        if a.limit_windows and len(idx) > a.limit_windows: idx = [idx[j] for j in np.linspace(0, len(idx) - 1, a.limit_windows, dtype=int)]
        sets[c] = ds; out += [(c, i) for i in idx]; print(f"  {c}: {'held-out 대화 ' + str(len(held)) if held else '전부'} · 창 {len(idx)}", flush=True)
    return sets, out


def window_ref4(ds, q, mask, tok):
    """번역 모드 학습 타깃 q 의 참조 → (슬롯별 전사 [(단어, 시각)], 슬롯별 번역 [(단어, 방출 시각)], [(슬롯, 끝 시각, p_end)] EOT).
    전사·EOT 시각은 원래 시각(창 기준), 번역 시각은 타깃의 방출 청크 끝 (k+1)·80 ms(flush = K) — 가설 방출 시각과 같은 규약.
    번역은 참조 번역이 없는 행의 방출 청크(miss)에 놓인 것도 뺀다 — 가설 쪽(parse_emitted4 drop_tr_chunks)과 같은 규칙."""
    K = q["K"]; n = len(q["streams"]) // 2; src, tr, eot = [[] for _ in range(n)], [[] for _ in range(n)], []; miss = q["miss_chunks"]   # 슬롯 n(2 화자면 2)
    for pos, sa in enumerate(q["src_at"]):
        if sa is None: continue
        k = q["chunk_at"][pos]
        if 0 <= k < K and k in mask: continue
        x = q["streams"][sa[0]][sa[1]]
        if len(x) == 3: eot.append((sa[0] % n, x[1], q["eot_p"].get((sa[0], sa[1]))))
        elif sa[0] < n: src[sa[0]].append((x[1], x[0]))
        elif k not in miss: tr[sa[0] - n].append(((k + 1) * 0.08, x[0]))
    return [words_of(tok, x) for x in src], [words_of(tok, x) for x in tr], eot


def score4(ds, q, ref_w, ref_tr, ref_eot, hyp_src, hyp_tr, hyp_eot, delta, words):
    """한 창의 점수 → dict(받아쓰기·귀속, eot, src_err, tr[참조 슬롯마다 원문·목표 언어·참조·가설·지연])."""
    from vapasr.data.cst_eval import score_window, match_eot, eot_tol, ed, units, lag_matched
    langs = q["langs"]; hw = [words(h) for h in hyp_src]; n = max(len(ref_w), len(hw)); ref_w = list(ref_w) + [[]] * (n - len(ref_w)); ref_tr = list(ref_tr) + [[]] * (n - len(ref_tr))
    hw += [[]] * (n - len(hw)); hyp_tr = list(hyp_tr) + [[]] * (n - len(hyp_tr))                   # 슬롯 수를 맞춘다(가설 = 모델 슬롯 수, 참조 = 창 화자 수)
    if len(set(langs)) == 1: sc = score_window(ref_w, hw, langs[0])
    else:                                                                                      # 교차 언어: 슬롯이 언어로 고정 — 순열 없음
        e = [ed([u for w, _ in ref_w[k] for u in units(w, langs[k])], [u for w, _ in hw[k] for u in units(w, langs[k])]) for k in range(2)]
        n = [sum(len(units(w, langs[k])) for w, _ in ref_w[k]) for k in range(2)]
        sc = dict(n_ref=sum(n), n_hyp=sum(sum(len(units(w, langs[k])) for w, _ in hw[k]) for k in range(2)), slot=sum(e), cp=sum(e), agn=sum(e), swap=False, perm=[0, 1])
    perm = sc["perm"]; inv = {perm[h]: h for h in range(len(perm))}; R = [r for r in range(len(perm)) if ref_w[r] or ref_tr[r] or r < 2]   # 참조 슬롯(빈 슬롯은 A·B 만 남긴다 — V1 행 형식)
    lg = lambda r: langs[r] if r < len(langs) else langs[0]; tg = lambda r: q["tgt"][r] if r < len(q["tgt"]) else q["tgt"][0]
    sc["src_err"] = [(lg(r), ed([u for w, _ in ref_w[r] for u in units(w, lg(r))], [u for w, _ in hw[inv[r]] for u in units(w, lg(r))]),
                      sum(len(units(w, lg(r))) for w, _ in ref_w[r])) for r in R]
    sc["eot"] = match_eot(ref_eot, hyp_eot, perm, eot_tol(delta))
    sc["tr"] = []
    for r in R:
        h = words(hyp_tr[inv[r]])
        sc["tr"].append(dict(src_lang=lg(r), tgt_lang=tg(r), ref=" ".join(w for w, _ in ref_tr[r]), hyp=" ".join(w for w, _ in h), n_ref=len(ref_tr[r]),
                             n_hyp=len(h), lag=lag_matched(ref_tr[r], h)))
    return sc


def run(a):
    import numpy as np, torch
    from types import SimpleNamespace
    from vapasr.data.cst_eval import parse_emitted4, emitted_from_sequence
    from vapasr.data.ast_dataset import load_ast_targets
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    prev = {}
    if a.from_predictions:
        import glob
        for f in glob.glob(os.path.join(a.from_predictions, "predictions-rank*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); prev[(r["id"], r["delta"], r["tgt_opt"])] = r
        print(f"  원출력 {len(prev)} 행 ← {a.from_predictions}", flush=True)
    if a.oracle or a.from_predictions:
        from transformers import AutoTokenizer
        model = None; tok = AutoTokenizer.from_pretrained(a.checkpoint); cfg = json.load(open(os.path.join(a.checkpoint, "config.json"))); sp = cfg["sp_ids"]; blocked = cfg["blocked_ids"]
    else:
        from vapasr.hf.infer import load_model
        from vapasr.hf.batch_decode import decode_batch
        model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=getattr(torch, a.dtype)); sp = model.config.sp_ids; blocked = model.config.blocked_ids
    from vapasr.data.cst_dialogue import SLOTS
    n_slots = max(2, int(getattr(a, "max_speakers", None) or 2))
    if model is not None and len(model.config.slot_tokens) < n_slots:                      # V1(2 슬롯) 기준선: 모델 슬롯 수로 디코드·채점
        print(f"  모델 슬롯 {model.config.slot_tokens} < 창 화자 상한 {n_slots} — 모델 슬롯 수로 디코드·채점", flush=True); n_slots = len(model.config.slot_tokens)
    spk = tuple(sp[t] for t in SLOTS[:n_slots]); EOT = sp["<EOT>"]; TRANS = int(tok.convert_tokens_to_ids("<TRANS>"))
    if not a.oracle and (EOT in blocked or TRANS in blocked): print(f"!! decode 차단: <EOT> {EOT in blocked} · <TRANS> {TRANS in blocked}", flush=True)
    TR, tr_st, tr_fp = load_ast_targets([p for x in a.trans for p in x.split(",") if p]); print(f"  번역 타깃 {dict(tr_st)}", flush=True)
    sets, items = windows_of(a, tok, TR)
    items = [x for j, x in enumerate(items) if j % a.world_size == a.rank]
    keep_tgt = set(a.tgts.split(",")) if a.tgts else None
    out = a.out; os.makedirs(out, exist_ok=True)
    conf = dict(checkpoint=os.path.abspath(a.checkpoint), config_sha256=hashlib.sha256(open(os.path.join(a.checkpoint, "config.json"), "rb").read()).hexdigest(),
                split_sha256=hashlib.sha256(open(a.split, "rb").read()).hexdigest() if a.split else None, trans_sha256=tr_fp, corpora=a.corpora, deltas=a.deltas,
                limit_windows=a.limit_windows, tgts=a.tgts, dtype=a.dtype, world_size=a.world_size, oracle=bool(a.oracle), from_predictions=a.from_predictions,
                **({"backchannel": list(a.backchannel)} if a.backchannel else {}), **({"trans_lead_space": True} if a.trans_lead_space else {}), **win_conf(a))
    cp = os.path.join(out, f"config-rank{a.rank}.json")
    if os.path.exists(cp): assert json.load(open(cp)) == conf, "재개 설정 불일치 — 새 --out 을 쓴다"
    else: json.dump(conf, open(cp, "w"), indent=1)
    dest = os.path.join(out, f"predictions-rank{a.rank}.jsonl")
    done = {(r["id"], r["delta"], r["tgt_opt"]) for r in map(json.loads, open(dest))} if os.path.exists(dest) else set()
    dev = next(model.parameters()).device if model is not None else None; t0 = time.time(); n = 0
    words = lambda toks: words_of(tok, toks)
    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for c in dict.fromkeys(x[0] for x in items):
            ds = sets[c]; todo = [i for cc, i in items if cc == c]
            for b0 in range(0, len(todo), a.batch_size):
                batch = todo[b0:b0 + a.batch_size]; feats = []; meta = []
                for i in batch:
                    cid, t0w, L = ds.items[i]; K = int(round(L / 0.08)); meta.append((i, cid, t0w, L, K))
                    if a.oracle or a.from_predictions: continue
                    wav = np.asarray(ds.mono(cid)[int(round(t0w * 16000)): int(round((t0w + L) * 16000))], dtype=np.float32); w = torch.from_numpy(wav).to(dev)
                    feats.append(model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0])
                opts = [o for o in ds.tgt_options(batch[0]) if keep_tgt is None or o in keep_tgt]
                for opt in opts:
                    for delta in a.deltas:
                        if all((f"{cid}@{t0w}", delta, opt) in done for _, cid, t0w, *_ in meta): continue
                        qs = [ds.cst4_sequence(m[0], delta, opt) for m in meta]
                        prefixes = [ds.cst_prefix(q["langs"][:n_slots], delta, list(q["tgt"])[:n_slots]) for q in qs]   # 모델 슬롯 수까지 선언
                        if a.from_predictions:
                            keys = [(f"{m[1]}@{m[2]}", delta, opt) for m in meta]
                            if not all(k in prev for k in keys): print(f"  !! 원출력 없음 δ{delta} {opt} — 건너뜀", flush=True); continue
                            states = [SimpleNamespace(emitted=[tuple(e) for e in prev[k]["emitted"]], forced=prev[k]["forced"], flush_rounds=prev[k]["flush_rounds"]) for k in keys]
                        elif a.oracle:
                            states = [SimpleNamespace(emitted=emitted_from_sequence(q["ids"], q["is_input"], q["chunk_of"], sp["<NEXT_AUDIO>"], sp["<EMPTY_AUDIO>"]), forced=0,
                                                      flush_rounds=0) for q in qs]
                        else:
                            states = [None] * len(meta)
                            for pre in dict.fromkeys(tuple(p) for p in prefixes):                      # 선언 슬롯 수가 다른 창은 prefix 별로 나눠 돈다
                                js = [j for j, p in enumerate(prefixes) if tuple(p) == pre]
                                for j, st_ in zip(js, decode_batch(model, [feats[j] for j in js], [list(pre)] * len(js), max_flush_rounds=a.max_flush)): states[j] = st_
                        for (i, cid, t0w, L, K), q, s in zip(meta, qs, states):
                            if (f"{cid}@{t0w}", delta, opt) in done: continue
                            d = ds.dlgs[cid]; mask = ds.mask_chunks(d, t0w, L, K, delta)
                            ref_w, ref_tr, ref_eot = window_ref4(ds, q, mask, tok)
                            hs, ht, he = parse_emitted4(s.emitted, spk, TRANS, EOT, drop_chunks=mask, drop_tr_chunks=q["miss_chunks"])
                            sc = score4(ds, q, ref_w, ref_tr, ref_eot, hs, ht, he, delta, words)
                            row = dict(id=f"{cid}@{t0w}", corpus=c, lang=d.lang, langs=q["langs"], tgt_opt=opt, tgts=list(q["tgt"]), delta=delta, L=L, K=K, speakers=q["speakers"],
                                       n_spk=sum(1 for st in q["streams"][: len(q["streams"]) // 2] if st), masked_chunks=len(mask), miss_chunks=len(q["miss_chunks"]), forced=s.forced,
                                       flush_rounds=s.flush_rounds, emitted=[list(e) for e in s.emitted], ref_text=[" ".join(w for w, _ in r) for r in ref_w],
                                       hyp_text=[" ".join(w for w, _ in words(h)) for h in hs], **sc)
                            fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
                print(f"  {c}: {min(b0 + a.batch_size, len(todo))}/{len(todo)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


def summarize(a):
    import glob
    from vapasr.data.cst_eval import pool
    from vapasr.data.ast_eval import corpus_scores
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out, "predictions-rank*.jsonl"))) for l in open(p, encoding="utf-8")]
    res = {}
    for key in sorted({(r["corpus"], r["delta"]) for r in rows}):
        rr = [r for r in rows if (r["corpus"], r["delta"]) == key]; xl = rr[0]["langs"][0] != rr[0]["langs"][1]
        v = dict(pool(rr), exposure=SEEN.get(key[0], "unseen"), windows=len({r["id"] for r in rr}), rows=len(rr), forced_tokens=sum(r["forced"] for r in rr))
        if xl: v = {k: x for k, x in v.items() if not k.startswith(("err_", "swap"))}           # 교차 언어는 단위가 섞여 합산 오류율을 쓰지 않는다
        if not v["eot_pos"]: v = {k: x for k, x in v.items() if not k.startswith("eot_")}       # <EOT> 참조가 없다(합성 — meta.mask_eot)
        src = collections.defaultdict(lambda: [0, 0])
        for r in rr:
            for lg, e, nn in r["src_err"]: src[lg][0] += e; src[lg][1] += nn
        v["src_err_by_lang"] = {lg: dict(err=round(100 * e / max(1, nn), 2), unit="CER" if lg == "Korean" else "WER", ref_units=nn) for lg, (e, nn) in sorted(src.items())}
        tr = collections.defaultdict(list)
        for r in rr:
            for x in r["tr"]:
                if x["n_ref"]: tr[(x["src_lang"], x["tgt_lang"])].append(x)
        v["translation"] = {}
        for (sl, tl), xs in sorted(tr.items()):
            lags = [y for x in xs for y in x["lag"]]
            v["translation"][f"{sl}->{tl}"] = dict(corpus_scores([x["hyp"] for x in xs], [x["ref"] for x in xs], tl), segments=len(xs),
                                                   empty_share=round(sum(not x["hyp"] for x in xs) / len(xs), 4), hyp_ref_words=round(sum(x["n_hyp"] for x in xs) / max(1, sum(x["n_ref"] for x in xs)), 3),
                                                   lag_median=round(statistics.median(lags), 3) if lags else None, lag_p90=round(sorted(lags)[int(0.9 * (len(lags) - 1))], 3) if lags else None,
                                                   matched_word_share=round(len(lags) / max(1, sum(x["n_ref"] for x in xs)), 3))
        res[f"{key[0]}/d{key[1]}"] = v
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        s = " · ".join(f"{lg} {x['unit']} {x['err']:.2f}" for lg, x in v["src_err_by_lang"].items())
        t = " · ".join(f"{d} BLEU {x['bleu']:.1f} chrF {x['chrf']:.1f} 지연 {x['lag_median']}" for d, x in v["translation"].items())
        print(f"{k:24s} {s}" + (f" cp {v['err_cp']:.2f} swap {v['swap_share']:.3f}" if "err_cp" in v else "") +
              (f" · EOT R {v['eot_recall']:.3f} P {v['eot_precision']:.3f}" if "eot_recall" in v else " · EOT 참조 없음") + f" · {t} · 창 {v['windows']} ({v['exposure']})")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--checkpoint", required=True); q.add_argument("--data", required=True); q.add_argument("--split", default=None)
    q.add_argument("--trans", action="append", required=True, help="번역 타깃(ast_build_targets·sim_build_dialogues 산출, 반복·쉼표)"); q.add_argument("--out", required=True)
    q.add_argument("--corpora", default="aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc"); q.add_argument("--tgts", default=None, help="평가할 목표(예 English,German,xl) — 없으면 전부")
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 6, 10]); q.add_argument("--limit-windows", type=int, default=200)
    q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1); q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--max-flush", type=int, default=None); q.add_argument("--dtype", default="float32"); q.add_argument("--encoder", default=None)
    q.add_argument("--oracle", action="store_true", help="모델 없이 정답 시퀀스를 채점(지표 sanity)"); q.add_argument("--from-predictions", default=None)
    q.add_argument("--backchannel", default=None, type=lambda s: (float(s.split(",")[0]), int(s.split(",")[1])), help="참조 EOT 의 맞장구 규칙(최대초,최대토큰)")
    q.add_argument("--trans-lead-space", action="store_true", help="참조 번역 행마다 앞 공백(학습 --trans-lead-space 와 같은 형식으로 학습한 모델)")
    q.add_argument("--max-speakers", type=int, default=2, help="V2: 창 화자 상한(최대 4)"); q.add_argument("--window", type=lambda s: tuple(float(x) for x in s.split(",")), default=(20.0, 40.0))
    q.add_argument("--hop", type=float, default=10.0); q.add_argument("--window-draws", type=int, default=0); q.add_argument("--declare", type=int, default=None)
    q = sub.add_parser("summarize"); q.add_argument("--out", required=True)
    a = ap.parse_args(); {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
