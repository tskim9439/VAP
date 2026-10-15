#!/usr/bin/env python
"""Stage 3 평가(P4) — held-out 2 인 대화 창을 두 슬롯 prefix 로 스트리밍 디코드해 화자 귀속·받아쓰기·턴 끝 <EOT> 를 잰다.

  python experiments/cst_stage3_eval.py run --checkpoint <final> --data <phase2 dir> --split <phase2>/splits/v2.json --out <dir> \\
      [--corpora aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc] [--deltas 4,6,10] [--limit-windows 400] [--rank r --world-size n] [--verify]
  python experiments/cst_stage3_eval.py summarize --out <dir>

창: 코퍼스마다 split held-out 대화의 20–40 s 무음 시작 격자 창(CstDialogueDataset, seed 0 — 학습 창 규칙과 같다). --limit-windows 는 길이 순 고른 간격 표본.
  71631 held-out 은 Stage 1·2 학습에 노출됐다(P0) — 'seen' 으로 표시한다. 134-1 held-out 은 화자 10 명이 71631 에 나온다(대화는 다름).
입력: mono 캐시 crop(학습과 같은 오디오), 창마다 인코더 한 번(singleton), 디코드는 언어·δ 별 배치(vapasr/hf/batch_decode.decode_batch, 학습과 같은 차단 목록).
  --verify: 언어·δ 마다 첫 배치의 처음·끝 창을 model.stream_decode 로 다시 풀어 토큰·시각이 같은지 확인한다.
  --oracle: 모델 없이 학습 타깃 시퀀스(정답)를 디코드 출력처럼 바꿔 채점한다 — 지표 sanity(slot·cp 0, 양성 EOT 재현율 1).
  --from-predictions DIR: 그 폴더 predictions-rank*.jsonl 의 원출력(emitted)을 다시 채점한다(디코드 없이 — 채점 규칙을 고친 뒤).
참조 = 그 δ 의 학습 타깃 시퀀스(cst_sequence)에서 전사 없는 음성 청크(mask_chunks — 학습 label −100)에 놓인 원소를 뺀 것. 전사 토큰 → 단어(시각 = 마지막
  토큰의 원래 끝 시각), EOT = episode 끝(p_end). 가설도 같은 청크에서 나온 토큰·EOT 를 버린다 — 참조·가설이 같은 규칙(2026-10-15 오라클 점검: 창 전체 토큰을
  참조로 두면 마스크 청크 때문에 오라클도 KO CER 7–15 % 가 나왔다).
지표(vapasr/data/cst_eval): slot·cp(화자 귀속)·agn(화자 무시) 오류율 — 한국어 CER(공백 제외), 영어·독일어 WER. 슬롯 뒤집힘 비율,
  EOT 재현율(양성 = 교대·침묵)·정밀도·보류 오방출·지연. 출력: predictions-rank{r}.jsonl(창·δ 별 원출력과 점수), summary.json."""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEN = {"aihub71631": "seen(Stage 1·2 노출)", "aihub134-1": "partly(화자 10 명 71631 과 겹침)"}


def windows_of(a, tok):
    """코퍼스별 held-out 창 → [(코퍼스, 데이터셋, 창 index)]."""
    import numpy as np
    from vapasr.data.cst_dialogue import CstDialogueDataset
    split = json.load(open(a.split, encoding="utf-8")); out = []; sets = {}
    for c in a.corpora.split(","):
        held = set(split["corpora"].get(c, {}).get("heldout", []))
        if not held: print(f"  {c}: held-out 없음 — 건너뜀", flush=True); continue
        ds = CstDialogueDataset([os.path.join(a.data, f"{c}.refined.dialogues.jsonl")], tok, delays=tuple(a.deltas), mono_cache_dir=os.path.join(a.data, "mono"), seed=0)
        idx = [i for i, it in enumerate(ds.items) if it[0] in held]
        idx.sort(key=lambda i: (ds.items[i][2], ds.items[i][0], ds.items[i][1]))
        if a.limit_windows and len(idx) > a.limit_windows: idx = [idx[j] for j in np.linspace(0, len(idx) - 1, a.limit_windows, dtype=int)]
        sets[c] = ds; out += [(c, i) for i in idx]; print(f"  {c}: held-out 대화 {len(held)} · 창 {len(idx)}", flush=True)
    return sets, out


def run(a):
    import numpy as np, torch
    from vapasr.data.cst_dialogue import window_speakers, window_streams
    from vapasr.data.cst_eval import parse_emitted, score_window, match_eot, eot_tol, emitted_from_sequence
    from vapasr.data.ast_eval import token_bytes
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    prev = {}
    if a.from_predictions:
        import glob
        for f in glob.glob(os.path.join(a.from_predictions, "predictions-rank*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); prev[(r["id"], r["delta"])] = r
        print(f"  원출력 {len(prev)} 행 ← {a.from_predictions}", flush=True)
    if a.oracle or a.from_predictions:
        from transformers import AutoTokenizer
        model = None; tok = AutoTokenizer.from_pretrained(a.checkpoint); cfg = json.load(open(os.path.join(a.checkpoint, "config.json"))); sp = cfg["sp_ids"]; blocked = cfg["blocked_ids"]
    else:
        from vapasr.hf.infer import load_model
        from vapasr.hf.batch_decode import decode_batch
        model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=getattr(torch, a.dtype)); sp = model.config.sp_ids; blocked = model.config.blocked_ids
    spk = (sp["<SPK_A>"], sp["<SPK_B>"]); EOT = sp["<EOT>"]
    if EOT in blocked and not a.oracle: print("!! <EOT> 가 decode 차단 목록에 있다(Stage 3 학습 전 체크포인트) — EOT 는 나오지 않는다", flush=True)
    sets, items = windows_of(a, tok)
    items = [x for j, x in enumerate(items) if j % a.world_size == a.rank]
    out = os.path.join(a.out); os.makedirs(out, exist_ok=True)
    conf = dict(checkpoint=os.path.abspath(a.checkpoint), config_sha256=hashlib.sha256(open(os.path.join(a.checkpoint, "config.json"), "rb").read()).hexdigest(),
                split_sha256=hashlib.sha256(open(a.split, "rb").read()).hexdigest(), corpora=a.corpora, deltas=a.deltas, limit_windows=a.limit_windows, dtype=a.dtype,
                world_size=a.world_size, eot_blocked=EOT in blocked, oracle=bool(a.oracle), from_predictions=a.from_predictions)
    cp = os.path.join(out, f"config-rank{a.rank}.json")
    if os.path.exists(cp): assert json.load(open(cp)) == conf, "재개 설정 불일치 — 새 --out 을 쓴다"
    else: json.dump(conf, open(cp, "w"), indent=1)
    dest = os.path.join(out, f"predictions-rank{a.rank}.jsonl")
    done = {(r["id"], r["delta"]) for r in map(json.loads, open(dest))} if os.path.exists(dest) else set()
    dev = next(model.parameters()).device if model is not None else None; checked = set(); t0 = time.time(); n = 0

    def words(toks):
        """[(시각, tid)] → [(단어, 그 단어 마지막 바이트 토큰의 시각)]."""
        if not toks: return []
        bs = token_bytes(tok, [t for _, t in toks]); raw = b"".join(bs); cum, c = [], 0
        for b in bs: c += len(b); cum.append(c)
        import bisect, re
        return [(m.group().decode("utf-8", "replace"), toks[min(bisect.bisect_left(cum, m.end()), len(toks) - 1)][0]) for m in re.finditer(rb"\S+", raw)]

    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for c in dict.fromkeys(x[0] for x in items):
            ds = sets[c]; todo = [i for cc, i in items if cc == c]
            for b0 in range(0, len(todo), a.batch_size):
                batch = todo[b0:b0 + a.batch_size]; feats = []; meta = []
                for i in batch:
                    cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / 0.08)); meta.append((i, cid, t0w, L, K, d))
                    if a.oracle or a.from_predictions: continue
                    wav = np.asarray(ds.mono(cid)[int(round(t0w * 16000)): int(round((t0w + L) * 16000))], dtype=np.float32)
                    w = torch.from_numpy(wav).to(dev)
                    feats.append(model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0])
                for delta in a.deltas:
                    if all((f"{cid}@{t0w}", delta) in done for _, cid, t0w, *_ in meta): continue
                    prefixes = [ds.cst_prefix(m[5].lang, delta) for m in meta]
                    if a.from_predictions:
                        from types import SimpleNamespace
                        if not all((f"{m[1]}@{m[2]}", delta) in prev for m in meta): print(f"  !! 원출력 없음 δ{delta} — 건너뜀", flush=True); continue
                        states = [SimpleNamespace(emitted=[tuple(e) for e in prev[(f"{m[1]}@{m[2]}", delta)]["emitted"]], forced=prev[(f"{m[1]}@{m[2]}", delta)]["forced"],
                                                  flush_rounds=prev[(f"{m[1]}@{m[2]}", delta)]["flush_rounds"]) for m in meta]
                    elif a.oracle:
                        from types import SimpleNamespace
                        states = []
                        for m in meta:
                            q = ds.cst_sequence(m[0], delta)
                            states.append(SimpleNamespace(emitted=emitted_from_sequence(q["ids"], q["is_input"], q["chunk_of"], sp["<NEXT_AUDIO>"], sp["<EMPTY_AUDIO>"]), forced=0, flush_rounds=0))
                    else:
                        states = decode_batch(model, feats, prefixes, max_flush_rounds=a.max_flush)
                    key = (ds.dlgs[meta[0][1]].lang, delta)
                    if a.verify and model is not None and key not in checked:
                        for j in sorted({0, len(meta) - 1}):
                            serial, forced, _, rounds = model.stream_decode(feats[j], prefixes[j], max_flush_rounds=a.max_flush)
                            assert serial == states[j].emitted and forced == states[j].forced, f"직렬/배치 불일치 {meta[j][1]}@{meta[j][2]} δ{delta}"
                        checked.add(key); print(f"PARITY {key} PASS", flush=True)
                    for (i, cid, t0w, L, K, d), s in zip(meta, states):
                        if (f"{cid}@{t0w}", delta) in done: continue
                        eps = ds.eps[cid]; sp_order = window_speakers(eps, t0w, L)
                        streams, eot_p, wst = window_streams(eps, sp_order, t0w, L, EOT)
                        q = ds.cst_sequence(i, delta); mask = ds.mask_chunks(d, t0w, L, K, delta); ref_tok = [[], []]; ref_eot = []
                        for pos, src in enumerate(q["src_at"]):                      # 참조 = 타깃 시퀀스에서 마스크 청크 밖 원소(학습 label 과 같은 범위)
                            if src is None or (0 <= q["chunk_at"][pos] < K and q["chunk_at"][pos] in mask): continue
                            x = streams[src[0]][src[1]]
                            if len(x) == 3: ref_eot.append((src[0], x[1], eot_p.get((src[0], src[1]))))
                            else: ref_tok[src[0]].append((x[1], x[0]))
                        ref_w = [words(ref_tok[0]), words(ref_tok[1])]
                        hs, he = parse_emitted(s.emitted, spk, EOT, drop_chunks=mask)
                        sc = score_window(ref_w[:2], [words(hs[0]), words(hs[1])], d.lang)
                        sc["eot"] = match_eot(ref_eot, he, sc["perm"], eot_tol(delta))
                        row = dict(id=f"{cid}@{t0w}", corpus=c, lang=d.lang, delta=delta, L=L, K=K, speakers=sp_order, n_spk=len(sp_order), masked_chunks=len(mask),
                                   forced=s.forced, flush_rounds=s.flush_rounds, emitted=[list(e) for e in s.emitted],
                                   ref_text=[" ".join(w for w, _ in r) for r in ref_w[:2]], hyp_text=[" ".join(w for w, _ in words(h)) for h in hs], **sc)
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
                print(f"  {c}: {min(b0 + a.batch_size, len(todo))}/{len(todo)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


def summarize(a):
    import glob
    from vapasr.data.cst_eval import pool
    rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(a.out, "predictions-rank*.jsonl"))) for l in open(p, encoding="utf-8")]
    res = {}
    for c in sorted({r["corpus"] for r in rows}):
        for d in sorted({r["delta"] for r in rows if r["corpus"] == c}):
            rr = [r for r in rows if r["corpus"] == c and r["delta"] == d]
            res[f"{c}/d{d}"] = dict(pool(rr), exposure=SEEN.get(c, "unseen"), unit="CER" if rr[0]["lang"] == "Korean" else "WER",
                                    one_speaker_windows=sum(r["n_spk"] == 1 for r in rr), forced_tokens=sum(r["forced"] for r in rr))
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        print(f"{k:22s} {v['unit']} slot {v['err_slot']:6.2f} cp {v['err_cp']:6.2f} agn {v['err_agn']:6.2f} swap {v['swap_share']:.3f} · "
              f"EOT R {v['eot_recall']:.3f} P {v['eot_precision']:.3f} hold {v['eot_hold_fire']:.3f} lat {v['eot_lat_median']} · 창 {v['windows']} ({v['exposure']})")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--checkpoint", required=True); q.add_argument("--data", required=True); q.add_argument("--split", required=True)
    q.add_argument("--out", required=True); q.add_argument("--corpora", default="aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc")
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[4, 6, 10]); q.add_argument("--limit-windows", type=int, default=400)
    q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1); q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--max-flush", type=int, default=None); q.add_argument("--dtype", default="float32"); q.add_argument("--encoder", default=None); q.add_argument("--verify", action="store_true")
    q.add_argument("--oracle", action="store_true", help="모델 없이 정답 시퀀스를 채점(지표 sanity)")
    q.add_argument("--from-predictions", default=None, help="이 폴더의 원출력을 다시 채점(디코드 없음)")
    q = sub.add_parser("summarize"); q.add_argument("--out", required=True)
    a = ap.parse_args(); {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
