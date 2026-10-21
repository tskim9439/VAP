# V3 진단(mxc 호스트 GPU): cst_stage4_taxi.run 을 그대로 돌리되 (1) 창마다 <EOT> 방출(슬롯 언어·세션 시각)을 eot-d{δ}.jsonl 로 남기고
#   (2) --tag-bias B 면 슬롯 태그(<SPK_A>·<SPK_B>) logit 에 B 를 더한다(lm_head 출력 hook — 놓친 교대가 보정 문제인지 보는 진단, TAXI 로 값을 고르지 않는다).
#   python taxi_v3_probe.py --checkpoint <ck> --encoder <nemotron> --sessions <taxi-natural/sessions.jsonl> --out <dir> [--tag-bias 0] [--deltas 10] [--max-window 90]
#   → <out>/hyp·asr-d*.jsonl(TAXI 전사 — mxc 에만), eot-d*.jsonl(세션·슬롯 언어·시각, 글자 없음), probe.json. 채점은 cst_stage4_taxi.py score --out <dir>.
#   (3) --classify-onsets(기본 끔 — 끄면 산출 그대로): 창별 원출력 [(k, tid)] 을 해석 래퍼에서 보존했다가 디코드 뒤 참조 화자 교대마다 온셋을 나눈다
#   (cst_attr_turns.onset_classes — 같은 정렬 W 5 s·시스템 오프셋, OK·RESUME·REASSERT·SILENT·ABSORBED × 이전 슬롯 <EOT> 있음/없음 × 새 화자 언어)
#   → onset-classes-d{δ}.json(집계만, 세션 id·글자 없음).
#   V3 R4(config xl_slot_order 'arrival'): 슬롯 언어가 창마다 다르므로 run 의 map_slots 결과를 창 순서대로 받아 eot·온셋 분류에 쓴다.
import argparse, collections, json, os, sys
CODE = os.environ.get("CODE", "/soundai/users/tskim/VAPKT"); sys.path.insert(0, os.path.join(CODE, "experiments")); sys.path.insert(0, CODE)
import torch
import cst_stage4_taxi as T
import vapasr.data.cst_eval as E
import vapasr.hf.infer as I


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--encoder", default=None); ap.add_argument("--sessions", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10])
    ap.add_argument("--max-window", type=float, default=90.0); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--tag-bias", type=float, default=0.0)
    ap.add_argument("--classify-onsets", action="store_true", help="교대마다 온셋 분류 → onset-classes-d{δ}.json(집계만, 기본 끔)"); a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.sessions, encoding="utf-8")]
    jobs = [(r, w) for r in recs for w in T.windows_of(r, a.max_window)]
    order = [(r, w, d) for b0 in range(0, len(jobs), a.batch_size) for d in a.deltas for r, w in jobs[b0:b0 + a.batch_size]]   # run() 의 해석 호출 순서
    calls = []; tags = collections.Counter(); kept = []; ctx = {}; maps = []                          # kept·ctx: --classify-onsets 때만 채운다(창별 원출력·tokenizer·특수 토큰)
    parse0, load0, map0 = E.parse_emitted4, I.load_model, T.map_slots

    def parse(emitted, spk_ids, trans_id, eot_id, **kw):
        src, tr, eots = parse0(emitted, spk_ids, trans_id, eot_id, **kw)
        calls.append(eots); tags["tag_tokens"] += sum(1 for _, t in emitted if int(t) in spk_ids)
        if a.classify_onsets: kept.append([(int(k), int(t)) for k, t in emitted]); ctx.update(spk=tuple(int(x) for x in spk_ids), trans=int(trans_id), eot=int(eot_id))
        return src, tr, eots

    def load(*x, **kw):
        model, tok = load0(*x, **kw)
        if a.classify_onsets: ctx["tok"] = tok
        if a.tag_bias:
            ids = [int(tok.convert_tokens_to_ids(t)) for t in (getattr(model.config, "slot_tokens", None) or ["<SPK_A>", "<SPK_B>"])][:2]
            def hook(_m, _i, o):
                o[..., ids] += a.tag_bias
                return o
            model.thinker.lm_head.register_forward_hook(hook)
        return model, tok

    def map_slots(*x):                                                                            # R4: 창별 슬롯 언어(해석 호출마다 한 번 — 순서 같음)
        m = map0(*x); maps.append(m)
        return m

    E.parse_emitted4, I.load_model, T.map_slots = parse, load, map_slots
    T.run(argparse.Namespace(checkpoint=a.checkpoint, encoder=a.encoder, sessions=a.sessions, out=a.out, deltas=a.deltas, max_window=a.max_window,
                             batch_size=a.batch_size, max_flush=None, dtype="float32", asr_only=False))
    assert len(calls) == len(order), f"해석 호출 {len(calls)} 대 창×δ {len(order)}"
    assert not maps or len(maps) == len(order), f"슬롯 짝 {len(maps)} 대 창×δ {len(order)}"
    wl = maps or [T.slots_of(r)[1] for r, _, _ in order]                                            # 창×δ 마다 슬롯 언어
    fo = {d: open(os.path.join(a.out, f"eot-d{d}.jsonl"), "w", encoding="utf-8") for d in a.deltas}; n = collections.Counter()
    for (r, (w0, _), d), eots, langs in zip(order, calls, wl):
        for k, tt in eots:
            if k < len(langs): fo[d].write(json.dumps(dict(session=r["session_id"], lang=langs[k], t=round(w0 + tt, 3))) + "\n"); n[f"d{d}"] += 1
    for f in fo.values(): f.close()
    json.dump(dict(checkpoint=a.checkpoint, tag_bias=a.tag_bias, deltas=a.deltas, max_window=a.max_window, windows=len(jobs), eot=dict(n), **tags, **({"slot_map": "arrival"} if maps else {})),
              open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    print("PROBE", json.dumps(dict(eot=dict(n), **tags)), flush=True)
    if a.classify_onsets:
        import cst_attr_turns as A
        runs = {d: collections.defaultdict(list) for d in a.deltas}
        for (r, (w0, _), d), em, langs in zip(order, kept, wl): runs[d][r["session_id"]].append((w0, em, langs) if maps else (w0, em))
        for d in a.deltas:
            oc = A.onset_classes(recs, runs[d], ctx["tok"], ctx["spk"], ctx["trans"], ctx["eot"]); oc.update(delta=d, checkpoint=a.checkpoint, tag_bias=a.tag_bias)
            json.dump(oc, open(os.path.join(a.out, f"onset-classes-d{d}.json"), "w"), indent=1, ensure_ascii=False)
            print("ONSET", d, json.dumps({c: v["all"] for c, v in oc["classes"].items()}), f"교대 {oc['changes']} · noalign {oc['noalign']} · cross_window {oc['cross_window']}", flush=True)


if __name__ == "__main__":
    main()
