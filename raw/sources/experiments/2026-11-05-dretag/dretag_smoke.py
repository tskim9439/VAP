#!/usr/bin/env python
"""디코드 때 재귀속 스모크(2026-11-05, 서버 호스트 GPU 하나, 제출 없음): E13 final(config 전문가 N3D)로 AMI eval 회의 하나의 앞 S 초를
cst_stage3_eval 회의 통째와 같은 경로(encode_long · long_block_decode, 문맥 30 s, δ10, 선언 4)로 재귀속 없이·있이(BlockRetag 3,1) 디코드해 견준다.
  셈: 출력 토큰 수, 재귀속 셈(boxes·moved·tokens·rounds), 디코드 초, 활동 어김(follow_stats off_other — 재귀속 뒤 0 이어야 한다),
  사후 재귀속(follow_stats moved)과 디코드 때 재귀속의 슬롯별 단어가 얼마나 같은가(첫 재귀속 라운드까지는 같아야 하고, 그 뒤는 문맥이 달라 갈라질 수 있다).
  python dretag_smoke.py --checkpoint <E13 final> --encoder <nemotron> --data <ami-bench-v1> [--cid ami-sdm:ES2004b] [--seconds 180]
읽기만 한다(결과는 표준 출력)."""
import argparse, json, os, sys, time

REPO = "/soundai/users/tskim/VAPKT"
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--encoder", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--cid", default="ami-sdm:ES2004b"); ap.add_argument("--seconds", type=float, default=180.0); ap.add_argument("--delta", type=int, default=10)
    a = ap.parse_args()
    import numpy as np, torch
    from vapasr.hf.infer import load_model
    from vapasr.hf.long_session import encode_long, long_block_decode
    from vapasr.hf.batch_decode import BlockRetag
    from vapasr.data.cst_eval import parse_emitted, follow_stats
    from cst_stage3_eval import windows_of, words_of
    model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=torch.float32)
    c = a.cid.split(":")[0]
    ns = argparse.Namespace(data=a.data, split=None, corpora=c, deltas=[a.delta], limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0, max_speakers=4,
                            window_draws=0, declare=4, xl_order="lang", subset="eval")
    sets, items = windows_of(ns, tok); ds = sets[c]; i = next(i for _, i in items if ds.items[i][0] == a.cid)
    langs = ds.slots(i)[1][:4]; prefix = ds.cst_prefix(langs, a.delta); D = len(langs)
    wav = np.asarray(ds.mono(a.cid)[: int(a.seconds * 16000)], dtype=np.float32); K = int(round(len(wav) / 1280))
    with torch.inference_mode():
        t0 = time.time(); feats = encode_long(model, wav, K); enc_s = time.time() - t0
        A = model.block_act(feats); Ab = (A.float() > 0.5).cpu().numpy()
        spk = tuple(model.config.sp_ids[t] for t in model.config.slot_tokens); EOT = model.config.sp_ids["<EOT>"]; BG = model.config.sp_ids.get("<ONSET>")
        res = {}
        for name, rt in (("base", None), ("retag", BlockRetag(A, a.delta, 3, 1))):
            t0 = time.time(); em, fc, rd, used, nf, tk = long_block_decode(model, feats, prefix, D, context_s=30.0, retag=rt); dec = time.time() - t0
            hs, _ = parse_emitted(em, spk, EOT, begin_id=BG); hw = [words_of(tok, h) for h in hs]
            st, moved = follow_stats(hw, Ab, a.delta)
            res[name] = dict(tokens=len(em), forced=fc, forwards=nf, dec_s=round(dec, 1), chunk_ms=round(float(np.mean(tk)), 1), words=sum(map(len, hw)),
                             off_other=st["off_other"], on=st["on"], off_none=st["off_none"], retag=dict(rt.stats) if rt else None, hw=hw, moved=moved)
            print(f"{name}: 토큰 {len(em)} · 단어 {res[name]['words']} · 어김(다른 슬롯) {st['off_other']} · on {st['on']} · 무음 {st['off_none']} · 디코드 {dec:.1f} s"
                  f"(청크 {res[name]['chunk_ms']} ms, forward {nf}) · 재귀속 {res[name]['retag']}", flush=True)
    b, r = res["base"], res["retag"]
    same = lambda x, y: sum(1 for s in range(min(len(x), len(y))) for w in set(map(tuple, x[s])) & set(map(tuple, y[s])))
    print(f"인코딩 {enc_s:.1f} s · K {K} · 선언 {D} · 회의 {a.cid} 앞 {a.seconds:.0f} s", flush=True)
    print(f"사후 재귀속(base 를 옮김) 대 디코드 때 재귀속: 슬롯·시각까지 같은 단어 {same(b['moved'], r['hw'])} / 사후 {sum(map(len, b['moved']))} · 디코드 때 {r['words']}", flush=True)
    print("SMOKE_OK" if r["off_other"] <= 0.01 * max(1, r["words"]) and r["retag"]["moved"] > 0 else "SMOKE_CHECK", flush=True)   # 여러 라운드에 걸친 단어는 마지막 토큰 라운드로 세므로 조금 남을 수 있다


if __name__ == "__main__":
    main()
