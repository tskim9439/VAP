"""V2-a 두 스트림 입력 점검(mxc 호스트 GPU 하나, 2026-10-20) — 학습 전에 실제 Nemotron·V1 체크포인트로 세 가지를 본다.
  1. 탭 스트림 = 인코더 L 층 출력: model.encode(...)[1] 이 enc.layers[L] forward hook 출력(같은 프레임 자르기)과 같다.
  2. 0 초기화 = V1: add_enc_tap 직후 LLM 입력(chunk_embed)이 V1 과 같고, held-out 대화 창 배치 디코드의 방출 토큰이 V1 과 같다(δ10).
  3. 인과성: 창 앞부분(0.32 s 배수에서 자른 오디오)만 넣은 인코딩의 두 스트림이 전체 오디오 인코딩의 같은 프레임과 같다(탭 가지로 미래 문맥이 새지 않는다).
     판정은 상대 차이(최대 차이 ÷ 그 스트림 값의 최대 크기) < 1e-4 — 누설이면 자른 끝 근처에서 O(1) 로 커진다. 최대 차이 프레임과 끝 4 프레임 값도 남긴다.
  python v2a_check.py --ckpt <V1 체크포인트> --data <phase2> --split <v2.json> --out <json> [--layer 0] [--windows 4]
아무것도 지우지 않는다."""
import argparse, json, os, sys
ROOT = "/soundai/users/tskim/VAPKT"; sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--layer", type=int, default=0); ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--corpora", default="aihub134-2,otoSpeech,kec"); ap.add_argument("--delta", type=int, default=10)
    a = ap.parse_args()
    import numpy as np, torch
    from vapasr.hf.infer import load_model
    from vapasr.hf.batch_decode import decode_batch
    from cst_stage3_eval import windows_of
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"; model, tok = load_model(a.ckpt, encoder_path=os.environ.get("MXC_NEMOTRON_DIR"), dtype=torch.float32); model.eval()
    assert model.adapter_tap is None, "V1 체크포인트(탭 없음)를 준다"
    sets, items = windows_of(argparse.Namespace(data=a.data, split=a.split, corpora=a.corpora, deltas=[a.delta], limit_windows=a.windows, backchannel=(1.0, 3)), tok)
    W = []
    for c, i in items:
        ds = sets[c]; cid, t0, L = ds.items[i]; K = int(round(L / 0.08))
        wav = np.asarray(ds.mono(cid)[int(round(t0 * 16000)): int(round((t0 + L) * 16000))], dtype=np.float32)
        W.append(dict(id=f"{cid}@{t0}", w=torch.from_numpy(wav).to(dev), K=K, prefix=ds.cst_prefix(ds.dlgs[cid].lang, a.delta)))
    enc = lambda w, K: model.encode(w[None], torch.tensor([len(w)], device=dev), torch.tensor([K], device=dev))[0]
    def dec(feats):                                                                           # decode_batch 는 한 배치에 prefix 하나(언어·δ)만 받는다 → prefix 별로 나눠 돌린다
        out = [None] * len(W)
        for pre in dict.fromkeys(tuple(x["prefix"]) for x in W):
            js = [j for j, x in enumerate(W) if tuple(x["prefix"]) == pre]
            for j, s in zip(js, decode_batch(model, [feats[j] for j in js], [list(pre)] * len(js))): out[j] = s
        return out
    res = dict(ckpt=a.ckpt, layer=a.layer, windows=len(W), delta=a.delta)
    with torch.inference_mode():
        f1 = [enc(x["w"], x["K"]) for x in W]; ce1 = [model.chunk_embed(f[None])[0] for f in f1]
        s1 = dec(f1)
        model.add_enc_tap(a.layer); assert model.adapter_tap.net[-1].weight.abs().sum() == 0
        got = {}; h = model.encoder.enc.layers[a.layer].register_forward_hook(lambda m, i_, o: got.__setitem__("x", o[0] if isinstance(o, tuple) else o))
        f2, d_tap, d_main = [], 0.0, 0.0
        for x, f in zip(W, f1):
            y = enc(x["w"], x["K"]); f2.append(y); z = got["x"][0].float(); n = min(x["K"], z.shape[0])
            assert y.shape == (2, x["K"], f.shape[-1]), y.shape
            d_tap = max(d_tap, float((y[1, :n] - z[:n]).abs().max())); d_main = max(d_main, float((y[0] - f[0]).abs().max()))
        h.remove()
        res.update(shape=list(f2[0].shape), tap_vs_hook_maxabs=d_tap, main_vs_v1_maxabs=d_main)
        res["ce_vs_v1_maxabs"] = max(float((model.chunk_embed(f[None])[0] - c).abs().max()) for f, c in zip(f2, ce1))
        s2 = dec(f2)
        res["decode_same"] = sum(p.emitted == q.emitted and p.forced == q.forced for p, q in zip(s1, s2)); res["decode_tokens"] = sum(len(p.emitted) for p in s1)
        ser, forced, _, _ = model.stream_decode(f2[0], W[0]["prefix"]); res["serial_vs_batch_same"] = ser == s2[0].emitted and forced == s2[0].forced
        cz = []
        for x, y in zip(W, f2):                                                               # 3. 인과성
            Kc = (x["K"] // 2) // 4 * 4; yc = enc(x["w"][: Kc * 1280], Kc); r = dict(id=x["id"], K=x["K"], Kc=Kc)
            for s, nm in ((0, "main"), (1, "tap")):
                d = (yc[s, :Kc] - y[s, :Kc]).abs().max(-1).values; scale = float(y[s, :Kc].abs().max())                   # 프레임별 최대 차이, 그 스트림 값의 최대 크기
                r.update({f"{nm}_all": float(d.max()), f"{nm}_but_last4": float(d[: Kc - 4].max()), f"{nm}_last4": float(d[Kc - 4:].max()),
                          f"{nm}_rel": float(d.max()) / max(scale, 1e-12), f"{nm}_argmax_frame": int(d.argmax()), f"{nm}_scale": scale})
            cz.append(r)
        res["causal"] = cz
    ok = (d_tap < 1e-4 and res["ce_vs_v1_maxabs"] < 1e-4 and res["decode_same"] == len(W) and res["serial_vs_batch_same"]
          and all(c["tap_rel"] < 1e-4 and c["main_rel"] < 1e-4 for c in cz))                                      # 인과성: 상대 차이가 fp32 수치 잡음 수준(누설이면 O(1))
    res["ok"] = bool(ok); json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1)); print("V2A_CHECK", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
