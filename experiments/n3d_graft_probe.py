#!/usr/bin/env python
"""고정 화자 분리 전문가를 우리 인코더 하나로 합치는 길(E11 통합, 2026-10-29 — wiki output-n3d-branch-20261029 4 절)의 첫 판가름: 이식 층 고르기.
Nemotron 3 Diarization(N3D)은 mel → 특징 쌓기(×8) → Transformer 31 층 → 화자 머리다. 우리 인코더(ASR FastConformer, 같은 80 ms 격자)의 탭에서 N3D 의 m 층 입력 h_m 을
만들 수 있으면, N3D 의 m 층 아래(와 mel 가지)를 떼고 m 층 위 + 머리만 우리 인코더 위에 둔다 — 인코더가 하나인 화자 분리 머리(이식, graft).
  1) 맞춤: dev 창에서 우리 탭(--taps, 기본 체크포인트의 diar_taps + 마지막 층)을 이어 붙인 X 로 h_m(m = 0 … 31, 0 = embed_norm 뒤 = 0 층 입력)을 능형 회귀(λ --ridge).
  2) 잼: eval 창에서 층마다 R²(h_m 예측), 그리고 --graft-layers 의 m 마다 ĥ_m 으로 N3D m 층 위·머리를 돌린 화자 확률의 DER(우리 참조, 앞 4 채널·최적 짝)과
     N3D 자신(교사)과의 프레임 일치. 교사 DER 도 같은 창에서 함께 잰다. 선형 이식이라 학습한 어댑터의 하한이다(학습하면 더 낫다).
  N3D 는 창 통째(오프라인 — 창 안 전체 어텐션)로 돌린다. 스트리밍(AOSC) 이식은 어댑터를 학습할 때 맞춘다.
  python experiments/n3d_graft_probe.py --ckpt <VapAsr 체크포인트(E9 final)> --n3d <.nemo> --out <dir> [--data ami-bench-v1] [--corpora ami-sdm] [--fit-windows 300]
      [--test-windows 300] [--taps diar|all|odd|1,3,…] [--pre] [--ridge 10] [--graft-layers 0,4,8,12,16,20,24,28]
  --pre(2026-10-31, 고정 화자 분리기 대체 모색): 우리 인코더 서브샘플링 출력(0 층 입력 — mel 바로 위)을 X 앞에 붙인다. N3D 는 mel 을 8 개씩 쌓아 바로
      Transformer 로 가므로, 아래쪽 이식(m 작음)은 탭보다 서브샘플링 출력에서 더 잘 복원될 수 있다.
산출: <out>/probe.json(층마다 r2·test DER·교사 일치), <out>/probe.log. 아무것도 지우지 않는다."""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR = 0.08, 16000


def n3d_hidden(m, wav, wav_len):
    """N3D 오프라인 한 번 → (층 입력 hs (L + 1 개, 각 (B, T, 512) — hs[j] = j 층 입력, hs[L] = 마지막 층 출력), 교사 확률 (B, T, 8) 80 ms, 길이 (B,))."""
    enc = m.encoder
    feats, flen = m.preprocessor(input_signal=wav.float(), length=wav_len.long())
    feats = feats[:, :, : int(flen.max())]
    got = {}
    hooks = [enc.embed_norm.register_forward_hook(lambda mod, i_, o: got.__setitem__(0, o))]
    hooks += [layer.register_forward_hook(lambda mod, i_, o, j=j: got.__setitem__(j + 1, o if torch.is_tensor(o) else o[0])) for j, layer in enumerate(enc.layers)]
    try: emb, elen = m.frontend_encoder(processed_signal=feats, processed_signal_length=flen)
    finally:
        for h in hooks: h.remove()
    p = m.forward_infer(emb, elen)
    p = m.sortformer_modules.downsample_preds(p, m.upsample_factor, lengths=elen * m.upsample_factor) if m.high_resolution else p
    return [got[j] for j in range(len(enc.layers) + 1)], p, elen


def n3d_from_layer(m, x, length, j):
    """j 층 입력 x (B, T, 512) → N3D j 층 위 + 머리 → 화자 확률 (B, T, 8) 80 ms(load_n3d 의 SDPA 주의 — n3d.run_layers)."""
    from vapasr.hf.n3d import run_layers
    p = m.forward_infer(run_layers(m, x, length, j), length)
    return m.sortformer_modules.downsample_preds(p, m.upsample_factor, lengths=length * m.upsample_factor) if m.high_resolution else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--n3d", required=True); ap.add_argument("--out", required=True); ap.add_argument("--nemotron-dir", default=None)
    ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1"); ap.add_argument("--corpora", default="ami-sdm")
    ap.add_argument("--window", default="20,40"); ap.add_argument("--fit-windows", type=int, default=300); ap.add_argument("--test-windows", type=int, default=300)
    ap.add_argument("--taps", default="diar"); ap.add_argument("--pre", action="store_true", help="서브샘플링 출력을 X 앞에 붙인다")
    ap.add_argument("--ridge", type=float, default=10.0); ap.add_argument("--graft-layers", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.hf.diar import der_counts, load_encoder, tap_layers
    from vapasr.hf.n3d import load_n3d
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True); log = open(os.path.join(a.out, "probe.log"), "a", buffering=1)
    say = lambda s: (print(s, flush=True), log.write(s + "\n"))
    enc, _ = load_encoder(a.ckpt, a.nemotron_dir, "cuda"); cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    nl = len(enc.enc.layers)
    taps = tuple(cfg["diar_taps"]) if a.taps == "diar" else tuple(tap_layers(a.taps, nl))
    m, _ = load_n3d(a.n3d, "cuda"); L = len(m.encoder.layers); graft = [int(x) for x in a.graft_layers.split(",") if x != ""]
    assert all(0 <= j <= L for j in graft), f"--graft-layers 0–{L}"
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    ns = argparse.Namespace(corpora=a.corpora, data=a.data, split=None, deltas=[10], limit_windows=0, backchannel=(1.0, 3), max_speakers=4,
                            window=tuple(float(v) for v in a.window.split(",")), hop=10.0, window_draws=1, declare=4)
    sets, items = windows_of(ns, tok)
    sub = lambda c, i: ("dev" if sets[c].items[i][0].split(":", 1)[1] in set(man.get("dev", [])) else "eval")
    rng = np.random.RandomState(0); order = rng.permutation(len(items))
    fit = [items[j] for j in order if sub(*items[j]) == "dev"][: a.fit_windows]; test = [items[j] for j in order if sub(*items[j]) == "eval"][: a.test_windows]
    say(f"[{time.strftime('%F %T')}] 탭 {taps}(+ 마지막 층{' + 서브샘플링 출력' if a.pre else ''}) · N3D {L} 층 · 맞춤 dev {len(fit)} 창 · 잼 eval {len(test)} 창 · λ {a.ridge} · 이식 층 {graft}")

    def batches(its):
        for b0 in range(0, len(its), a.batch):
            wavs, Ks, meta = [], [], []
            for c, i in its[b0: b0 + a.batch]:
                ds = sets[c]; cid, t0w, Lw = ds.items[i]; K = int(round(Lw / CHUNK_S)); d = ds.dlgs[cid]
                wavs.append(np.asarray(ds.mono(cid)[int(round(t0w * SR)): int(round((t0w + Lw) * SR))], dtype=np.float32)); Ks.append(K)
                sp = window_speakers(ds.eps[cid], t0w, Lw)[:4]
                ref = torch.tensor(slot_activity(ds.eps[cid], sp, t0w, Lw, K, n_slots=4), dtype=torch.float32).reshape(K, 4)
                mk = torch.ones(K, dtype=torch.bool)
                for s_, e_ in ds._bad_spans(d):
                    x0, x1 = s_ - t0w, e_ - t0w
                    if x1 > 0 and x0 < Lw: mk[max(0, int(x0 / CHUNK_S)): min(K, int(math.ceil(x1 / CHUNK_S)))] = False
                meta.append((K, ref, mk))
            W = torch.zeros(len(wavs), max(len(w) for w in wavs))
            for j, w in enumerate(wavs): W[j, : len(w)] = torch.from_numpy(w)
            wl = torch.tensor([len(w) for w in wavs]); Kt = torch.tensor(Ks)
            with torch.inference_mode():
                got = {}
                hk = enc.enc.pre_encode.register_forward_hook(lambda m_, i_, o: got.__setitem__("pre", o[0] if isinstance(o, tuple) else o)) if a.pre else None
                try: f = enc(W.cuda(), wl.cuda(), Kt.cuda(), taps=taps)                                  # (B, 1 + 탭, Kmax, 1024)
                finally:
                    if hk is not None: hk.remove()
                if a.pre:                                                                                 # 서브샘플링 출력 (B, T', 1024) → 스트림 하나 더(같은 프레임 자르기·0 패딩)
                    pre = got["pre"].float()[:, : f.shape[2]]; f = torch.cat([torch.nn.functional.pad(pre, (0, 0, 0, f.shape[2] - pre.shape[1]))[:, None], f], 1)
                hs, p, elen = n3d_hidden(m, W.cuda(), wl.cuda())
            T = min(f.shape[2], hs[0].shape[1], p.shape[1])
            X = f[:, :, :T].permute(0, 2, 1, 3).reshape(f.shape[0], T, -1).float()                    # (B, T, (1 + 탭)·1024)
            yield X, [h[:, :T].float() for h in hs], p[:, :T].float(), elen.clamp(max=T), [(min(K, T), ref, mk) for K, ref, mk in meta]

    t0 = time.time(); XtX = None; XtY = {}; Xs = Ys = None; n = 0
    for X, hs, _, _, meta in batches(fit):                                                          # 능형 회귀 충분 통계(유효 프레임만, 평균은 따로)
        for b, (K, _, _) in enumerate(meta):
            x = X[b, :K].double(); n += K
            XtX = x.T @ x if XtX is None else XtX + x.T @ x; Xs = x.sum(0) if Xs is None else Xs + x.sum(0)
            for j, h in enumerate(hs):
                y = h[b, :K].double(); XtY[j] = XtY.get(j, 0) + x.T @ y
                Ys = {} if Ys is None else Ys; Ys[j] = Ys.get(j, 0) + y.sum(0)
    say(f"  맞춤 통계 {n} 프레임 · {time.time() - t0:.0f}s")
    D = XtX.shape[0]; mx = Xs / n; C = XtX - n * torch.outer(mx, mx)
    A = torch.linalg.cholesky(C + a.ridge * n * torch.eye(D, dtype=C.dtype, device=C.device) / D)
    Wj, bj = {}, {}
    for j in range(L + 1):
        my = Ys[j] / n; Cy = XtY[j] - n * torch.outer(mx, my)
        Wj[j] = torch.cholesky_solve(Cy, A); bj[j] = my - mx @ Wj[j]
    say(f"  풀이 {time.time() - t0:.0f}s(차원 {D})")
    res = {j: dict(sse=0.0, sst=0.0) for j in range(L + 1)}; der = {j: {k: 0.0 for k in ("ref", "miss", "fa", "conf_best", "agree", "frames")} for j in graft + ["teacher"]}
    for X, hs, p, elen, meta in batches(test):
        with torch.inference_mode():
            for j in range(L + 1):
                yh = (X.double() @ Wj[j] + bj[j]).float()
                for b, (K, _, _) in enumerate(meta):
                    y = hs[j][b, :K]; res[j]["sse"] += float(((yh[b, :K] - y) ** 2).sum()); res[j]["sst"] += float(((y - y.mean(0)) ** 2).sum())
                if j in graft:
                    pg = n3d_from_layer(m, yh, elen.cuda(), j)
                    for b, (K, ref, mk) in enumerate(meta):
                        pr = (pg[b, :K, :4] > a.threshold).float().cpu(); dc = der_counts(pr, ref, mk); g = der[j]
                        for k in ("ref", "miss", "fa", "conf_best"): g[k] += dc[k]
                        t8 = (p[b, :K] > a.threshold).cpu(); g["agree"] += float(((pg[b, :K] > a.threshold).cpu() == t8).all(-1)[mk].sum()); g["frames"] += float(mk.sum())
            for b, (K, ref, mk) in enumerate(meta):
                dc = der_counts((p[b, :K, :4] > a.threshold).float().cpu(), ref, mk); g = der["teacher"]
                for k in ("ref", "miss", "fa", "conf_best"): g[k] += dc[k]
                g["agree"] += float(mk.sum()); g["frames"] += float(mk.sum())
    out = dict(taps=list(taps), pre=bool(a.pre), ridge=a.ridge, fit_windows=len(fit), test_windows=len(test), dim=D,
               r2={j: round(1 - res[j]["sse"] / max(res[j]["sst"], 1e-9), 4) for j in range(L + 1)},
               der={str(j): dict(der=round(100 * (g["miss"] + g["fa"] + g["conf_best"]) / max(1, g["ref"]), 2), miss=round(100 * g["miss"] / max(1, g["ref"]), 2),
                                 fa=round(100 * g["fa"] / max(1, g["ref"]), 2), conf=round(100 * g["conf_best"] / max(1, g["ref"]), 2),
                                 agree=round(g["agree"] / max(1, g["frames"]), 4)) for j, g in der.items()})
    json.dump(out, open(os.path.join(a.out, "probe.json"), "w"), indent=1)
    say("  R² 층 입력 h_m: " + " · ".join(f"{j}:{v}" for j, v in out["r2"].items()))
    for j, v in out["der"].items(): say(f"  이식 m={j:>7}: DER {v['der']}(놓침 {v['miss']} · 헛 {v['fa']} · 혼동 {v['conf']}) · 교사와 프레임 일치 {v['agree']}")
    say(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
