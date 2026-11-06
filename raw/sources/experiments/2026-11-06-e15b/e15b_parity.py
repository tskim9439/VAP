#!/usr/bin/env python
"""E15b 시작 대칭·디코드 스모크(2026-11-06, 서버 호스트 GPU 하나, 제출 없음): AMI eval 회의 하나의 앞 S 초를 cst_stage3_eval 회의 통째와 같은 경로
(encode_long · long_block_decode, 문맥 30 s, δ10, 선언 4)로
  --mode init  : E13 final 그대로 디코드 → 같은 모델에 add_slot_enc(L, H)(0 초기화)를 붙여 다시 디코드 — 출력이 같아야 한다(섞인 소리 스트림 차이·토큰 일치 수를 적는다).
  --mode ckpt  : 학습한 E15b 체크포인트(--checkpoint)로 디코드가 끝까지 도는지(슬롯 토큰 크기·청크 ms).
  --mode slotoff: 학습한 E15b 체크포인트를 그대로 · 슬롯 소리 토큰 0(model.slot_enc_off — cst_stage3_eval --slot-enc-off 와 같다)으로 두 번 디코드해 토큰이 얼마나 다른지(2026-11-06 진단).
  python e15b_parity.py --checkpoint <dir> --encoder <nemotron> --data <ami-bench-v1> [--mode init|ckpt] [--layer 16] [--hidden 1024] [--cid ami-sdm:ES2004b] [--seconds 120]
읽기만 한다(결과는 표준 출력)."""
import argparse, os, sys, time

REPO = "/soundai/users/tskim/VAPKT"
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--encoder", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["init", "ckpt", "slotoff"], default="init"); ap.add_argument("--layer", type=int, default=16); ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--cid", default="ami-sdm:ES2004b"); ap.add_argument("--seconds", type=float, default=120.0); ap.add_argument("--delta", type=int, default=10)
    a = ap.parse_args()
    import numpy as np, torch
    from vapasr.hf.infer import load_model
    from vapasr.hf.long_session import encode_long, long_block_decode
    from cst_stage3_eval import windows_of
    model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=torch.float32); model.eval()
    c = a.cid.split(":")[0]
    ns = argparse.Namespace(data=a.data, split=None, corpora=c, deltas=[a.delta], limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0, max_speakers=4,
                            window_draws=0, declare=4, xl_order="lang", subset="eval")
    sets, items = windows_of(ns, tok); ds = sets[c]; i = next(i for _, i in items if ds.items[i][0] == a.cid)
    langs = ds.slots(i)[1][:4]; prefix = ds.cst_prefix(langs, a.delta); D = len(langs)
    wav = np.asarray(ds.mono(a.cid)[: int(a.seconds * 16000)], dtype=np.float32); K = int(round(len(wav) / 1280))

    def run(tag):
        with torch.inference_mode():
            t0 = time.time(); f = encode_long(model, wav, K); te = time.time() - t0
            t0 = time.time(); em, fc, rd, used, nf, tk = long_block_decode(model, f, prefix, D, context_s=30.0); td = time.time() - t0
            X = model.slot_audio_of(f, model.block_act(f))                                       # inference_mode 안에서(밖이면 학습 가능 가중과 섞여 autograd 오류)
        print(f"{tag}: 특징 {tuple(f.shape)} · 슬롯 토큰 {None if X is None else (tuple(X.shape), round(float(X.abs().mean()), 6))} · 토큰 {len(em)} · 인코딩 {te:.1f} s · 디코드 {td:.1f} s"
              f"(청크 {float(np.mean(tk)):.1f} ms) · GPU 최대 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        return f, em

    if a.mode == "init":
        assert getattr(model, "adapter_slotinj", None) is None, "init 모드는 E15b 가 아닌 체크포인트(E13 final)"
        f0, e0 = run("E13")
        model.add_slot_enc(a.layer, a.hidden); model.eval()
        f1, e1 = run(f"E13 + 슬롯 조건 인코더(층 {a.layer}, 0 초기화)")
        d0 = float((f1[0] - f0[0]).abs().max()); dslot = float((f1[2:] - f1[0:1]).abs().max())
        same = sum(x == y for x, y in zip(e0, e1)); first = next((j for j, (x, y) in enumerate(zip(e0, e1)) if x != y), None)
        print(f"섞인 소리 스트림 최대 차 {d0:.2e} · 슬롯 스트림 − 섞인 소리 최대 {dslot:.2e} · 토큰 같음 {same}/{max(len(e0), len(e1))} · 처음 다른 위치 {first}", flush=True)
        print("PARITY_OK" if e0 == e1 and f1.shape[0] == 2 + 4 else "PARITY_CHECK", flush=True)
    elif a.mode == "slotoff":
        assert getattr(model, "adapter_slotinj", None) is not None, "slotoff 모드는 E15b 체크포인트"
        f0, e0 = run("E15b(슬롯 켬)")
        model.slot_enc_off = True
        f1, e1 = run("E15b(슬롯 토큰 0)")
        same = sum(x == y for x, y in zip(e0, e1)); first = next((j for j, (x, y) in enumerate(zip(e0, e1)) if x != y), None)
        print(f"방출 같음 {same}/{max(len(e0), len(e1))}(길이 {len(e0)} 대 {len(e1)}) · 처음 다른 위치 {first}", flush=True)   # 방출 항목은 튜플이다
        print("SLOTOFF_SAME" if e0 == e1 else "SLOTOFF_DIFF", flush=True)
    else:
        assert getattr(model, "adapter_slotinj", None) is not None, "ckpt 모드는 E15b 체크포인트"
        f, e = run("E15b 체크포인트"); print("DECODE_OK" if len(e) > 0 and f.shape[0] == 2 + int(model.config.slot_enc_cfg["n"]) else "DECODE_CHECK", flush=True)


if __name__ == "__main__":
    main()
