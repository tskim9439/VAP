#!/usr/bin/env python
"""N3D 흡수 변환(2026-10-31, 사용자 "N3D 흡수하는 방향도 좋아" — wiki output-diarizer-replacement-20261031 길 3): 고정 전문가를 쓰는 체크포인트(E11 — config 의
n3d_cfg.path 가 .nemo)를 N3D 가지가 안에 든 체크포인트로 바꾼다(VapAsr.absorb_n3d — 학습 없음).
  python experiments/n3d_absorb.py --ckpt <E11 final> --out <새 폴더> [--share-mel] [--nemotron-dir <dir>]
- 가중치: n3d_net.* 가 safetensors 에 들어가고(고정), 틀은 n3d_config.yaml 이다. .nemo 없이 from_pretrained 가 다시 세운다.
  토크나이저 파일 등 나머지는 원본에서 복사한다(학습 상태 optimizer·scheduler·trainer_state·rng 는 뺀다).
- mel: 기본은 우리 인코더 전처리기가 N3D 것과 같은 값을 낼 때만 공유한다. N3D 체크포인트의 filterbank·window 가 bf16 이라 보통은 N3D 자신의 mel 모듈을
  가지 안에 둔다(출력이 E11 과 같다). --share-mel 은 우리 것을 쓴다(실험용 — 출력이 조금 바뀐다).
- 대조: 저장한 폴더를 from_pretrained 로 다시 읽어 ① n3d_net 가중치가 .nemo 와 같고 ② 시험 신호(30 s)의 전문가 확률이 고정 전문가와 같은지 본다
  (--share-mel 이면 ②는 차이만 적는다). out/absorb.json 에 남기고, 어긋나면 <out>.tmp<pid> 에 둔 채 멈춘다. <out> 이 이미 있으면 덮어쓰지 않는다."""
import argparse, json, os, shutil, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP = ("optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "rng_state.pth", "config.json", "model.safetensors.index.json")


def signal(sec: float = 30.0, seed: int = 0):
    """두 사람이 번갈아 말하는 꼴의 시험 신호(잡음 + 음높이가 다른 두 성분) → (wav (1, L), wav_len (1,), K (1,))."""
    g = torch.Generator().manual_seed(seed); n = int(16000 * sec); x = (torch.randn(n, generator=g) * 0.03).float(); t = torch.arange(n) / 16000.0
    for j, f0 in enumerate((180.0, 260.0)):
        on = ((t // 3.0).long() % 2 == j).float(); x += on * 0.25 * torch.sin(2 * np.pi * f0 * t) * (torch.sin(2 * np.pi * 4 * t) > -0.3).float()
    return x[None], torch.tensor([n]), torch.tensor([n // 1280])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--nemotron-dir", default=None)
    ap.add_argument("--share-mel", action="store_true", help="N3D mel 을 우리 인코더 전처리기로(값이 달라도 — 실험용)")
    a = ap.parse_args()
    assert not os.path.exists(a.out), f"이미 있다: {a.out} — 덮어쓰지 않는다"
    from vapasr.hf import VapAsrForStreamingASR
    from vapasr.hf.n3d import N3DExpert, load_n3d
    t0 = time.time(); m = VapAsrForStreamingASR.from_pretrained(a.ckpt, encoder_path=a.nemotron_dir).eval()
    assert m.n3d is not None and not (m.config.n3d_cfg or {}).get("embedded"), f"고정 전문가(add_n3d — E11)를 쓰는 체크포인트만: {a.ckpt}"
    path = m.n3d.path(); m.absorb_n3d(share_mel=True if a.share_mel else None)
    print(f"[{time.strftime('%H:%M:%S')}] 흡수: {a.ckpt} ← N3D {path} · mel 공유 {m.config.n3d_cfg['shared_mel']} · {time.time() - t0:.0f}s", flush=True)
    tmp = f"{a.out}.tmp{os.getpid()}"; m.save_pretrained(tmp)
    for f in sorted(os.listdir(a.ckpt)):
        src = os.path.join(a.ckpt, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") and f not in SKIP and not os.path.exists(os.path.join(tmp, f)): shutil.copy2(src, os.path.join(tmp, f))
    r = VapAsrForStreamingASR.from_pretrained(tmp, encoder_path=a.nemotron_dir).eval()                # 대조: .nemo 없이 다시 세운 가지
    ref, _ = load_n3d(path); sa, sb = ref.state_dict(), r.n3d_net.state_dict()
    w_eq = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k].to(sa[k].device)) for k in sa)
    w, wl, K = signal(); p_ref = N3DExpert({**{k: v for k, v in m.config.n3d_cfg.items() if k not in ("embedded", "shared_mel", "train")}, "path": path}).probs(w, wl, K)
    p_new = r.n3d.probs(w, wl, K); diff = float((p_new - p_ref).abs().max())
    rec = dict(src=os.path.abspath(a.ckpt), out=os.path.abspath(a.out), n3d=path, shared_mel=bool(r.config.n3d_cfg.get("shared_mel")), weights_equal=bool(w_eq),
               probs_maxdiff=diff, probs_frames=int(p_ref.shape[1]), act_agree=float(((p_new > 0.5) == (p_ref > 0.5)).float().mean()), time=time.strftime("%F %T"))
    json.dump(rec, open(os.path.join(tmp, "absorb.json"), "w"), indent=1); print(f"대조: {rec}", flush=True)
    assert w_eq and (a.share_mel or diff <= 1e-5), f"흡수 대조 불일치 — {tmp} 에 두고 멈춘다: {rec}"
    os.replace(tmp, a.out); print(f"DONE → {a.out}", flush=True)


if __name__ == "__main__":
    main()
