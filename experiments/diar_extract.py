#!/usr/bin/env python
"""E6·E7 모델(VapAsr 체크포인트)의 화자 분리 헤드를 diar.pt 로 꺼낸다(2026-10-27, E7 — diar_train --init-branch·diar_eval run [--session] 용).
  python experiments/diar_extract.py --model <out/final 또는 checkpoint-N> --out <dir>
→ <dir>/diar.pt(state·cfg·taps·init = 그 모델 — load_encoder 가 거기서 인코더를 읽는다(stage 3 는 인코더도 학습한다)·step None) + diar.json.
읽기 전용 입력, <dir> 가 이미 있으면 멈춘다(덮어쓰지 않는다)."""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract(model_dir: str) -> dict:
    from safetensors import safe_open
    from vapasr.hf.diar import DiarBranch
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    assert cfg.get("diar_head_cfg") and cfg.get("diar_taps"), f"화자 분리 헤드가 없는 체크포인트: {model_dir}"
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    state = {}
    for f in files:
        with safe_open(os.path.join(model_dir, f), "pt") as h:
            for k in h.keys():
                if k.startswith("diar_head."): state[k[len("diar_head."):]] = h.get_tensor(k).float()
    assert state, f"diar_head.* 가중치가 없다: {model_dir}"
    DiarBranch(**cfg["diar_head_cfg"]).load_state_dict(state)                                      # 키·모양 확인(strict)
    return dict(state=state, cfg=dict(cfg["diar_head_cfg"]), taps=[int(x) for x in cfg["diar_taps"]], init=os.path.abspath(model_dir), step=None,
                threshold=float(cfg.get("act_threshold", 0.5)), src="diar_extract", **({"aosc": cfg["diar_aosc"]} if cfg.get("diar_aosc") else {}),
                **({"shift": int(cfg["act_shift"])} if cfg.get("act_shift") else {}))                     # E8: 헤드 출력 p = 프레임 p − R(diar_eval 이 맞춰 잰다)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    if os.path.exists(a.out): sys.exit(f"이미 있다(덮어쓰지 않는다): {a.out}")
    x = extract(a.model); os.makedirs(a.out)
    torch.save(x, os.path.join(a.out, "diar.pt"))
    json.dump({k: v for k, v in x.items() if k != "state"}, open(os.path.join(a.out, "diar.json"), "w"), indent=1)
    print(f"EXTRACT_OK {a.out} · 가중치 {len(x['state'])} · cfg {x['cfg']} · 탭 {x['taps']}", flush=True)


if __name__ == "__main__":
    main()
