#!/usr/bin/env python
"""D1 산출물 인코더 복구 — 동결 인코더가 저장되지 않아 재로드 시 .nemo 원본이 붙던 체크포인트에 초기화 체크포인트(E2)의 학습된 인코더를 붙여 encoder_saved=True 로 다시 저장한다.
  python experiments/p2_fix_encoder.py --model /soundai/Model/VAPASR/p2-D1/final --encoder-from /soundai/Model/VAPASR/hf-E2/final --out /soundai/Model/VAPASR/p2-D1/final-enc"""
import os, sys, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--encoder-from", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
import torch
from safetensors.torch import load_file
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
enc_sd = {}
for f in sorted(glob.glob(os.path.join(a.encoder_from, "*.safetensors"))):
    for k, v in load_file(f).items():
        if k.startswith("encoder.enc."): enc_sd[k[len("encoder.enc."):]] = v
assert enc_sd, "encoder-from 에 인코더 가중치 없음"
m = VapAsrForStreamingASR.from_pretrained(a.model); tok = load_tokenizer(a.model)
before = {k: v.clone() for k, v in list(m.encoder.enc.state_dict().items())[:3]}
m.encoder.enc.load_state_dict(enc_sd); m.config.encoder_saved = True; m.config.encoder_trainable = False
changed = sum(int(not torch.equal(before[k], m.encoder.enc.state_dict()[k])) for k in before)
os.makedirs(a.out, exist_ok=True); m.save_pretrained(a.out); tok.save_pretrained(a.out)
print(f"인코더 {len(enc_sd)} 키 교체(표본 3 중 {changed} 변경) → {a.out}")
m2 = VapAsrForStreamingASR.from_pretrained(a.out); ok = all(torch.equal(m2.encoder.enc.state_dict()[k].cpu(), enc_sd[k]) for k in list(enc_sd)[:20]); print("재로드 인코더 일치:", ok)
