#!/usr/bin/env python
"""VapAsr 체크포인트의 thinker(Qwen3-ASR 텍스트 모델 = Qwen3 디코더)를 HF Qwen3ForCausalLM 형식으로 뽑아 mlx-lm 으로 변환한다.
  python experiments/live/export_thinker_mlx.py --ckpt ~/Desktop/VAPKT-models/hf-E2-final [--q-bits 8]
  → <ckpt>-thinker-hf/ (중간 HF 디렉토리), <ckpt>-thinker-mlx[-q8]/ (mlx-lm 모델)
mRoPE(interleaved, section [24,20,20])는 위치 3 스트림이 모두 같을 때(텍스트·오디오 임베딩 시퀀스) 표준 RoPE 와 같다 — probe_live_mlx.py 로 logits 동일성을 검증."""
import os, sys, json, argparse, shutil, torch
from safetensors.torch import load_file, save_file
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--q-bits", type=int, default=0); ap.add_argument("--dtype", default="float16"); a = ap.parse_args()
ck = os.path.abspath(a.ckpt.rstrip("/")); hf = ck + "-thinker-hf"; out = ck + ("-thinker-mlx-q%d" % a.q_bits if a.q_bits else "-thinker-mlx")
cfg = json.load(open(os.path.join(ck, "config.json"))); tc = cfg["thinker"]["text_config"]
hfc = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"], num_hidden_layers=tc["num_hidden_layers"],
           num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"], rms_norm_eps=tc["rms_norm_eps"], vocab_size=tc["vocab_size"],
           rope_theta=tc["rope_theta"], max_position_embeddings=tc.get("max_position_embeddings", 40960), tie_word_embeddings=True, hidden_act="silu", attention_bias=False, torch_dtype=a.dtype,
           bos_token_id=tc.get("bos_token_id"), eos_token_id=tc.get("eos_token_id"))
os.makedirs(hf, exist_ok=True); json.dump(hfc, open(os.path.join(hf, "config.json"), "w"), indent=1)
sd = load_file(os.path.join(ck, "model.safetensors")); th = {k[len("thinker."):]: v.to(getattr(torch, a.dtype)).contiguous() for k, v in sd.items() if k.startswith("thinker.")}
print("thinker tensors", len(th), "emb", tuple(th["model.embed_tokens.weight"].shape)); save_file(th, os.path.join(hf, "model.safetensors"), metadata={"format": "pt"})
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"):
    if os.path.exists(os.path.join(ck, f)): shutil.copy(os.path.join(ck, f), hf)
from mlx_lm import convert
if os.path.exists(out): shutil.rmtree(out)
convert(hf_path=hf, mlx_path=out, quantize=bool(a.q_bits), q_bits=a.q_bits or None, q_group_size=64 if a.q_bits else None, dtype=a.dtype)
print("→", out, sorted(os.listdir(out)))
