"""Training speed-ups shared by experiments/{s3_train_hf,p2_train_hf,semcommit_train}.py.

  --batch-max-tokens N  dynamic batch by max length (vapasr/hf/packing.py, padded): batch size n with n × longest estimated
                        sequence ≤ N — short streams get large batches, long ones small. 0 = fixed --bs-en/--bs-ko. (p2_train_hf.py has
                        the same thing as --max-tokens.) Recommended: H200 semcommit, bs 16 ≈ 4.7 k padded tokens / 16 GB.
  --pack-max-tokens N   sequence packing: each batch = one thinker row of Σ ≤ N estimated tokens (samples concatenated,
                        block-diagonal causal mask). Only worth it with a varlen attention kernel (flash_attention_2): with sdpa it
                        was 2.6× slower than padded batches on H200 (attention over the whole row, mostly masked).
  --pack-max-bs M       samples per budget batch cap (packing or padded; encoder batch size).
  --attn-impl X         thinker attention: sdpa (default) | flex_attention | eager. flash_attention_2 needs the flash-attn package (not in the env).
                        flex_attention + packing is refused: on torch 2.9 / transformers 4.57 the packed BlockMask gives the right loss but
                        wrong gradients (k_proj grad 100 % off vs sdpa, 2026-09-27 mxc H200) — sdpa with the packed 4D mask is exact.
  --optim O             TrainingArguments.optim, default adamw_torch_fused (torch ≥ 2.8 default; the env has torch 2.9).
  --deepspeed CFG       DeepSpeed config JSON (configs/deepspeed/*.json) passed to TrainingArguments — needs the deepspeed package.
Liger (RMSNorm/SwiGLU/RoPE) stays in vapasr/hf/liger.py; apply_speedups turns it on unless disabled.
Kept outside vapasr.hf (whose __init__ imports transformers/torch) so scripts can add the args before CUDA_VISIBLE_DEVICES is set.
"""
import argparse
import json


def add_speed_args(ap: argparse.ArgumentParser, batch_budget: bool = True):
    g = ap.add_argument_group("speed")
    if batch_budget:
        g.add_argument("--batch-max-tokens", type=int, default=0, help="dynamic padded batch: n × longest ≤ this many tokens (0 = fixed bs)")
    g.add_argument("--pack-max-tokens", type=int, default=0, help="sequence packing token budget per batch (0 = off; needs a varlen kernel to pay)")
    g.add_argument("--pack-max-bs", type=int, default=64, help="max samples per token-budget batch (packing or --batch-max-tokens)")
    g.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "flex_attention", "eager", "flash_attention_2"], help="thinker attention implementation")
    g.add_argument("--optim", default="adamw_torch_fused", help="TrainingArguments.optim (adamw_torch = old default)")
    g.add_argument("--deepspeed", default=None, help="DeepSpeed config JSON (ZeRO-2 recommended: configs/deepspeed/zero2.json)")
    return g


def set_attn_impl(model, impl: str, log=print):
    """Switch the thinker's attention implementation in place (the thinker is built from a config dict, so it never gets one
    explicitly). flash_attention_2 is refused early when flash-attn is missing."""
    if impl == "flash_attention_2":
        import importlib.util
        if importlib.util.find_spec("flash_attn") is None:
            raise SystemExit("--attn-impl flash_attention_2 needs the flash-attn package (not installed)")
    th = model.thinker
    try:
        th.set_attn_implementation(impl)
    except (ValueError, AttributeError) as e:
        # Qwen3ASRThinkerForConditionalGeneration does not declare _supports_flex_attn although its attention goes through
        # transformers' ALL_ATTENTION_FUNCTIONS; set the configs directly (tests/test_packing.py checks packed == padded under it).
        log(f"set_attn_implementation({impl}) refused ({str(e)[:80]}...) — setting config._attn_implementation directly")
        for c in (th.config, getattr(th.config, "text_config", None)):
            if c is not None: c._attn_implementation = impl
    got = getattr(th.config, "_attn_implementation", None)
    log(f"thinker attention: {got}")
    return got


def apply_speedups(model, a, log=print, liger: bool = True):
    """Attention implementation + Liger. Returns a dict for the run record."""
    if a.pack_max_tokens and a.attn_impl == "flex_attention":
        raise SystemExit("--attn-impl flex_attention with --pack-max-tokens: packed BlockMask backward is wrong on this stack — use sdpa")
    if a.pack_max_tokens and getattr(a, "batch_max_tokens", 0):
        raise SystemExit("--pack-max-tokens and --batch-max-tokens are exclusive")
    if a.pack_max_tokens and a.attn_impl != "flash_attention_2":
        log(f"! packing with {a.attn_impl}: attention runs over the whole packed row (O(N²), mostly masked) — measured 2.6× slower than "
            "padded batches with sdpa; prefer --batch-max-tokens unless flash_attention_2 is available")
    rec = dict(attn_impl=set_attn_impl(model, a.attn_impl, log), pack_max_tokens=a.pack_max_tokens, batch_max_tokens=getattr(a, "batch_max_tokens", 0), pack_max_bs=a.pack_max_bs,
               optim=a.optim, deepspeed=a.deepspeed, liger=False)
    if liger:
        try:
            from .hf.liger import apply_liger_to_thinker
            rec["liger"] = apply_liger_to_thinker(model); log(f"liger: {rec['liger']}")
        except ImportError as e:
            log(f"liger not applied ({e})")
    if a.pack_max_tokens and a.attn_impl == "eager":
        log("! packing with eager attention materialises a dense mask — prefer sdpa or flex_attention")
    return rec


def training_args_kwargs(a) -> dict:
    """optim + deepspeed for TrainingArguments. With DeepSpeed the fp32 frozen encoder must not be cast to bf16: the shipped
    configs keep bf16 off in DeepSpeed and use torch autocast (bf16) instead, so TrainingArguments(bf16=...) must follow."""
    kw = dict(optim=a.optim)
    if a.deepspeed:
        cfg = json.load(open(a.deepspeed))
        kw["deepspeed"] = a.deepspeed
        if (cfg.get("bf16") or {}).get("enabled") is False and (cfg.get("torch_autocast") or {}).get("enabled"):
            kw["bf16"] = False                     # autocast comes from the DeepSpeed engine; Trainer's own bf16 would cast the model
    return kw
