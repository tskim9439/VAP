"""Liger 커널을 Qwen3-ASR thinker 에 적용 (qwen_asr 패키지의 자체 클래스라 Liger 의 apply_liger_kernel_to_qwen3 가 닿지 않는다 → 인스턴스 패치).
RMSNorm(Qwen3ASRThinkerTextRMSNorm) · SwiGLU MLP(Qwen3ASRThinkerTextMLP) · RoPE(apply_rotary_pos_emb) 를 교체. fused linear CE 는 쓰지 않는다
(우리 forward 는 라벨 위치에서만 lm_head 를 계산해 이미 메모리를 줄였다)."""
import torch.nn as nn

def apply_liger_to_thinker(model) -> dict:
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module, _patch_swiglu_module
    except ImportError as e:
        raise ImportError("liger-kernel 이 없습니다: pip install liger-kernel") from e
    import qwen_asr.core.transformers_backend.modeling_qwen3_asr as M
    M.apply_rotary_pos_emb = liger_rotary_pos_emb                                  # 모듈 전역 함수 → attention.forward 가 참조
    n = dict(rms_norm=0, swiglu=0, rope=1)
    rms = tuple(getattr(M, c) for c in ("Qwen3ASRThinkerTextRMSNorm", "Qwen3ASRTextRMSNorm") if hasattr(M, c))      # thinker 디코더 층은 Qwen3ASRText* 클래스를 쓴다
    mlp = tuple(getattr(M, c) for c in ("Qwen3ASRThinkerTextMLP", "Qwen3ASRTextMLP") if hasattr(M, c))
    for mod in model.thinker.modules():
        if isinstance(mod, rms): _patch_rms_norm_module(mod); n["rms_norm"] += 1
        elif isinstance(mod, mlp): _patch_swiglu_module(mod, LigerSwiGLUMLP); n["swiglu"] += 1
    return n
