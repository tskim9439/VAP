"""Qwen3-ASR AuT 인코더에 attention 마스크를 복원/주입한다.

배경: qwen-asr 0.0.6 의 Qwen3ASRAudioEncoder.forward 는 _prepare_attention_mask 를 호출하지 않아
sdpa/eager 경로에서 attention_mask=None → 전체 발화 양방향 (output-encoder-causality-audit).
레이어 forward 는 attention_mask 를 받으므로, 레이어를 감싸 cu_seqlens 로부터 마스크를 만들어 넣는다.

modes
  block           cu_seqlens 블록 대각 (FA2 varlen 과 동일한 원래 의도). 블록 = n_window_infer.
  chunked-causal  블록 내 양방향 + 이전 블록들 전부(또는 left_ctx_blocks 개)에 attention, 미래 블록 차단.
  causal          프레임 단위 lower-triangular (모델이 본 적 없는 마스크 — 열화 측정용).
  none            패치 해제 (원래 동작).

사용
  from experiments.qwen_aut_mask import patch_aut
  patch_aut(enc, mode="chunked-causal", block_fbank_frames=100, left_ctx_blocks=None)
"""
import torch

_STATE = {}

def _build_mask(T, cu, mode, left_ctx_blocks, dtype, device):
    cu = [int(c) for c in cu]
    blk = torch.zeros(T, dtype=torch.long, device=device)
    for i in range(1, len(cu)):
        blk[cu[i - 1]:cu[i]] = i - 1
    q = blk[:, None]; k = blk[None, :]
    if mode == "block":
        allow = q == k
    elif mode == "chunked-causal":
        allow = k <= q
        if left_ctx_blocks is not None:
            allow &= (q - k) <= left_ctx_blocks
    elif mode == "causal":
        idx = torch.arange(T, device=device)
        allow = idx[None, :] <= idx[:, None]
        if left_ctx_blocks is not None:
            allow &= (q - k) <= left_ctx_blocks          # 프레임 causal + 좌측 블록 한도 (학습 블록 8 s 에 맞춤)
    else:
        raise ValueError(mode)
    m = torch.full((1, 1, T, T), torch.finfo(dtype).min, dtype=dtype, device=device)
    m.masked_fill_(allow[None, None], 0.0)
    return m

def patch_aut(enc, mode="block", block_fbank_frames=None, left_ctx_blocks=None):
    """enc: Qwen3ASRAudioEncoder. block_fbank_frames: n_window_infer (100 = 1 s, conv chunk 배수여야 함)."""
    unpatch_aut(enc)
    if mode == "none":
        return enc
    if block_fbank_frames is not None:
        _STATE.setdefault(id(enc), {})["n_window_infer"] = enc.n_window_infer
        enc.n_window_infer = block_fbank_frames
    cache = {}
    for layer in enc.layers:
        orig = layer.forward
        def fwd(hidden_states, cu_seqlens, attention_mask=None, _orig=orig, **kw):
            if attention_mask is None:
                key = (hidden_states.shape[0], tuple(int(c) for c in cu_seqlens), hidden_states.dtype)
                if key not in cache:
                    cache.clear()
                    cache[key] = _build_mask(hidden_states.shape[0], cu_seqlens, mode, left_ctx_blocks, hidden_states.dtype, hidden_states.device)
                attention_mask = cache[key]
            return _orig(hidden_states, cu_seqlens, attention_mask=attention_mask, **kw)
        layer.forward = fwd
        _STATE.setdefault(id(enc), {}).setdefault("orig", []).append((layer, orig))
    _STATE[id(enc)]["mode"] = mode
    return enc

def unpatch_aut(enc):
    st = _STATE.pop(id(enc), None)
    if not st: return
    for layer, orig in st.get("orig", []):
        layer.forward = orig
    if "n_window_infer" in st:
        enc.n_window_infer = st["n_window_infer"]
