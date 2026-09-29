"""Thinker varlen attention without the flash-attn package — torch's bundled FlashAttention-2 varlen kernel
(torch.ops.aten._flash_attention_forward: int32 cu_seqlens, GQA, autograd) registered in transformers' AttentionInterface.

Why: with sdpa a packed thinker row gets an O(N²) 4D block mask (thinker fwd+bwd on a 15.5 k-token KO-like batch, mxc H200
2026-09-29: packed sdpa 2.34 s vs padded sdpa 0.25 s, packed varlen 0.18 s), and a padded batch with even one pad token leaves the
flash path for the masked kernel (no causal skip, repeat_kv for GQA). The varlen kernel costs Σ nᵢ² per batch and skips pad tokens
entirely.

  register()        AttentionInterface[IMPL] = varlen_attention_forward, AttentionMaskInterface[IMPL] = transformers'
                    flash_attention_mask (packed: None, padded: the (B, L) bool mask, no padding: None). Idempotent.
                    Then thinker.set_attn_implementation(IMPL) (vapasr/speedup.set_attn_impl does both for --attn-impl varlen).
  is_varlen(th)     the thinker's attention modules see thinker.model.config._attn_implementation == IMPL.
  packed_kwargs()   cu_seq_lens_q/k (int32) + max_length_q/k for thinker.model(**kw) — from packed position_ids (restart = new
                    sample, same rule as transformers' find_packed_sequence_indices) or from per-sample lengths.

The name must not start with "flash_attention" nor contain "/": transformers would then try the kernels hub (a download) or
import flash_attn. The attention function, in order:
  1. cu_seq_lens_q in kwargs → use them (VapAsrForStreamingASR.forward builds them; no host sync inside the layers). A 2D padding
     mask next to them is refused (NotImplementedError): the segments would silently override it.
  2. 2D padding mask → unpad once per forward (memo), run varlen over the real tokens, scatter back (pad rows stay 0).
  3. no mask, Lq == Lk, position_ids → segments per row from position_ids (memo; a sample never spans rows, restarts split a
     row — the rule sdpa's packed-sequence detection uses) — protects callers that run the thinker on packed rows without cu
     kwargs (plain causal over a row would be silently wrong). For contiguous positions this is exactly cu = arange·L.
     The function cannot see the KV cache (past_key_values is not forwarded to it), so this also applies to the prefill of an
     empty cache; there sdpa does not detect packing (transformers only looks for it without past_key_values) and runs plain
     causal, so a cache prefill with restarting position_ids differs from sdpa. No caller does that (stream_decode lets the
     thinker count positions).
  4. otherwise uniform rows (cu = arange·L): KV-cache decode steps (q shorter than k). FA2 aligns causal bottom-right when
     q_len < k_len, which is what chunked decode with a cache needs.
The aten op has no autocast kernel, so q/k/v are cast explicitly (without Liger, q/k come out of RMSNorm·RoPE in fp32 while v is
bf16). On CPU (tests) a per-segment SDPA reference with the same semantics runs instead of the kernel.
Only torch is imported at module level (transformers inside register()).
"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F

IMPL = "vapasr_varlen"
_MEMO: dict = {}          # kind -> (tensor, version, value): strong ref + identity so a freed address can never alias a new tensor
_WARNED: set = set()
log = logging.getLogger(__name__)


def register() -> str:
    """Register IMPL with transformers (attention function + mask builder). Safe to call more than once."""
    from transformers.modeling_utils import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
    AttentionInterface.register(IMPL, varlen_attention_forward)
    AttentionMaskInterface.register(IMPL, flash_attention_mask)
    return IMPL


def is_varlen(thinker) -> bool:
    """True when the thinker's text model (whose config the attention layers read) uses IMPL."""
    cfg = getattr(getattr(thinker, "model", thinker), "config", None)
    return getattr(cfg, "_attn_implementation", None) == IMPL


def _cu_from_lengths(lens: torch.Tensor) -> torch.Tensor:
    lens = lens.to(torch.int32)
    return torch.cat([lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)])


def _pos_cu(p: torch.Tensor) -> dict:
    """(R, N) positions → cu over the R·N flattened tokens: a segment starts at every row start and wherever p[i] != p[i-1] + 1."""
    brk = torch.ones_like(p, dtype=torch.bool); brk[:, 1:] = p[:, 1:] != p[:, :-1] + 1
    starts = brk.flatten().nonzero().flatten().to(torch.int32)
    cu = torch.cat([starts, starts.new_tensor([p.numel()])])
    mx = int((cu[1:] - cu[:-1]).max()) if p.numel() else 0
    return dict(cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=mx, max_length_k=mx)


def packed_kwargs(position_ids: Optional[torch.Tensor] = None, lengths=None, device=None) -> dict:
    """cu_seq_lens_q/k (int32, on the input's device) + max_length_q/k (python int) for one packed row.
    lengths (B,): cu = [0, cumsum(lengths)] — samples laid out back to back.
    position_ids (1, N) (or (3, 1, N) mRoPE): a new segment starts wherever p[i] != p[i-1] + 1 (masking_utils'
    find_packed_sequence_indices rule, i.e. exactly the block mask sdpa builds from the same position_ids).
    Host syncs: nonzero (position_ids only) + max."""
    if lengths is not None:
        lens = torch.as_tensor(lengths, device=device)
        cu = _cu_from_lengths(lens); mx = int(lens.max()) if lens.numel() else 0
        return dict(cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=mx, max_length_k=mx)
    p = position_ids
    if p.dim() == 3: p = p[0]
    if p.dim() == 1: p = p[None]
    if p.shape[0] != 1: raise ValueError(f"packed_kwargs: position_ids must be one packed row, got {tuple(position_ids.shape)}")
    return _pos_cu(p)


def _row_segments(pos: torch.Tensor, B: int) -> dict:
    """Branch 3: cu for B rows of position_ids ((B, L), (1, L) broadcast like transformers does, or (3, B, L) mRoPE)."""
    p = pos[0] if pos.dim() == 3 else pos
    if p.dim() == 1: p = p[None]
    if p.shape[0] != B:
        if p.shape[0] != 1: raise ValueError(f"{IMPL}: position_ids {tuple(pos.shape)} do not match batch {B}")
        p = p.expand(B, -1)
    return _pos_cu(p)


def _version(t: torch.Tensor):
    try:
        return t._version
    except RuntimeError:              # inference tensors (stream_decode runs under inference_mode) have no version counter
        return None


def _memo(kind: str, t: torch.Tensor, fn):
    """fn(t) cached for the same tensor object at the same version — all layers of one forward (and the gradient-checkpoint
    recompute, which replays the same kwargs) share one host sync."""
    hit = _MEMO.get(kind); ver = _version(t)
    if hit is not None and hit[0] is t and hit[1] == ver:
        return hit[2]
    v = fn(t); _MEMO[kind] = (t, ver, v)
    return v


def _unpad(mask: torch.Tensor):
    m = mask.bool(); lens = m.sum(1, dtype=torch.int32)
    return m.flatten().nonzero().flatten(), _cu_from_lengths(lens), int(lens.max()) if lens.numel() else 0


def _reference_varlen(q, k, v, cu_q, cu_k, causal: bool, scale, dropout: float = 0.0):
    """CPU reference with the kernel's semantics: per segment SDPA, GQA by repeating kv heads, causal aligned bottom-right
    (row i sees keys j ≤ i + Lk − Lq). q (Tq, H, D), k/v (Tk, Hkv, D); the segments cover all Tq rows."""
    rep = q.shape[1] // k.shape[1]; cq, ck = cu_q.tolist(), cu_k.tolist(); parts = []
    for i in range(len(cq) - 1):
        a, b, c, d = cq[i], cq[i + 1], ck[i], ck[i + 1]
        if b == a: continue
        qi = q[a:b].transpose(0, 1); ki = k[c:d].transpose(0, 1).repeat_interleave(rep, 0); vi = v[c:d].transpose(0, 1).repeat_interleave(rep, 0)
        m = torch.ones(b - a, d - c, dtype=torch.bool, device=q.device).tril((d - c) - (b - a)) if causal else None
        parts.append(F.scaled_dot_product_attention(qi, ki, vi, attn_mask=m, dropout_p=dropout, scale=scale).transpose(0, 1))
    return torch.cat(parts) if parts else q.new_zeros(q.shape)


def _flash_varlen(q, k, v, cu_q, cu_k, max_q: int, max_k: int, causal: bool, scale, dropout: float):
    if q.is_cuda:
        return torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, int(max_q), int(max_k), float(dropout), bool(causal), False, scale=scale)[0]
    return _reference_varlen(q, k, v, cu_q, cu_k, causal, scale, dropout)


def _warn_once(key: str, msg: str):
    if key not in _WARNED:
        _WARNED.add(key); log.warning(msg)


def varlen_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling: Optional[float] = None,
                             sliding_window=None, softcap=None, **kwargs):
    """transformers attention function: query (B, H, Lq, D), key/value (B, Hkv, Lk, D) → ((B, Lq, H, D), None)."""
    if sliding_window is not None or softcap is not None:
        raise NotImplementedError(f"{IMPL}: sliding_window / softcap are not supported (Qwen3-ASR uses neither)")
    if kwargs.get("output_attentions") or kwargs.get("head_mask") is not None:
        _warn_once("attn_out", f"{IMPL}: output_attentions / head_mask are ignored (like sdpa)")
    B, H, Lq, D = query.shape; Hk, Lk = key.shape[1], key.shape[2]; odt = query.dtype
    autocast = query.is_cuda and torch.is_autocast_enabled("cuda")
    if not query.is_cuda:
        dt = odt                                                                   # CPU reference: exact, no cast
    elif autocast:
        dt = torch.get_autocast_dtype("cuda")                                      # the aten op has no autocast kernel
    elif odt in (torch.float16, torch.bfloat16):
        dt = odt
    else:
        dt = torch.bfloat16
        _warn_once("fp32", f"{IMPL}: {odt} inputs without autocast — attention runs in bf16 (FlashAttention has no fp32 kernel)")
    q = query.transpose(1, 2).reshape(B * Lq, H, D).to(dt)                         # (B,H,L,D) views of (B,L,H,D) → no copy
    k = key.transpose(1, 2).reshape(B * Lk, Hk, D).to(dt)
    v = value.transpose(1, 2).reshape(B * Lk, Hk, D).to(dt)
    causal = kwargs.pop("is_causal", None)
    causal = bool(getattr(module, "is_causal", True)) if causal is None else bool(causal)
    pos = kwargs.get("position_ids"); cu_q = kwargs.get("cu_seq_lens_q")
    if cu_q is not None:                                                           # 1. caller-built segments (packed row)
        if attention_mask is not None:
            raise NotImplementedError(f"{IMPL}: cu_seq_lens_* together with a padding mask — pass one or the other (the segments would ignore the mask)")
        cu_k = kwargs.get("cu_seq_lens_k"); cu_k = cu_q if cu_k is None else cu_k
        mq = kwargs.get("max_length_q"); mk = kwargs.get("max_length_k")
        if mq is None: mq = int((cu_q[1:] - cu_q[:-1]).max())
        if mk is None: mk = mq if cu_k is cu_q else int((cu_k[1:] - cu_k[:-1]).max())
        o = _flash_varlen(q, k, v, cu_q, cu_k, mq, mk, causal, scaling, dropout)
    elif attention_mask is not None:                                               # 2. padded batch: unpad per layer
        if attention_mask.dim() != 2:
            raise NotImplementedError(f"{IMPL}: got a {attention_mask.dim()}D attention mask — register() the mask builder and set the implementation with set_attn_implementation")
        if Lq != Lk:
            raise NotImplementedError(f"{IMPL}: padding mask together with a KV cache (q {Lq} vs k {Lk}) is not supported")
        idx, cu, mx = _memo("mask", attention_mask, _unpad)
        o = q.new_zeros(q.shape).index_copy(0, idx, _flash_varlen(q[idx], k[idx], v[idx], cu, cu, mx, mx, causal, scaling, dropout))
    elif Lq == Lk and pos is not None and pos.shape[-1] == Lq:                     # 3. no mask: segments per row from position_ids
        fa = _memo(f"pos{B}", pos, lambda p: _row_segments(p, B))
        o = _flash_varlen(q, k, v, fa["cu_seq_lens_q"], fa["cu_seq_lens_k"], fa["max_length_q"], fa["max_length_k"], causal, scaling, dropout)
    else:                                                                          # 4. uniform rows: KV-cache decode steps
        cq = torch.arange(0, (B + 1) * Lq, Lq, device=q.device, dtype=torch.int32)
        ck = torch.arange(0, (B + 1) * Lk, Lk, device=q.device, dtype=torch.int32)
        o = _flash_varlen(q, k, v, cq, ck, Lq, Lk, causal, scaling, dropout)
    o = o.view(B, Lq, H, D)
    return (o if (autocast or not query.is_cuda) else o.to(odt)), None
