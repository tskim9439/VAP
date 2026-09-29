"""Sequence packing for the thinker (LLM) side + token-budget batching by packed length — shared by Phase 1 (s3/semcommit) and Phase 2.

Why only the thinker is packed: the Nemotron encoder has a 56-frame left context, so audio from different streams must not be
concatenated. The encoder keeps its padded per-sample batch (B, Kmax); the thinker sees one row that concatenates every
sample's token sequence, with position_ids restarting at 0 per sample and no attention_mask. transformers' create_causal_mask
then builds a block-diagonal causal mask (masking_utils: packed sequences are detected from position_ids when attention_mask
is None and there is no cache), so samples never attend to each other. The loss shift is safe at boundaries: every sample
starts with prefix tokens labelled -100, so no target crosses from one sample into the next.

  sampler  PackedBudgetSampler: sort by estimated length, cut consecutive runs with Σ est ≤ max_tokens (and n ≤ max_bs) —
           one packed row per batch, so memory is set by the budget, not by the longest sample × batch size.
           padded=True is the non-packing variant (n × longest ≤ budget) for the normal padded collate.

When packing pays: the thinker attention over a packed row is only cheap with a varlen kernel. With sdpa the packed 4D mask makes
attention O(N_row²) (mostly masked): thinker fwd+bwd alone 2.34 s packed vs 0.25 s padded for a 15.5 k-token KO-like batch (H200,
2026-09-29), i.e. almost all of the 2.36 s/step of the end-to-end pack:16384 sdpa run (2.6× slower than padded bs 16).
flex_attention's packed backward is wrong on torch 2.9 / transformers 4.57 (tests/test_packing.py, speedup.py). With --attn-impl
varlen (torch's bundled FlashAttention-2 varlen, vapasr/hf/varlen_attention.py) attention costs Σ nᵢ² and pad tokens disappear;
padded batches are then unpadded inside the model, so packing and padded + varlen do the same thinker work — they differ only in the
sampler rule (Σ ≤ budget vs n × longest ≤ budget) and the encoder batch size. Long / mixed-length sequences (EN long streams, Phase 2
windows) need varlen (thinker 3–5× faster than padded sdpa). KO main end to end (~283 tokens, 0.4 % padding, H200 2026-09-29):
budget:16384 sdpa 0.39–0.42 s/step → budget:16384@varlen 0.31–0.32 → pack:16384@varlen 0.31 (+22–35 % audio h/GPU h; with 0.4 %
padding the gain presumably comes from sdpa's masked-kernel fallback), and pack:32768@varlen 1,900 audio h/GPU h (+36 % over
budget:32768 sdpa) — experiments/train_speed_bench.py, raw/sources/experiments/2026-09-29-train-speed-bench-varlen-mxc.
  collate  packed(collate_fn): run the normal padded collate, then pack_batch() → ids/labels/... (1, N), position_ids (1, N),
           chunk_of as a global index into the flattened encoder output (B·Kmax), mask dropped.
  model    VapAsrForStreamingASR.forward(position_ids=...) takes the packed path (see modeling_vapasr.py); under varlen it turns
           position_ids into cu_seqlens (varlen_attention.packed_kwargs) so no layer needs a mask.
"""
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

ROW_KEYS = ("ids", "is_audio", "labels", "pos_weight", "labels_alt", "soft_w_full")   # (B, L) tensors concatenated along L


def est_lens(ds) -> np.ndarray:
    """Per-item sequence length estimate (tokens). Datasets with est_lens() (Phase 2 windows) use it; mono/semcommit items use
    prefix + 2 slots per chunk (audio_pad, NEXT_AUDIO) + transcript tokens + flush/tail slack — conservative (over-estimates)."""
    if hasattr(ds, "est_lens"):
        return np.asarray(ds.est_lens(), dtype=np.int64)
    return np.array([24 + 2 * (int(it["K"]) + 4) + len(it.get("tokens") or ()) + int(it.get("n_sem", 0)) for it in ds.items], dtype=np.int64)


class PackedBudgetSampler(Sampler):
    """Token-budget batches (≤ max_bs items), items sorted by estimate so a batch has similar audio lengths (little encoder padding),
    batch order shuffled per epoch (seed+epoch), DDP split like BucketBatchSampler.
      padded=False  Σ est ≤ max_tokens — for packing (one row = the sum).
      padded=True   n × max est ≤ max_tokens — dynamic batch size from the longest item for the ordinary padded collate
                    (short streams get big batches, long ones small; memory set by the budget)."""

    def __init__(self, ds, max_tokens: int, max_bs: int = 64, seed: int = 0, rank: int = 0, world: int = 1, drop_last: bool = False,
                 padded: bool = False):
        est = est_lens(ds)
        order = sorted(range(len(ds)), key=lambda i: (int(est[i]), i))
        cost = (lambda n, tot, e: (n + 1) * e) if padded else (lambda n, tot, e: tot + e)   # ascending order → e is the batch max
        self.batches, cur, tot = [], [], 0
        for i in order:
            e = int(est[i])
            if cur and (cost(len(cur), tot, e) > max_tokens or len(cur) >= max_bs):
                self.batches.append(cur); cur, tot = [], 0
            cur.append(i); tot += e                  # an item longer than max_tokens still gets its own batch
        size = (lambda b: len(b) * int(est[b[-1]])) if padded else (lambda b: int(sum(int(est[i]) for i in b)))
        if cur and not (drop_last and self.batches and size(cur) < max_tokens // 2):
            self.batches.append(cur)
        self.tokens = [size(b) for b in self.batches]
        self.padded = padded
        self.max_tokens, self.seed, self.rank, self.world, self.epoch = max_tokens, seed, rank, world, 0
        self.n = len(self.batches) // world

    def set_epoch(self, e: int): self.epoch = e

    def __iter__(self):
        b = list(self.batches); random.Random(f"{self.seed}:{self.epoch}").shuffle(b)
        return iter(b[self.rank::self.world][: self.n])

    def __len__(self): return self.n

    def describe(self) -> dict:
        bs = [len(b) for b in self.batches]
        return dict(mode="padded" if self.padded else "packed", batches=len(self.batches), per_rank=self.n, max_tokens=self.max_tokens, bs_min=min(bs, default=0),
                    bs_mean=round(sum(bs) / max(1, len(bs)), 1), bs_max=max(bs, default=0),
                    fill=round(sum(self.tokens) / max(1, len(self.tokens)) / self.max_tokens, 3))


def pack_batch(b: dict) -> dict:
    """Padded collate output → one packed row. Encoder inputs (wav/wav_len/K or feats, activity) keep their per-sample shape;
    the model flattens the encoder output to (1, B·Kmax) so chunk_of here is sample·Kmax + chunk."""
    mask = b["mask"].bool(); B = mask.shape[0]; n = mask.sum(1).tolist()
    Kmax = int(b["K"].max()) if "K" in b else int(b["feats"].shape[2])
    out = {k: v for k, v in b.items() if k not in ROW_KEYS + ("mask", "chunk_of")}
    for k in ROW_KEYS:
        if k in b and torch.is_tensor(b[k]) and b[k].dim() == 2:
            out[k] = torch.cat([b[k][i, : n[i]] for i in range(B)])[None]
    ch = b["chunk_of"]
    out["chunk_of"] = torch.cat([torch.where(ch[i, : n[i]] >= 0, ch[i, : n[i]] + i * Kmax, ch[i, : n[i]]) for i in range(B)])[None]
    out["position_ids"] = torch.cat([torch.arange(k, device=mask.device) for k in n])[None]
    if "soft_pos" in b and "soft_b" in b:                       # collate_dialogue's flat soft targets (row, col) → (0, packed col)
        off = torch.tensor([0] + n[:-1], device=b["soft_pos"].device).cumsum(0)
        out["soft_pos"] = b["soft_pos"] + off[b["soft_b"].long()]; out["soft_b"] = torch.zeros_like(b["soft_b"])
    if "activity" in b:                                          # (B, K, R) must line up with the encoder output (B, Kmax) before the model flattens both
        a, am = b["activity"], b.get("activity_mask", torch.ones(b["activity"].shape[:2], device=b["activity"].device))
        if a.shape[1] != Kmax:
            pad = Kmax - a.shape[1]
            a = torch.nn.functional.pad(a, (0, 0, 0, pad)) if pad > 0 else a[:, :Kmax]
            am = torch.nn.functional.pad(am, (0, pad)) if pad > 0 else am[:, :Kmax]
        out["activity"], out["activity_mask"] = a, am
    out["pack_kmax"] = Kmax; out["pack_n"] = B
    return out


def packed(collate_fn: Callable) -> Callable:
    """collate_fn → collate_fn followed by pack_batch (picklable for DataLoader workers)."""
    return _Packed(collate_fn)


class _Packed:
    def __init__(self, fn): self.fn = fn
    def __call__(self, batch): return pack_batch(self.fn(batch))


def packing_sampler_factory(max_tokens: int, max_bs: int = 64, pack: bool = True) -> Optional[Callable]:
    """For loaders that build one sampler per dataset: (ds, seed, rank, world) → PackedBudgetSampler, or None when off.
    pack=True: Σ-budget batches + packed collate. pack=False: padded dynamic batches (n × longest ≤ max_tokens), normal collate.
    The returned factory carries .pack so the loader knows whether to wrap the collate."""
    if not max_tokens:
        return None
    f = lambda ds, seed=0, rank=0, world=1: PackedBudgetSampler(ds, max_tokens, max_bs=max_bs, seed=seed, rank=rank, world=world, padded=not pack)
    f.pack = pack
    return f


def budget_sampler_factory(pack_max_tokens: int = 0, batch_max_tokens: int = 0, max_bs: int = 64) -> Optional[Callable]:
    """--pack-max-tokens (packing) or --batch-max-tokens (padded dynamic batch) → factory; both 0 → None (fixed bs buckets)."""
    if pack_max_tokens and batch_max_tokens:
        raise ValueError("--pack-max-tokens and --batch-max-tokens are exclusive")
    return packing_sampler_factory(pack_max_tokens, max_bs, pack=True) or packing_sampler_factory(batch_max_tokens, max_bs, pack=False)
