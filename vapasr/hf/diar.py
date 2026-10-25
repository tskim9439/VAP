"""화자 분리 가지(E1, 2026-10-25 — wiki output-diarization-integration-20261025): 인코더 여러 층 출력의 학습 가중합 → Transformer → 슬롯 활동 sigmoid.
Sortformer(arXiv:2409.06656)·Streaming Sortformer(arXiv:2507.18446)와 같은 출력 규약이다. 슬롯 0 = 먼저 말한 화자(도착 순서)이고, 손실은 도착 순서 BCE(Sort loss)와
순열 불변 BCE(PIT)의 가중합이다. 우리 대화 슬롯 규칙(cst_dialogue.window_speakers — 창 안에서 먼저 말한 사람 = A)과 같은 순서라 라벨은 slot_activity 그대로 쓴다.
- DiarBranch: (B, S, K, D) 인코더 스트림(model.encode / NemotronOnline taps) → (B, K, N) logits. 위치 정보는 깊이별 conv(길이와 무관, wav2vec2 식).
- sort_pit_loss: 프레임 마스크(패딩·전사 없는 음성)를 뺀 BCE.
- der_counts: 80 ms 프레임 DER 셈(collar 0) — 도착 순서 그대로의 짝과 혼동이 가장 적은 짝 둘 다.
- load_encoder: VapAsr 체크포인트(조각 저장 포함)에서 인코더(encoder.enc.*)만 읽어 NemotronOnline 을 만든다(동결·fp32).
"""
import itertools, json, os
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPos(nn.Module):
    def __init__(self, d: int, k: int = 31, groups: int = 16):
        super().__init__(); self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=groups)

    def forward(self, x):                                                                      # (B, K, d)
        return x + F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))


class DiarBranch(nn.Module):
    def __init__(self, n_streams: int, d_in: int = 1024, d: int = 512, layers: int = 12, heads: int = 8, ffn: int = 2048, n_spk: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cfg = dict(n_streams=n_streams, d_in=d_in, d=d, layers=layers, heads=heads, ffn=ffn, n_spk=n_spk, dropout=dropout)
        self.mix = nn.Parameter(torch.zeros(n_streams)); self.ln = nn.LayerNorm(d_in); self.proj = nn.Linear(d_in, d); self.pos = ConvPos(d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, n_spk)

    def forward(self, feats, pad_mask: Optional[torch.Tensor] = None):
        """feats (B, S, K, D_in) → logits (B, K, N). pad_mask (B, K) True = 패딩 프레임(어텐션에서 뺀다)."""
        w = torch.softmax(self.mix, 0).to(feats.dtype); x = self.proj(self.ln((feats * w[None, :, None, None]).sum(1)))
        if pad_mask is not None: x = x.masked_fill(pad_mask[..., None], 0.0)                     # conv 전에 — 끝 패딩이 0 채움과 같아진다
        x = self.pos(x)
        return self.out(self.norm(self.tf(x, src_key_padding_mask=pad_mask)))


def sort_pit_loss(logits, target, mask, alpha: float = 0.5):
    """logits·target (B, K, N), mask (B, K) 1 = 손실에 넣는 프레임 → (loss, sort, pit). 둘 다 (마스크 프레임 × N) 평균 BCE.
    sort = 도착 순서 그대로, pit = 창마다 가장 작은 슬롯 순열. loss = α·sort + (1 − α)·pit."""
    m = mask.float(); n = logits.shape[-1]; den = (m.sum(1) * n).clamp(min=1)                   # (B,)
    pair = F.binary_cross_entropy_with_logits(logits[..., :, None].expand(-1, -1, -1, n), target[..., None, :].expand(-1, -1, n, -1).float(), reduction="none")
    C = (pair * m[..., None, None]).sum(1)                                                     # (B, N 예측, N 정답)
    sort = C.diagonal(dim1=1, dim2=2).sum(-1) / den
    perms = torch.tensor(list(itertools.permutations(range(n))), device=logits.device)           # (P, N)
    pc = C[:, torch.arange(n, device=logits.device)[None, :], perms].sum(-1)                    # (B, P): Σ_i C[i, p(i)]
    pit = pc.min(-1).values / den
    return alpha * sort.mean() + (1 - alpha) * pit.mean(), sort.mean().detach(), pit.mean().detach()


def der_counts(pred, ref, mask=None) -> dict:
    """pred·ref (K, N) 0/1 텐서, mask (K,) 1 = 셈에 넣는 프레임 → dict(ref, miss, fa, conf_arrival, conf_best). 단위는 프레임 × 화자.
    DER = (miss + fa + conf) / ref. arrival = 슬롯 순서 그대로, best = 혼동이 가장 적은 슬롯 순열(miss·fa 는 순열과 무관)."""
    p, r = pred.float(), ref.float()
    if mask is not None: keep = mask.bool(); p, r = p[keep], r[keep]
    n_ref, n_hyp = r.sum(-1), p.sum(-1)
    miss = (n_ref - n_hyp).clamp(min=0).sum(); fa = (n_hyp - n_ref).clamp(min=0).sum(); both = torch.minimum(n_ref, n_hyp).sum()
    N = r.shape[-1]; corr = lambda perm: (r * p[:, list(perm)]).sum()
    best = max(corr(q) for q in itertools.permutations(range(N)))
    return dict(ref=float(n_ref.sum()), miss=float(miss), fa=float(fa), conf_arrival=float(both - corr(range(N))), conf_best=float(both - best))


def load_encoder(ckpt: str, nemotron_dir: Optional[str] = None, device: str = "cuda"):
    """VapAsr 체크포인트 디렉토리(model.safetensors 또는 조각 + index)에서 encoder.enc.* 만 읽어 NemotronOnline 을 만든다(동결·eval). right_context 는 config 의 값."""
    from safetensors import safe_open
    from ..features.online import NemotronOnline
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    sd = {}
    for f in files:
        with safe_open(os.path.join(ckpt, f), "pt") as h:
            for k in h.keys():
                if k.startswith("encoder.enc."): sd[k[len("encoder.enc."):]] = h.get_tensor(k)
    assert sd, f"인코더 가중치(encoder.enc.*)가 없다: {ckpt}"
    enc = NemotronOnline(right_context=int(cfg.get("encoder_right_context", 0) or 0), path=nemotron_dir)
    missing, unexpected = enc.enc.load_state_dict(sd, strict=False)
    params = {n for n, _ in enc.enc.named_parameters()}
    assert not (set(missing) & params) and not unexpected, f"인코더 키 불일치: 빠짐 {sorted(set(missing) & params)[:5]} · 남음 {unexpected[:5]}"
    enc.set_trainable(False)
    return enc.to(device).eval(), len(sd)


def tap_layers(spec: str, n_layers: int) -> Sequence[int]:
    """'1,3,5' 또는 'all' 또는 'odd' → 층 번호(0 부터, 마지막 층은 스트림 0 으로 늘 들어간다)."""
    if spec == "all": return tuple(range(n_layers - 1))
    if spec == "odd": return tuple(range(1, n_layers - 1, 2))
    t = tuple(int(x) for x in spec.split(",") if x != "")
    assert all(0 <= j < n_layers for j in t), f"탭 층 0–{n_layers - 1}: {t}"
    return t
