"""고정 용량 probe head. 인코더가 달라도 입력 projection(D→d_model)만 다르다.

feats (B, 2, T, D) → 채널별 공유 projection + 화자 임베딩 → 두 화자 프레임을 concat(특징 축) → causal Transformer(d_model, n_layers)
→ vap logits (B, T, 256), vad logits (B, T, 2). causal mask 로 probe 가 lookahead 를 추가하지 않게 한다.
"""
import torch, torch.nn as nn
from ..data.targets import vap_bin_times

class ProbeHead(nn.Module):
    def __init__(self, d_in: int, frame_hz: float, d_model: int = 256, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.frame_hz = frame_hz; self.d_model = d_model
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model))
        self.spk = nn.Parameter(torch.zeros(2, d_model)); nn.init.normal_(self.spk, std=0.02)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model); self.vap_out = nn.Linear(d_model, 256); self.vad_out = nn.Linear(d_model, 2)
        self._obj = None

    def forward(self, feats):                                   # (B, 2, T, D)
        B, _, T, _ = feats.shape
        h = self.proj(feats) + self.spk[None, :, None, :]        # (B, 2, T, d)
        h = self.merge(torch.cat([h[:, 0], h[:, 1]], -1))         # (B, T, d)
        h = h + self.pos(torch.arange(T, device=h.device))[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), 1)
        h = self.norm(self.enc(h, mask=mask, is_causal=True))
        return self.vap_out(h), self.vad_out(h)

    def objective(self):
        if self._obj is None:
            from vap.objective import ObjectiveVAP
            self._obj = ObjectiveVAP(bin_times=vap_bin_times(self.frame_hz), frame_hz=self.frame_hz)
        return self._obj

    @torch.inference_mode()
    def probs(self, feats):
        """→ dict(p_now (B,T,2), p_future (B,T,2), probs (B,T,256), vad (B,T,2))  — 원 VAP 의 get_probs 재사용."""
        vap_logits, vad_logits = self.forward(feats); out = self.objective().get_probs(vap_logits.float().cpu())
        out["vad"] = torch.sigmoid(vad_logits).cpu(); return out

    def n_params(self, include_proj=False):
        return sum(p.numel() for n, p in self.named_parameters() if include_proj or not n.startswith("proj"))
