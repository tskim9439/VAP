"""G1 이식 머리(2026-10-29, vapasr/hf/n3d_graft): 어댑터(인과·능형 회귀 시작점), 실제 Nemotron 3 Diarization 이 있으면 — layer 0 이식에 교사 h_0 을 넣은 스트리밍이
교사 스트리밍과 같고(캐시 무음 자리 포함), layer m 오프라인 학생에 교사 h_m 을 넣으면 교사 출력과 같다, 길이 다른 배치 줄 = 단독. CPU."""
import importlib.util, os

import numpy as np
import pytest
import torch

from vapasr.hf.n3d_graft import GraftAdapter


def test_adapter_starts_linear_and_is_causal():
    torch.manual_seed(0); B, S, K, D = 2, 3, 40, 8
    ad = GraftAdapter(S, D, 6, blocks=2, kernel=4, ffn=12); f = torch.randn(B, S, K, D)
    W, b = torch.randn(S * D, 6), torch.randn(6); ad.set_linear(W, b)
    assert torch.allclose(ad(f), f.permute(0, 2, 1, 3).reshape(B, K, S * D) @ W + b, atol=1e-5)          # 블록 출력 0 → 붙인 직후 = 능형 회귀 선형 이식
    with torch.no_grad():
        for p in ad.blocks.parameters(): p.add_(torch.randn_like(p) * 0.3)
    y = ad(f); f2 = f.clone(); t = 23; f2[:, :, t:] += 3.0; y2 = ad(f2)
    assert torch.allclose(y[:, :t], y2[:, :t], atol=1e-5) and not torch.allclose(y[:, t:], y2[:, t:])           # 인과 — t 이후를 바꿔도 앞은 같다


_N3D = os.environ.get("VAPASR_N3D_PATH", "")
real = pytest.mark.skipif(not (_N3D and os.path.exists(_N3D) and importlib.util.find_spec("nemo") is not None), reason="VAPASR_N3D_PATH(.nemo)·NeMo 가 있을 때만")


def _signal(sec=10.0, seed=2):
    g = torch.Generator().manual_seed(seed); x = (torch.randn(int(16000 * sec), generator=g) * 0.05).float()
    t = torch.arange(len(x)) / 16000.0; n = int(16000 * sec * 0.45); x[:n] += 0.3 * torch.sin(2 * np.pi * 200 * t[:n]) * (torch.sin(2 * np.pi * 2 * t[:n]) > 0)
    x[-n:] += 0.25 * torch.sin(2 * np.pi * 330 * t[-n:]); return x


@real
def test_layer0_graft_stream_equals_teacher_stream():
    from vapasr.hf.n3d import load_n3d, stream_probs
    from vapasr.hf.n3d_graft import stream
    m, _ = load_n3d(_N3D); x = _signal(); w = x[None]; wl = torch.tensor([len(x)])
    with torch.no_grad():
        feats, flen = m.preprocessor(input_signal=w, length=wl); pre, plen = m.encoder.pre_encode(feats[:, :, : int(flen.max())], flen)
        h0 = m.encoder.embed_norm(pre); sil = m.encoder.embed_norm(m.sortformer_modules.learnable_sil_emb.data[None])[0]
    for ch in (4, 16):
        pt = stream_probs(m, w, wl, ch, 0, dtype="fp32"); pg = stream(m, h0, plen.long(), 0, sil, chunk=ch)
        T = min(pt.shape[1], pg.shape[1]); assert T >= pt.shape[1] - 1
        assert torch.allclose(pt[:, :T], pg[:, :T], atol=1e-4), (ch, float((pt[:, :T] - pg[:, :T]).abs().max()))
    hb = torch.zeros(2, h0.shape[1], h0.shape[2]); hb[0] = h0[0]; hb[1, :60] = h0[0, :60]                  # 길이 다른 배치 줄 = 단독
    pb = stream(m, hb, torch.tensor([int(plen[0]), 60]), 0, sil, chunk=4); ps = stream(m, h0[:, :60], torch.tensor([60]), 0, sil, chunk=4)
    assert torch.allclose(pb[1, :60], ps[0], atol=1e-5) and not pb[1, 60:].any() and torch.allclose(pb[0], stream(m, h0, plen.long(), 0, sil, chunk=4)[0], atol=1e-5)


@real
def test_offline_student_with_teacher_hidden_equals_teacher():
    from vapasr.hf.n3d import load_n3d
    from vapasr.hf.n3d_graft import pool80, student_offline, teacher
    m, _ = load_n3d(_N3D); x = _signal(8.0, 5); W = torch.zeros(2, len(x)); W[0] = x; W[1, : 16000 * 5] = x[: 16000 * 5]; wl = torch.tensor([len(x), 16000 * 5])
    for layer in (0, 8):
        h, p10, elen = teacher(m, W, wl, layer)
        with torch.no_grad(): ps = student_offline(m, h, elen.long(), layer)
        for j in range(2):
            n = int(elen[j]) * 8; assert torch.allclose(ps[j, :n], p10[j, :n], atol=1e-4), (layer, j, float((ps[j, :n] - p10[j, :n]).abs().max()))
    assert pool80(p10, elen.long()).shape[1] == p10.shape[1] // 8
