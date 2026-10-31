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


def test_encoder_taps_segments_equal_whole():
    """긴 한 줄은 겹침 조각으로 — 프레임마다 그 프레임 오디오만 보는 가짜 인코더면 조각 = 통째(조각 경계·자르기 점검)."""
    from vapasr.hf.n3d_graft import encoder_taps

    class Enc:
        def __call__(self, wav, wav_len, K, taps=()):
            Kk = int(K.max()); x = torch.nn.functional.pad(wav, (0, max(0, Kk * 1280 - wav.shape[1])))[:, : Kk * 1280].reshape(wav.shape[0], Kk, 1280)
            base = torch.stack([x.mean(-1), x.std(-1), x[..., 0]], -1)                                   # (B, K, 3)
            return torch.stack([base * (j + 1) for j in range(1 + len(taps))], 1)                           # (B, 1 + 탭, K, 3)
    g = torch.Generator().manual_seed(0); K = 130; w = torch.randn(1, K * 1280, generator=g)
    whole = Enc()(w, torch.tensor([w.shape[1]]), torch.tensor([K]), taps=(1, 3))
    seg = encoder_taps(Enc(), w, torch.tensor([w.shape[1]]), torch.tensor([K]), (1, 3), ch=4, core=32)
    assert seg.shape == whole.shape and torch.allclose(seg, whole, atol=1e-6)


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


@real
def test_stream_grad_matches_values_and_reaches_cached_frames():
    """G1b: grad=True 스트리밍은 grad=False 와 값이 같고, 뒤 청크의 손실이 캐시·FIFO 에 담긴 앞 청크 ĥ 까지 gradient 를 보낸다."""
    from vapasr.hf.n3d import load_n3d
    from vapasr.hf.n3d_graft import stream, teacher
    m, _ = load_n3d(_N3D); x = _signal(9.0, 7); w = x[None]; wl = torch.tensor([len(x)])
    h, _, elen = teacher(m, w, wl, 8); sil = h[0].mean(0).detach()
    p0 = stream(m, h, elen.long(), 8, sil, chunk=16)
    hg = h.detach().clone().requires_grad_(True); p1 = stream(m, hg, elen.long(), 8, sil, chunk=16, grad=True)
    assert torch.allclose(p0, p1.detach(), atol=1e-5)
    T = p1.shape[1]; p1[:, T - 16:].sum().backward()                                                   # 마지막 청크만의 손실
    g = hg.grad[0].abs().sum(-1); assert g[: T - 16].sum() > 0 and g[T - 16:].sum() > 0                      # 앞 프레임(캐시·FIFO)도 gradient 를 받는다


def test_partial_unfreeze_only_lower_layers():
    """U3: 아래 n 층만 학습(나머지·서브샘플링 고정), 모듈은 eval, forward 는 trainable 경로(NemotronOnline.trainable), Triton 서브샘플링은 끈다."""
    from vapasr.hf.n3d_graft import partial_unfreeze

    class Conv(torch.nn.Module):
        def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.ones(2)); self.fuse_triton = True

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.pre_encode = torch.nn.Module(); self.pre_encode.conv = Conv()
            self.layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5)) for _ in range(6)])

    class Enc(torch.nn.Module):
        def __init__(self): super().__init__(); self.enc = Inner(); self.trainable = False
    e = Enc(); e.train(); ps = partial_unfreeze(e, 4)
    assert e.trainable and not e.training and not e.enc.layers[0].training and e.enc.pre_encode.conv.fuse_triton is False
    assert {id(p) for p in ps} == {id(p) for l in e.enc.layers[:4] for p in l.parameters()} and all(p.requires_grad for p in ps)
    assert not any(p.requires_grad for l in e.enc.layers[4:] for p in l.parameters()) and not e.enc.pre_encode.conv.w.requires_grad
    with pytest.raises(AssertionError): partial_unfreeze(Enc(), 7)


@real
def test_stream_ckpt_same_values_and_grads():
    """U3: 걸음마다 활성값을 다시 계산하는 스트리밍(ckpt)이 그냥 grad 스트리밍과 값·gradient 가 같다(run_layers 의 SDPA 길이를 다시 적는다)."""
    from vapasr.hf.n3d import load_n3d
    from vapasr.hf.n3d_graft import stream, teacher
    m, _ = load_n3d(_N3D); x = _signal(9.0, 11); W = torch.zeros(2, len(x)); W[0] = x; W[1, : 16000 * 6] = x[: 16000 * 6]; wl = torch.tensor([len(x), 16000 * 6])
    h, _, elen = teacher(m, W, wl, 8); sil = h[0].mean(0).detach(); L = elen.long()
    outs, grads = [], []
    for ck in (False, True):
        hg = h.detach().clone().requires_grad_(True); p = stream(m, hg, L, 8, sil, chunk=16, grad=True, ckpt=ck)
        (p * torch.linspace(0.5, 1.5, p.shape[1])[None, :, None]).sum().backward(); outs.append(p.detach()); grads.append(hg.grad.clone())
    assert torch.allclose(outs[0], outs[1], atol=1e-6) and torch.allclose(grads[0], grads[1], atol=1e-5), float((grads[0] - grads[1]).abs().max())


@real
def test_g2_lora_starts_equal_and_trains_upper_only():
    """G2: 학생 N3D 사본에 LoRA(B 0)를 붙인 직후 오프라인·스트리밍 출력이 교사와 같고, 학습 대상은 start 층부터 위의 A·B 뿐이다.
    스트리밍 grad 에서 B 가 gradient 를 받고(A 는 B 가 0 이라 0), B 를 흔들면 출력이 달라진다. 다시 불러도 같은 텐서다."""
    from vapasr.hf.n3d import load_n3d
    from vapasr.hf.n3d_graft import LORA_TARGETS, n3d_trainable, stream, student_offline, teacher
    m, _ = load_n3d(_N3D); s, _ = load_n3d(_N3D); x = _signal(9.0, 13); w = x[None]; wl = torch.tensor([len(x)])
    h, _, elen = teacher(m, w, wl, 8); L = elen.long(); sil = h[0].mean(0).detach()
    cfg = dict(mode="lora", start=8, r=4, alpha=8.0, head=False); named = n3d_trainable(s, cfg)
    assert len(named) == 2 * len(LORA_TARGETS) * (len(s.encoder.layers) - 8)
    assert all(int(n.split(".")[2]) >= 8 and n.endswith((".A", ".B")) for n, _ in named) and not any(p.requires_grad for p in m.parameters())
    with torch.no_grad():
        assert torch.allclose(student_offline(s, h, L, 8), student_offline(m, h, L, 8), atol=1e-5)
        assert torch.allclose(stream(s, h, L, 8, sil, chunk=16), stream(m, h, L, 8, sil, chunk=16), atol=1e-5)
    p = stream(s, h, L, 8, sil, chunk=16, grad=True); (p * torch.linspace(0.5, 1.5, p.shape[1])[None, :, None]).sum().backward()
    assert sum(float(q.grad.abs().sum()) for n, q in named if n.endswith(".B")) > 0 and all(float(q.grad.abs().sum()) == 0 for n, q in named if n.endswith(".A"))
    with torch.no_grad():
        for n, q in named:
            if n.endswith(".B"): q.normal_(0, 0.02)
        assert not torch.allclose(student_offline(s, h, L, 8), student_offline(m, h, L, 8), atol=1e-4)
    assert [id(q) for _, q in n3d_trainable(s, cfg)] == [id(q) for _, q in named] and all(q.requires_grad for _, q in named)


@real
def test_g2_state_roundtrip_head_and_full_sets():
    """G2: 학습한 값(이름 → 텐서)을 새 사본에 load_n3d_trainable 로 넣으면 출력이 같고, 이름이 다르면 막는다. head 는 final_norm·머리(캐시 무음 자리 빼고),
    full 은 start 층부터 위의 파라미터 전부다."""
    from vapasr.hf.n3d import load_n3d
    from vapasr.hf.n3d_graft import load_n3d_trainable, n3d_trainable, student_offline, teacher
    s, _ = load_n3d(_N3D); x = _signal(6.0, 17); w = x[None]; wl = torch.tensor([len(x)])
    h, _, elen = teacher(s, w, wl, 8); L = elen.long()
    cfg = dict(mode="lora", start=8, r=4, alpha=8.0, head=True); named = n3d_trainable(s, cfg); names = [n for n, _ in named]
    assert any(n.startswith("sortformer_modules.") for n in names) and "encoder.final_norm.weight" in names and not any("learnable_sil_emb" in n for n in names)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for _, q in named: q.add_(torch.randn(q.shape, generator=g) * 0.01)
    state = {n: q.detach().clone() for n, q in named}
    s2, _ = load_n3d(_N3D); named2 = load_n3d_trainable(s2, cfg, state)
    assert [n for n, _ in named2] == names
    with torch.no_grad(): assert torch.allclose(student_offline(s, h, L, 8), student_offline(s2, h, L, 8), atol=1e-6)
    with pytest.raises(AssertionError): load_n3d_trainable(load_n3d(_N3D)[0], dict(cfg, start=9), state)
    f, _ = load_n3d(_N3D); fn = n3d_trainable(f, dict(mode="full", start=28, r=4, alpha=8.0, head=False))
    assert {n for n, _ in fn} == {f"encoder.layers.{j}.{k}" for j in (28, 29, 30) for k, _ in f.encoder.layers[j].named_parameters()}
