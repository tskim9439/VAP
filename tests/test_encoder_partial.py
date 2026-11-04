"""E15a(2026-11-04): NemotronOnline.set_trainable(layers=(lo, hi)) — 그 층만 gradient·train, 서브샘플링·다른 층은 고정·eval(dropout·BatchNorm 통계 고정).
모델 train()/eval() 을 거쳐도 그대로이고, 전체 학습·고정(layers 없음)은 예전과 같다. CPU, NeMo 없이 작은 대역 인코더로."""
import pytest, torch, torch.nn as nn
from vapasr.features.online import NemotronOnline


class _Enc(nn.Module):
    def __init__(self, n=6):
        super().__init__(); self.pre_encode = nn.Linear(4, 4)
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5)) for _ in range(n))


def _online(n=6):
    m = NemotronOnline.__new__(NemotronOnline); nn.Module.__init__(m); m.enc = _Enc(n); m.spec_aug = None; m.set_trainable(False); return m


def test_partial_layers_grad_and_mode():
    m = _online(); m.set_trainable(True, layers=(3, 6))
    assert [all(p.requires_grad for p in l.parameters()) for l in m.enc.layers] == [False] * 3 + [True] * 3
    assert not any(p.requires_grad for p in m.enc.pre_encode.parameters()) and m.train_layers == (3, 6)
    m.train(); assert [l.training for l in m.enc.layers] == [False] * 3 + [True] * 3 and not m.enc.pre_encode.training
    m.eval(); assert not any(l.training for l in m.enc.layers)
    m.train(); assert [l.training for l in m.enc.layers] == [False] * 3 + [True] * 3
    bn = m.enc.layers[0][1]; before = bn.running_mean.clone(); m.enc.layers[0](torch.randn(8, 4))
    assert torch.equal(before, bn.running_mean)                                                   # 고정 층의 BatchNorm 통계는 그대로
    bn = m.enc.layers[4][1]; before = bn.running_mean.clone(); m.enc.layers[4](torch.randn(8, 4))
    assert not torch.equal(before, bn.running_mean)                                               # 학습 층은 train


def test_full_and_frozen_unchanged():
    m = _online(); m.set_trainable(True); m.train()
    assert m.train_layers is None and all(p.requires_grad for p in m.enc.parameters()) and all(l.training for l in m.enc.layers)
    m.set_trainable(False); m.train()
    assert not any(p.requires_grad for p in m.enc.parameters()) and not m.enc.training and not any(l.training for l in m.enc.layers)
    m.set_trainable(False, layers=(1, 2)); assert m.train_layers is None and not any(p.requires_grad for p in m.enc.parameters())   # 고정이면 층 범위는 무시
    with pytest.raises(AssertionError, match="학습 층"): m.set_trainable(True, layers=(4, 9))
