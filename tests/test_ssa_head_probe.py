"""보조 헤드 진단(experiments/ssa_head_probe.py) — 헤드의 시작 항등·활동 조건·앞보기 한계·채움 불변. CPU, 모델·오디오 없이."""
import os, sys
import torch
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "experiments")]
import ssa_head_probe as H


def mk(layers=2, left=4, right=1):
    torch.manual_seed(0); return H.build_head(d=16, layers=layers, heads=4, ff=32, left=left, right=right, p=0.0, kp=0.0).eval()


def randomize(h):
    torch.manual_seed(1)
    for n, p in h.named_parameters():
        if p.dim() > 1 or "rel" in n: torch.nn.init.normal_(p, std=0.3)


def test_head_starts_as_identity():
    h = mk(); z = torch.randn(2, 9, 16); lens = torch.tensor([9, 6])
    out = h(z, lens, torch.ones(2, 9), torch.zeros(2, 9))
    assert torch.allclose(out, z, atol=1e-6)                                                  # 커널·층의 마지막 투영이 0 → 시작 = 인코더 출력 그대로


def test_target_activity_changes_output():
    h = mk(); randomize(h); z = torch.randn(1, 9, 16); lens = torch.tensor([9])
    a = h(z, lens, torch.ones(1, 9), torch.zeros(1, 9)); b = h(z, lens, torch.zeros(1, 9), torch.ones(1, 9))
    assert not torch.allclose(a, b, atol=1e-4)                                                # 대상·배경 활동을 바꾸면 출력이 달라진다
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_lookahead_is_limited_to_layers_times_right():
    h = mk(layers=2, left=4, right=1); randomize(h); T = 14; z = torch.randn(1, T, 16); lens = torch.tensor([T])
    mt, mb = torch.ones(1, T), torch.zeros(1, T)
    a = h(z, lens, mt, mb); z2 = z.clone(); z2[0, 10] += 5.0; b = h(z2, lens, mt, mb)
    changed = (a - b).abs().amax(-1)[0] > 1e-5
    assert not changed[:8].any()                                                               # 프레임 10 의 변화는 10 − 2 층 × 1 = 8 보다 앞 프레임에 닿지 않는다
    assert changed[8:11].all()


def test_padding_does_not_change_valid_frames():
    h = mk(); randomize(h); z = torch.randn(1, 6, 16)
    alone = h(z, torch.tensor([6]), torch.ones(1, 6), torch.zeros(1, 6))
    zp = torch.cat([z, torch.randn(1, 5, 16) * 9], 1); both = torch.cat([zp, torch.randn(1, 11, 16)], 0)
    mt = torch.cat([torch.ones(1, 6), torch.ones(1, 5)], 1).repeat(2, 1); mb = torch.zeros(2, 11)
    out = h(both, torch.tensor([6, 11]), mt, mb)
    assert torch.isfinite(out).all()                                                           # 채움 프레임 행도 NaN 이 아니다
    assert torch.allclose(out[0, :6], alone[0], atol=1e-5)                                     # 채움이 유효 프레임 출력을 바꾸지 않는다


def test_same_across_ranks_detects_difference():
    class Dist:                                                                               # all_reduce(MAX) 를 두 rank 흉내
        class ReduceOp: MAX = "max"
        def __init__(self, other): self.other = other
        def all_reduce(self, t, op=None): t.copy_(torch.maximum(t, self.other))
    p = [torch.nn.Parameter(torch.ones(3))]; c = torch.tensor([3.0, -3.0], dtype=torch.double)
    assert H.same_across_ranks(p, Dist(c))[0]
    assert not H.same_across_ranks(p, Dist(torch.tensor([4.0, -4.0], dtype=torch.double)))[0]
