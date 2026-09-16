"""Phase 2 손실 함수 — soft CE 의 두 점 혼합·가중, 활동 BCE·gather."""
import torch, pytest
from vapasr.hf.p2_losses import soft_ce, activity_bce, gather_audio_targets

def test_soft_ce_mixes_two_targets():
    torch.manual_seed(0); V = 10; logits = torch.randn(5, V); tgt = torch.tensor([1, 2, 3, 4, 5]); NEXT, EOT = 9, 8
    base, st = soft_ce(logits, tgt, None, None, NEXT, 0.3)
    ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none"); assert torch.allclose(base, ce.mean()) and st["n_soft"] == 0
    alt = torch.tensor([-100, 7, -100, -100, -100]); w = torch.tensor([1.0, 0.25, 1.0, 1.0, 1.0])
    mixed, st = soft_ce(logits, tgt, alt, w, NEXT, 0.3)
    ce_alt = torch.nn.functional.cross_entropy(logits[1:2], alt[1:2], reduction="none")[0]
    expect = ce.clone(); expect[1] = 0.25 * ce[1] + 0.75 * ce_alt
    assert torch.allclose(mixed, expect.mean()) and st["n_soft"] == 1

def test_soft_ce_weights_next_and_eot():
    torch.manual_seed(1); logits = torch.randn(4, 12); NEXT, EOT = 11, 10; tgt = torch.tensor([3, NEXT, EOT, 4])
    loss, st = soft_ce(logits, tgt, None, None, NEXT, 0.3, eot_id=EOT, eot_weight=2.0)
    ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="none"); wt = torch.tensor([1.0, 0.3, 2.0, 1.0])
    assert torch.allclose(loss, (ce * wt).sum() / wt.sum()) and st["n_eot"] == 1 and st["loss_eot"] == pytest.approx(ce[2].item(), rel=1e-5)

def test_activity_gather_and_bce():
    B, L, D, K, R = 2, 6, 4, 3, 2; h = torch.randn(B, L, D)
    is_audio = torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0]], dtype=torch.bool); chunk_of = torch.tensor([[0, -1, 1, -1, 2, -1], [0, -1, 1, -1, -1, -1]])
    activity = torch.zeros(B, K, R); activity[0, :, 0] = 1; activity[1, 1, 1] = 1; mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float)
    ha, tg = gather_audio_targets(h, is_audio, chunk_of, activity, mask)
    assert ha.shape == (5, D) and tg.shape == (5, R) and tg[:3, 0].sum() == 3 and tg[4, 1] == 1
    logits = torch.where(tg > 0.5, torch.full_like(tg, 5.0), torch.full_like(tg, -5.0)); loss, st = activity_bce(logits, tg)
    assert loss.item() < 0.01 and st["act_acc"] == 1.0
    empty, st0 = activity_bce(torch.zeros(0, R), torch.zeros(0, R)); assert empty.item() == 0.0
