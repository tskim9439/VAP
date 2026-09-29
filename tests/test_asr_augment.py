"""ASR 정확도 개선 학습 옵션(2026-09-29) — 모델 쪽: SpecAugment(vapasr/features/online.py), sem_mask 손실(vapasr/hf/modeling_vapasr.py)·packing 행 키. CPU."""
import importlib.util, math, pathlib
import pytest, torch
from vapasr.features.online import SpecAugment
from vapasr.hf.packing import ROW_KEYS

_HERE = pathlib.Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# ── SpecAugment
def test_spec_augment_masks_only_in_training_within_valid_frames():
    torch.manual_seed(0); f = torch.randn(3, 128, 200) + 5.0; fl = torch.tensor([200, 120, 50])
    sa = SpecAugment(2, 27, 5, 0.05)
    sa.eval(); assert torch.equal(sa(f, fl), f)                                                # 평가·추론은 그대로
    sa.train(); g = sa(f, fl); z = g == 0
    assert g.shape == f.shape and ((g == f) | z).all()                                       # 값은 그대로이거나 마스크 0
    col, row = z.all(1), z.all(2)                                                            # 시간 마스크 = 열 전체, 주파수 마스크 = 행 전체
    for b, L in enumerate(fl.tolist()):
        assert not col[b, L:].any() and int(col[b].sum()) <= 5 * math.floor(0.05 * L)          # 유효 프레임 안, 마스크당 ≤ ⌊0.05·L⌋
    assert (row.sum(1) <= 2 * 27).all() and z.any()
    torch.manual_seed(1); a1 = sa(f, fl); torch.manual_seed(1); a2 = sa(f, fl); assert torch.equal(a1, a2)   # 학습 시드를 따른다
    assert SpecAugment.from_spec("off") is None and SpecAugment.from_spec("nemo").time_masks == 10
    assert (SpecAugment.from_spec("light").time_masks, SpecAugment.from_spec("2,10,3,0.1").freq_width) == (5, 10)


# ── 손실: sem_mask 위치는 softmax 에서 <SEM_END> 를 뺀다
def test_sem_mask_removes_sem_from_softmax_and_rejects_sem_targets():
    tp = _load("test_packing"); m = tp._tiny("sdpa"); sem = 151723; m.config.sem_registry = {"<SEM_END>": sem}
    x = tp._xs(tp._batch([19, 31], [6, 12], seed=3)); assert not (x["labels"] == sem).any()
    with torch.no_grad(): l0 = float(m(**x).loss)
    sm = torch.zeros_like(x["labels"], dtype=torch.bool); sm[0] = True
    with torch.no_grad(): l1 = float(m(**x, sem_mask=sm).loss)
    assert l1 < l0                                                                            # SEM 확률 몫이 빠져 다른 타깃 CE 가 줄었다
    m.zero_grad(); m(**x, sem_mask=torch.ones_like(sm)).loss.backward()
    g = m.thinker.lm_head.weight.grad; tgt = int(x["labels"][x["labels"] >= 0][0])
    assert g[sem].abs().max() == 0 and g[tgt].abs().max() > 0                                # 모든 위치를 가리면 SEM 행 기울기 0
    lab = x["labels"].clone(); p = int((lab[0] != -100).nonzero()[0]); lab[0, p] = sem
    with pytest.raises(AssertionError, match="sem_mask"):
        m(**dict(x, labels=lab), sem_mask=sm)


def test_sem_mask_is_a_packed_row_key():
    assert "sem_mask" in ROW_KEYS                                                           # pack_batch 가 (B, L) → (1, N) 으로 잇는다
