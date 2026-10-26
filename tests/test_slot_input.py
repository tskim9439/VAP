"""E5 슬롯 활동 입력(2026-10-26): act_noise(정답 활동 흔들기)·모델 adapter_slot(0 초기화 → 그대로, 켜면 오디오 토큰 += Σ a_s·e_s)·데이터셋 act_in·collate. CPU."""
import random

import numpy as np
import pytest
import torch

from vapasr.data.act_noise import ActNoise, from_spec, _runs


def _act(K=200, S=4):
    a = np.zeros((K, S), np.float32); a[10:80, 0] = 1; a[60:140, 1] = 1; a[150:156, 2] = 1                    # 0·1 겹침 60–80, 2 는 말이 적다(3 %)
    return a


def test_act_noise_clean_and_drop_and_values():
    a = _act()
    assert np.array_equal(ActNoise(clean=1.0)(a, random.Random(0)), a)
    d = ActNoise(clean=0.0, drop=0.0, short=1.0, jitter=0, ovl=0.0, fa=0.0)(a, random.Random(0))
    assert not d[:, 2].any() and np.array_equal(d[:, :2], a[:, :2])                                         # 말 적은 화자만 통째로
    for seed in range(20):
        y = ActNoise(clean=0.0)(a, random.Random(seed)); assert y.shape == a.shape and set(np.unique(y)) <= {0.0, 1.0}
    j = ActNoise(clean=0.0, drop=0.0, short=0.0, jitter=3, ovl=0.0, fa=0.0)(a, random.Random(1))
    for s in (0, 1):
        (i0, j0), = _runs(a[:, s]); (i1, j1), = _runs(j[:, s]); assert abs(i1 - i0) <= 3 and abs(j1 - j0) <= 3   # 구간 끝만 흔들린다
    o = ActNoise(clean=0.0, drop=0.0, short=0.0, jitter=0, ovl=1.0, fa=0.0)(a, random.Random(2))
    assert not ((o[60:80, 0] > 0) & (o[60:80, 1] > 0)).any()                                                  # 겹침 구간에서 한 슬롯을 끈다
    assert from_spec("").settings() == ActNoise().settings() and from_spec("clean=1").clean == 1.0
    with pytest.raises(ValueError): from_spec("nope=1")


def _tiny():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("_tl", pathlib.Path(__file__).parent / "test_semcommit_tokens_loss.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.tiny_model()


def test_model_slot_input_zero_init_and_effect():
    m, _ = _tiny(); assert m.adapter_slot is None
    feats = torch.randn(2, 1, 7, m.config.adapter_d_in); act = torch.zeros(2, 7, 3); act[0, 2:5, 0] = 1; act[1, :, 2] = 1
    with pytest.raises(AssertionError): m.chunk_embed(feats, act)
    base = m.chunk_embed(feats); m.add_slot_input(4); assert m.config.slot_input_slots == 4
    assert torch.equal(m.chunk_embed(feats, act), base)                                                       # 0 초기화 → 붙인 직후 그대로
    with torch.no_grad(): m.adapter_slot.weight.copy_(torch.randn_like(m.adapter_slot.weight))
    got = m.chunk_embed(feats, act); W = m.adapter_slot.weight                                                 # (D, 4)
    want = base + torch.einsum("bks,ds->bkd", torch.nn.functional.pad(act, (0, 1)), W)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.allclose(m.chunk_embed(feats, act[:, :5]), base + torch.einsum("bks,ds->bkd", torch.nn.functional.pad(act[:, :5], (0, 1, 0, 2)), W), atol=1e-5)   # 짧으면 0 채움
    m.add_slot_input(4)
    with pytest.raises(AssertionError): m.add_slot_input(2)
    assert "adapter_slot" in m.trainable_state()


def test_dataset_act_in_and_collate(tmp_path):
    from test_dialogue_dataset import FakeTok
    from test_v2_slots import make_multi
    from vapasr.data.cst_dialogue import CstDialogueDataset, collate_cst
    q, d = make_multi(tmp_path, n=3, dur=72.0); kw = dict(delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, spk_activity=True)
    ds = CstDialogueDataset([q], FakeTok(), act_input=ActNoise(clean=1.0), **kw)
    random.seed(0); items = [ds[j] for j in range(3)]
    for it in items: assert it["act_in"].shape == (it["K"], 4) and torch.equal(it["act_in"], it["activity"])   # 깨끗하면 정답 활동 그대로
    b = collate_cst(items); Km = max(int(x["K"]) for x in items); assert b["act_in"].shape == (3, Km, 4)
    off = CstDialogueDataset([q], FakeTok(), **kw); random.seed(0); assert "act_in" not in off[0]
