"""E2 대상 화자 모드(2026-10-25): CstDialogueDataset(target_spk) — 한 슬롯 시퀀스·대상 화자 말만·target_act, collate_cst 패딩, 모델 adapter_inject(0 초기화)와
encode(act) 의 인코더 inject 전달. 인코더 훅 자체(NemotronOnline pre-encode)는 mxc 스모크에서 잰다. CPU."""
import random, types

import pytest
import torch

sf = pytest.importorskip("soundfile")
from test_dialogue_dataset import FakeTok  # noqa: E402
from test_v2_slots import make_multi  # noqa: E402
from vapasr.data.cst_dialogue import CstDialogueDataset, SLOTS, collate_cst, window_speakers  # noqa: E402


def test_target_sequence_one_slot_and_only_target(tmp_path):
    q, d = make_multi(tmp_path, n=4, dur=72.0)
    ds = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, declare=4)
    base = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, declare=4)
    i = 0; cid, t0, L = ds.items[i]; spk = window_speakers(ds.eps[cid], t0, L); assert len(spk) == 4
    for t in spk:
        s = ds.cst_sequence(i, 4, target=t)
        assert s["speakers"] == [t] and s["langs"] == ["English"] and ds.problems_of(s) == []
        assert s["ids"][: len(ds.cst_prefix(["English"], 4))] == ds.cst_prefix(["English"], 4)                 # 한 슬롯 선언
        spk_ids = {ds.tok.convert_tokens_to_ids(x) for x in SLOTS[1:]}; assert not spk_ids & set(s["ids"])           # B·C·D 태그 없음
        want = [x[0] for x in s["streams"][0]]; got = [base.cst_sequence(i, 4)["streams"][spk.index(t)]]       # 그 화자 흐름 = 전체 시퀀스의 그 슬롯 흐름
        assert want == [x[0] for x in got[0]]
    assert base.cst_sequence(i, 4)["ids"] == CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4,
                                                               declare=4).cst_sequence(i, 4)["ids"]            # 끈 경로는 그대로


def test_getitem_target_act_and_collate(tmp_path):
    q, d = make_multi(tmp_path, n=3, dur=72.0)
    ds = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, target_spk=True)
    random.seed(0); items = [ds[j] for j in range(3)]
    for it in items:
        ta = it["target_act"]; assert ta.shape == (it["K"],) and set(ta.unique().tolist()) <= {0.0, 1.0} and 0 < ta.sum() < it["K"] and it["n_spk"] == 1
    b = collate_cst(items); Km = max(int(x["K"]) for x in items)
    assert b["target_act"].shape == (3, Km) and all(torch.equal(b["target_act"][j, : items[j]["K"]], items[j]["target_act"]) for j in range(3))
    off = CstDialogueDataset([q], FakeTok(), delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4)
    random.seed(0); assert "target_act" not in off[0] and "target_act" not in collate_cst([off[0]])


def _tiny():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("_tl", pathlib.Path(__file__).parent / "test_semcommit_tokens_loss.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.tiny_model()


def test_model_inject_zero_init_and_encode_passes_act():
    m, _ = _tiny(); assert m.adapter_inject is None
    m.add_spk_inject(16); assert m.config.spk_inject_hidden == 16 and torch.equal(m.adapter_inject[-1].weight, torch.zeros_like(m.adapter_inject[-1].weight))
    x = torch.randn(2, 5, m.config.adapter_d_in); assert torch.equal(m.adapter_inject(x), torch.zeros_like(x))                # 붙인 직후 주입 = 0
    m.add_spk_inject(16)
    with pytest.raises(AssertionError): m.add_spk_inject(32)
    seen = {}
    class Enc(torch.nn.Module):
        def forward(self, wav, wav_len, K, taps=(), inject=None):
            seen["inject"] = inject; return torch.zeros(wav.shape[0], int(K.max()), m.config.adapter_d_in)
    m.encoder = Enc(); act = torch.ones(1, 7)
    y = m.encode(torch.zeros(1, 100), torch.tensor([100]), torch.tensor([7]), act=act); assert y.shape == (1, 1, 7, m.config.adapter_d_in)
    assert seen["inject"][0] is m.adapter_inject and seen["inject"][1] is act
    m.encode(torch.zeros(1, 100), torch.tensor([100]), torch.tensor([7])); assert seen["inject"] is None
    m2, _ = _tiny(); m2.encoder = Enc()
    with pytest.raises(AssertionError): m2.encode(torch.zeros(1, 100), torch.tensor([100]), torch.tensor([7]), act=act)
