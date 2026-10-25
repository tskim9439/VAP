"""화자 분리 가지(vapasr.hf.diar, E1 2026-10-25): 모양, 도착 순서 BCE·PIT 손실, 프레임 DER 셈, 탭 지정. CPU."""
import itertools

import pytest
import torch

from vapasr.hf.diar import DiarBranch, der_counts, sort_pit_loss, tap_layers


def test_branch_shapes_and_padding():
    torch.manual_seed(0); m = DiarBranch(3, d_in=32, d=32, layers=2, heads=4, ffn=64, n_spk=4).eval()
    f = torch.randn(2, 3, 50, 32); pad = torch.zeros(2, 50, dtype=torch.bool); pad[1, 30:] = True
    y = m(f, pad); assert y.shape == (2, 50, 4)
    f2 = f.clone(); f2[1, :, 30:] = 99.0                                                       # 패딩 프레임 값은 안쪽 출력에 영향이 없다(어텐션 제외, conv 는 0 으로 채운 뒤)
    assert torch.allclose(m(f2, pad)[1, :20], y[1, :20], atol=1e-5)


def test_sort_and_pit_loss():
    t = torch.zeros(1, 40, 4); t[0, :10, 0] = 1; t[0, 15:30, 1] = 1; t[0, 25:35, 2] = 1; mask = torch.ones(1, 40, dtype=torch.bool)
    good = (t * 2 - 1) * 20.0
    loss, s, p = sort_pit_loss(good, t, mask); assert s < 1e-6 and p < 1e-6 and loss < 1e-6
    sw = good[..., [1, 0, 2, 3]]                                                                # 슬롯을 바꿔 낸 예측: 순서 손실은 크고 PIT 는 0
    loss, s, p = sort_pit_loss(sw, t, mask); assert s > 1.0 and p < 1e-6 and abs(loss - 0.5 * s) < 1e-5
    m2 = mask.clone(); m2[0, :20] = False                                                       # 마스크 프레임은 손실에서 빠진다
    bad = good.clone(); bad[0, :20] = -good[0, :20]; assert sort_pit_loss(bad, t, m2)[1] < 1e-6


def test_der_counts():
    r = torch.tensor([[1, 0], [1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
    assert der_counts(r, r) == dict(ref=5.0, miss=0.0, fa=0.0, conf_arrival=0.0, conf_best=0.0)
    d = der_counts(r[:, [1, 0]], r); assert d["conf_arrival"] == 3.0 and d["conf_best"] == 0.0 and d["miss"] == d["fa"] == 0.0
    p = r.clone(); p[3, 1] = 0; p[4, 0] = 1                                                    # 겹침 하나 놓침, 무음에 헛 활동 하나
    d = der_counts(p, r); assert d["miss"] == 1.0 and d["fa"] == 1.0 and d["conf_best"] == 0.0
    d = der_counts(p, r, torch.tensor([1, 1, 1, 0, 0])); assert d["ref"] == 3.0 and d["miss"] == d["fa"] == 0.0


def test_tap_layers():
    assert tap_layers("odd", 24) == tuple(range(1, 23, 2)) and tap_layers("all", 4) == (0, 1, 2) and tap_layers("2,5", 24) == (2, 5)
    with pytest.raises(AssertionError): tap_layers("30", 24)


def test_eval_sweep_matches_der_counts(tmp_path):
    """diar_eval sweep(2026-10-25): --save-probs 로 남긴 창 확률에서 임계별로 다시 센 DER 이 der_counts 를 창마다 합친 값과 같다."""
    import argparse, importlib.util, json, pathlib
    import numpy as np
    spec = importlib.util.spec_from_file_location("_de", pathlib.Path(__file__).parents[1] / "experiments" / "diar_eval.py"); de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)
    (tmp_path / "probs").mkdir(); g = torch.Generator().manual_seed(0); want = {}
    with open(tmp_path / "windows-rank0.jsonl", "w") as fo:
        for j in range(5):
            K = 30 + j; ref = (torch.rand(K, 4, generator=g) > 0.6).float(); prob = torch.rand(K, 4, generator=g); m = torch.rand(K, generator=g) > 0.1
            wid = f"ami-sdm:M{j}@{10.0 * j}"; sub = "dev" if j < 3 else "eval"
            fo.write(json.dumps(dict(id=wid, corpus="ami-sdm", sub=sub)) + "\n")
            np.savez_compressed(tmp_path / "probs" / f"{wid}.npz", prob=prob.numpy().astype(np.float16), ref=ref.numpy().astype(np.uint8), mask=m.numpy())
            p16 = torch.from_numpy(prob.numpy().astype(np.float16).astype(np.float32))
            for t in (0.3, 0.5):
                c = want.setdefault((sub, t), {}); [c.__setitem__(k, c.get(k, 0.0) + v) for k, v in der_counts((p16 > t).float(), ref, m).items()]
    de.sweep(argparse.Namespace(out=str(tmp_path), thresholds="0.3,0.5")); res = json.load(open(tmp_path / "sweep.json"))
    for (sub, t), c in want.items():
        got = res[f"ami-sdm/{sub}"][f"{t:g}"]
        assert got["miss"] == round(100 * c["miss"] / c["ref"], 2) and got["conf_best"] == round(100 * c["conf_best"] / c["ref"], 2)
