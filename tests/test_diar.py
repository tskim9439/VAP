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


def test_branch_lookahead_is_causal_and_loads_bidirectional_weights():
    """lookahead R(2026-10-26, E1c·E6): 프레임 t 의 출력은 t + R 까지만 본다. 파라미터 모양이 같아 양방향 가중치를 그대로 불러온다(위치 conv 미래 탭은 0 으로 가린다)."""
    torch.manual_seed(0); bi = DiarBranch(3, d_in=32, d=32, layers=2, heads=4, ffn=64, n_spk=4).eval()
    ca = DiarBranch(3, d_in=32, d=32, layers=2, heads=4, ffn=64, n_spk=4, lookahead=2).eval(); ca.load_state_dict(bi.state_dict())
    assert ca.cfg["lookahead"] == 2 and "lookahead" not in bi.cfg
    f = torch.randn(1, 3, 60, 32); f2 = f.clone(); f2[:, :, 33:] += 2.0 * torch.randn(1, 3, 27, 32)            # 33 프레임부터 바꾼다(상수는 LayerNorm 이 지운다)
    y, y2 = ca(f), ca(f2)
    assert torch.allclose(y[0, :31], y2[0, :31], atol=1e-5) and not torch.allclose(y[0, 31:33], y2[0, 31:33], atol=1e-3)   # t ≤ 30 은 그대로, 31·32 는 33 을 본다
    assert not torch.allclose(bi(f)[0, :31], bi(f2)[0, :31], atol=1e-3)                                     # 양방향은 미래가 바뀌면 앞도 바뀐다


def _causal(n_streams=2, d=16):
    torch.manual_seed(0); return DiarBranch(n_streams, d_in=8, d=d, layers=2, heads=4, ffn=32, n_spk=4, dropout=0.0, lookahead=0).eval()


def test_embed_head_split_is_forward():
    m = _causal(); f = torch.randn(2, 2, 30, 8); pad = torch.zeros(2, 30, dtype=torch.bool); pad[1, 20:] = True
    assert torch.equal(m(f, pad), m.head(m.embed(f), pad))


def test_aosc_select_rules():
    from vapasr.hf.diar import AoscConfig, aosc_select
    cfg = AoscConfig(cache=10, sil=2, k1=2, k2=3)
    p = torch.full((12, 4), 0.01); p[0:4, 0] = 0.95; p[4:8, 1] = 0.9; p[8:10, 0] = 0.99; p[9, 1] = 0.9      # 9 = 겹침, 10·11 = 무음
    new = torch.zeros(12, dtype=torch.bool)
    fi, si = aosc_select(p, new, cfg)
    assert len(fi) <= 10 and si.tolist() == sorted(si.tolist())                                   # 화자(도착 순서)로 묶인다
    for s_ in (0, 1):                                                                             # 화자마다 무음 칸 2 개가 그 화자 프레임 뒤에
        fs = fi[si == s_].tolist(); assert fs[-2:] == [-1, -1] and all(x >= 0 for x in fs[:-2]) and fs[:-2] == sorted(fs[:-2])
    assert set(fi[si == 0].tolist()) - {-1} <= {0, 1, 2, 3, 8, 9} and 10 not in fi.tolist() and 11 not in fi.tolist()   # 무음 프레임은 화자 칸이 아니다
    small = AoscConfig(cache=6, sil=1, k1=1, k2=1); fi2, si2 = aosc_select(p, new, small)
    assert len(fi2) == 6 and (si2 == 0).any() and (si2 == 1).any()                               # 상한 안에서도 두 화자가 남는다(상위 k 가산점)
    n2 = new.clone(); n2[4] = True; q = p.clone(); q[4:8, 1] = 0.9                                  # 같은 점수면 새 프레임이 먼저 남는다(+recency)
    fi3, si3 = aosc_select(q, n2, AoscConfig(cache=4, sil=0, k1=0, k2=0)); assert 4 in fi3.tolist() or (si3 == 1).sum() == 0
    assert aosc_select(torch.zeros(0, 4), torch.zeros(0, dtype=torch.bool), cfg)[0].numel() == 0


def test_aosc_stream_matches_full_forward_before_first_update():
    from vapasr.hf.diar import AoscConfig, aosc_stream
    m = _causal(); f = torch.randn(1, 2, 40, 8); e = m.embed(f)[0]
    full = m.head(e[None])[0]
    for ch in (1, 6, 40):                                                                          # FIFO 가 넘치기 전이면 청크 크기와 상관없이 통째 인과 forward 와 같다
        y = aosc_stream(m, e, AoscConfig(fifo=64, period=32, chunk=ch)); assert torch.allclose(y, full, atol=1e-5)
    y = aosc_stream(m, torch.randn(300, 16), AoscConfig(cache=20, fifo=30, period=10, chunk=7)); assert y.shape == (300, 4) and torch.isfinite(y).all()


def test_aosc_clip_loss_single_step_and_grad():
    import random
    from vapasr.hf.diar import AoscConfig, aosc_clip_loss
    m = _causal(); f = torch.randn(1, 2, 50, 8); t = torch.zeros(50, 4); t[5:20, 0] = 1; t[18:40, 1] = 1; t[42:48, 2] = 1; mask = torch.ones(50, dtype=torch.bool)
    e = m.embed(f)[0]
    tot, st = aosc_clip_loss(m, e, t, mask, AoscConfig(fifo=64), random.Random(0), chunk=50)          # 한 step(캐시 없음) = 통째 손실
    ref, _, _ = sort_pit_loss(m.head(e[None]).float(), t[None], mask[None])
    assert st["steps"] == 1 and abs(float(tot) / st["frames"] - float(ref)) < 1e-5
    m.train(); e = m.embed(f)[0]
    tot, st = aosc_clip_loss(m, e, t, mask, AoscConfig(cache=12, fifo=10, period=5, sil=1, k1=2, k2=3), random.Random(1), chunk=4, perm_p=1.0)
    assert st["steps"] == 13 and st["frames"] == 50 and torch.isfinite(tot); tot.backward()
    assert m.proj.weight.grad is not None and m.out.weight.grad.abs().sum() > 0


def test_forward_aosc_mode_returns_mean_loss():
    from vapasr.hf.diar import AoscConfig
    m = _causal(); f = torch.randn(2, 2, 30, 8); act = torch.zeros(2, 30, 4); act[0, 2:10, 0] = 1; act[0, 12:25, 1] = 1; act[1, 5:28, 0] = 1
    mask = torch.ones(2, 30, dtype=torch.bool); mask[1, 25:] = False
    loss, st = m(f, aosc=dict(act=act, mask=mask, K=torch.tensor([30, 25]), cfg=AoscConfig(cache=8, fifo=10, period=5, sil=1), seeds=[0, 1], chunk=6))
    assert torch.isfinite(loss) and st["frames"] == 55 and st["steps"] == 10


def test_aosc_select_min_pos_drops_overlap_frames():
    from vapasr.hf.diar import AoscConfig, aosc_select
    p = torch.full((30, 4), 0.01); p[0:25, 0] = 0.95; p[25:28, 0] = 0.9; p[25:28, 1] = 0.9; p[28:30, 1] = 0.95   # 0: 혼자 25 + 겹침 3, 1: 겹침 3 + 혼자 2
    new = torch.zeros(30, dtype=torch.bool)
    fi, si = aosc_select(p, new, AoscConfig(cache=40, sil=0, min_pos=22))
    assert set(fi[si == 0].tolist()) == set(range(25)) and set(fi[si == 1].tolist()) == set(range(25, 30))   # 혼자 프레임이 22 개 이상인 화자만 겹침을 뺀다
    fi2, si2 = aosc_select(p, new, AoscConfig(cache=40, sil=0, min_pos=0)); assert set(fi2[si2 == 0].tolist()) == set(range(28))


def test_aosc_owners_and_self_cache_loss():
    import random
    from vapasr.hf.diar import AoscConfig, _owners, aosc_clip_loss
    t = torch.zeros(10, 4); t[0:4, 2] = 1; t[4:8, 0] = 1; t[8, 1] = 1
    c_idx = torch.tensor([0, 1, -1, 4, 5, 9, -1])                                                  # 묶음: 화자 2 프레임 · 화자 0 프레임 · 정답 활동 없는 프레임(무음 칸 포함)
    assert _owners([(0, [0, 1, 2]), (1, [3, 4]), (3, [5, 6])], c_idx, t, 4) == [2, 0, -1]
    assert _owners([(0, [0, 1]), (1, [3])], torch.tensor([0, 1, 9, 2]), t, 4) == [2, -1]              # 같은 정답 화자는 한 묶음만(일대일)
    m = _causal(); m.train(); e = m.embed(torch.randn(1, 2, 60, 8))[0]
    tgt = torch.zeros(60, 4); tgt[0:25, 0] = 1; tgt[20:50, 1] = 1; tgt[52:58, 2] = 1; mask = torch.ones(60, dtype=torch.bool)
    cfg = AoscConfig(cache=12, fifo=10, period=5, sil=1, k1=2, k2=3, thr=0.0)                       # thr 0: 학습 안 된 모델 예측으로도 캐시가 찬다
    tot, st = aosc_clip_loss(m, e, tgt, mask, cfg, random.Random(0), chunk=4, self_cache=True)
    assert st["self"] == 1.0 and st["steps"] == 15 and st["frames"] == 60 and torch.isfinite(tot); tot.backward()
    assert m.out.weight.grad.abs().sum() > 0 and m.proj.weight.grad is not None


def test_forward_aosc_passes_noise_and_self_p():
    import random
    from vapasr.hf.diar import AoscConfig, aosc_clip_loss
    m = _causal(); f = torch.randn(2, 2, 30, 8); act = torch.zeros(2, 30, 4); act[0, 2:10, 0] = 1; act[0, 12:25, 1] = 1; act[1, 5:28, 0] = 1; act[1, 9:20, 1] = 1
    mask = torch.ones(2, 30, dtype=torch.bool); cfg = AoscConfig(cache=8, fifo=10, period=5, sil=1)
    kw = dict(act=act, mask=mask, K=torch.tensor([30, 30]), cfg=cfg, seeds=[0, 1], chunk=6)
    assert m(f, aosc=dict(kw, self_p=1.0))[1]["self"] == 2.0 and m(f, aosc=dict(kw, self_p=0.0))[1]["self"] == 0.0
    e = m.embed(f)
    for nz in (0.0, 2.0):                                                                          # noise 가 aosc_clip_loss 까지 간다
        ref = sum(aosc_clip_loss(m, e[j], act[j], mask[j], cfg, random.Random(j), chunk=6, noise=nz)[0] for j in range(2)) / 60
        assert torch.allclose(m(f, aosc=dict(kw, noise=nz))[0], ref)


def test_e8_shift_frames_and_unshift():
    import importlib
    from vapasr.hf.diar import shift_frames
    x = torch.arange(12.0).reshape(1, 6, 2); y = shift_frames(x, 2)
    assert torch.equal(y[0, :2], torch.zeros(2, 2)) and torch.equal(y[0, 2:], x[0, :4]) and torch.equal(shift_frames(x, 0), x)
    m = torch.ones(1, 6, dtype=torch.bool); ms = shift_frames(m, 2, False); assert ms[0].tolist() == [False, False, True, True, True, True]
    assert shift_frames(x, 9).abs().sum() == 0 and shift_frames(x, 9).shape == x.shape                     # 창보다 길게 밀면 전부 채움 값
    unshift = importlib.import_module("experiments.diar_eval").unshift
    p = torch.arange(10.0)[:, None]; r = torch.arange(10.0)[:, None] + 100; mk = torch.ones(10, dtype=torch.bool)
    p2, r2, m2 = unshift(p, r, mk, 3); assert p2[:, 0].tolist() == list(range(3, 10)) and r2[:, 0].tolist() == list(range(100, 107)) and len(m2) == 7
    assert unshift(p, r, mk, 0)[0] is p
