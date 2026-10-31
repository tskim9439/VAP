"""experiments/layer_speaker_probe.py 의 구간 고르기·EER·등록 맞히기·report(모델 없이 npz 만)."""
import json, os

import numpy as np

from experiments.layer_speaker_probe import NW, PIECE, TRIM, choose, eer, pieces, report, score, wccn


def test_pieces_trim_cut_and_window_boundary():
    K = 1200; act = np.zeros((K, 2), np.int8); mk = np.ones(K, bool)
    act[10:110, 0] = 1                                                                           # 100 프레임 단독 → 깎아 96 → 30·30·30(6 프레임 남음 버림)
    act[480:560, 1] = 1                                                                          # N3D 창 경계(500)를 넘는 단독 구간 → 경계에서 끊긴다
    act[700:760, 0] = 1; act[740:760, 1] = 1                                                     # 겹침 프레임은 단독이 아니다
    mk[1000:1100] = False; act[990:1110, 1] = 1                                                  # 셈 마스크 구간은 뺀다
    ps = pieces(act, mk)
    assert ps[:3] == [(0, 10 + TRIM, 10 + TRIM + PIECE), (0, 42, 72), (0, 72, 102)]
    assert all(e - s <= PIECE and s // NW == (e - 1) // NW for _, s, e in ps)
    assert (1, 482, 500) not in ps and (1, 500, 530) in ps                                       # 경계 앞 18 프레임은 최소(20) 미만이라 버림
    assert all(not (s < 760 and e > 740) for _, s, e in ps)                                     # 겹침 프레임을 담지 않는다
    assert all(not (s < 1100 and e > 1000) for _, s, e in ps)


def test_choose_enroll_first_then_spread():
    segs = [(0, 100 * j, 100 * j + 30) for j in range(20)] + [(1, 100 * j + 50, 100 * j + 80) for j in range(3)]
    sel = choose(segs, 2, per_spk=5)
    a = [x for x in sel if x[0] == 0]
    assert [x[1] for x in a if x[3] == 1] == [0, 100, 200] and len(a) == 8                      # 앞 3 개 등록 + 고르게 5 개
    assert [x[1] for x in a if x[3] == 0] == [300, 700, 1100, 1500, 1900]
    assert not [x for x in sel if x[0] == 1]                                                     # 등록 + 1 개가 안 되는 화자는 뺀다
    assert [x[1] for x in sel] == sorted(x[1] for x in sel)


def test_eer_and_score_separable_vs_random():
    assert eer(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 0.0
    rng = np.random.RandomState(0); sc = rng.rand(20000); lab = rng.rand(20000) < 0.3
    assert 45 < eer(sc, lab) < 55
    rows = []
    for _ in range(3):                                                                            # 화자 4 명, 화자마다 뚜렷한 방향 + 작은 잡음
        spk = np.repeat(np.arange(4), 10); st = np.tile(np.array([0, 300, 600, 1500, 3000, 6000, 9000, 12000, 15000, 18000]), 4) + spk * 40; en = st + 30
        er = np.tile((np.arange(10) < 3).astype(int), 4); cen = rng.randn(4, 64) * 5
        X = cen[spk] + rng.randn(40, 64); rows.append((X - X.mean(0), spk, st, en, er))
    e, idacc, n = score(rows)
    assert all(x is not None and x < 5 for x in e) and all(x == 100.0 for x in idacc if x is not None)
    W = wccn([(X, spk) for X, spk, *_ in rows], 0.05); assert W.shape == (64, 64) and np.allclose(W, W.T)
    e2, i2, _ = score(rows, W); assert all(x < 5 for x in e2)


def test_report_reads_npz_only(tmp_path):
    rng = np.random.RandomState(1); d = tmp_path / "emb"; d.mkdir()
    for sub, mid in (("dev", "M1"), ("eval", "M2")):
        spk = np.repeat(np.arange(3), 8); st = np.tile(np.arange(8) * 4000, 3) + spk * 40; er = np.tile((np.arange(8) < 3).astype(int), 3)
        cen = rng.randn(3, 16) * 4
        np.savez(d / f"ami-sdm__{mid}.npz", spk=spk, start=st, end=st + 30, enroll=er, speakers=np.array(["a", "b", "c"]), sub=np.array(sub), K=np.array(40000),
                 emb_ours=(cen[spk][:, None] + rng.randn(24, 2, 16)).astype(np.float16), emb_e7a=(cen[spk][:, None] + rng.randn(24, 1, 16)).astype(np.float16))
    class A: out = str(tmp_path); shrink = 0.05
    report(A)
    r = json.load(open(os.path.join(tmp_path, "probe.json")))["res"]["ami-sdm"]
    assert [x["layer"] for x in r["ours"]] == [-1, 0] and r["e7a"][0]["layer"] == "cache"
    assert r["ours"][0]["id_wccn"][2] == 100.0 and r["ours"][0]["tests"][2] > 0
