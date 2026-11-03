"""E13 게이트 진단(2026-11-03): 추론 게이트만 바꾸는 경로 — gate_act(게이트만 다른 활동)·config.slot_gate 덮기·n3d_act 확률 남기기, 진단 스크립트 도움 함수. CPU."""
import importlib.util, pathlib, sys
from types import SimpleNamespace
import numpy as np, pytest, torch
pytest.importorskip("soundfile")

_HERE = pathlib.Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


_SB = _load("_t_test_slot_blocks", _HERE / "test_slot_blocks.py")
_GD = _load("_t_gate_diag", _HERE.parent / "raw/sources/experiments/2026-11-03-gate/gate_diag.py")


def _setup(seed=0):
    m, tok, bid = _SB._tiny_blocks(); m.eval(); torch.manual_seed(seed); K, D = 30, 4
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    m.config.slot_gate = 6; m.config.runaway_cap = 3
    feats = torch.randn(1, K, 4); act = (torch.rand(K, 4) > 0.8).float(); emb = m.get_input_embeddings().weight
    words = [50, 60, 70, bid, m.sp_ids["<EOT>"]]; NA = m.next_audio
    pick = lambda h: NA if (h % 10) < 5 + seed else words[(h // 10) % len(words)]
    return m, feats, act, emb, pick, K, D


@pytest.mark.parametrize("seed", [0, 1])
def test_gate_act_same_as_tag_activity_and_all_open_equals_gate_off(seed):
    from vapasr.hf.realtime import realtime_block_decode
    m, feats, act, emb, pick, K, D = _setup(seed); run = lambda **kw: realtime_block_decode(m, _SB._FakeLM(emb, pick), feats, [5, 6, 7], D, act=act, max_flush_rounds=4, **kw)
    base = run(); same = run(gate_act=act.numpy() > 0.5); opened = run(gate_act=np.ones((K, 4), bool))
    m.config.slot_gate = 0; off = run(); m.config.slot_gate = 6
    assert same[0] == base[0] and same[1] == base[1]                                             # 게이트 활동 = 태그 활동이면 그대로
    assert opened[0] == off[0] and opened[1] == off[1] and opened[0] != base[0]                    # 늘 열린 게이트 = 게이트 끔(태그 활동은 그대로)


def test_batch_gate_acts_match_realtime():
    from vapasr.hf.realtime import realtime_block_decode
    from vapasr.hf.batch_decode import decode_blocks_batch
    m, feats, act, emb, pick, K, D = _setup(2)
    ga = (torch.rand(K, 4) > 0.5).numpy() | (act.numpy() > 0.5)                                    # 태그 활동보다 넓은 게이트
    b0 = decode_blocks_batch(m, [feats], [[5, 6, 7]], D, max_flush_rounds=4, acts=[act])[0]
    b1 = decode_blocks_batch(m, [feats], [[5, 6, 7]], D, max_flush_rounds=4, acts=[act], gate_acts=[ga])[0]
    assert b1.n_tags > b0.n_tags                                                                  # 더 넓은 게이트 = 태그가 더 들어간다
    b2 = decode_blocks_batch(m, [feats], [[5, 6, 7]], D, max_flush_rounds=4, acts=[act], gate_acts=[act.numpy() > 0.5])[0]
    assert b2.emitted == b0.emitted and b2.n_tags == b0.n_tags


def test_n3d_act_keeps_probs_with_same_activity():
    from vapasr.hf.modeling_vapasr import VapAsrForStreamingASR as VapAsr
    p = torch.rand(2, 9, 8)
    class FakeExpert:
        cfg = {"threshold": 0.5}
        def probs(self, wav, wav_len, K, train=False): return p
        def act(self, wav, wav_len, K, n, train=False): return (p[..., :n] > 0.5).float()
    m = SimpleNamespace(n3d=FakeExpert(), config=SimpleNamespace(slot_input_slots=4), training=False)
    a0 = VapAsr.n3d_act(m, None, None, None)
    m.keep_n3d_probs = True; a1 = VapAsr.n3d_act(m, None, None, None)
    assert torch.equal(a0, a1) and torch.equal(m.last_n3d_probs, p) and a1.shape == (2, 9, 4)
    g = SimpleNamespace(threshold=0.3, probs=lambda wav, wav_len, K: p)                           # GraftExpert(G3) 꼴: threshold 속성
    m2 = SimpleNamespace(n3d=g, config=SimpleNamespace(slot_input_slots=4), training=False, keep_n3d_probs=True)
    assert torch.equal(VapAsr.n3d_act(m2, None, None, None), (p[..., :4] > 0.3).float())


def test_gate_diag_helpers_match_training_gate():
    from vapasr.data.cst_dialogue import gate_rounds
    rng = np.random.default_rng(0); A = rng.random((60, 3)) < 0.08
    for W in (1, 5, 12):
        g = gate_rounds(A.astype(int), W, 60); o = _GD.windows_open(A, W)
        assert all(set(np.nonzero(o[:, s])[0]) == g[s] for s in range(3))
    u = _GD.open_utt([("B", 0, 5), ("W", 0, 7), ("E", 0, 9), ("B", 1, 3)], 12, 2)
    assert list(np.nonzero(u[:, 0])[0]) == [6, 7, 8, 9] and list(np.nonzero(u[:, 1])[0]) == list(range(4, 12))
    ma, mo = _GD.match([("Yeah", 10, 0), ("so", 20, 1), ("so", 21, 1), ("okay", 40, 0)], [[("yeah,", 12), ("so", 22)], [("so", 19)]], 3)
    assert ma == [True, True, True, False] and mo == [True, True, False, False]
