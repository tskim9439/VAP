"""P1 화자 조건 인코더 검증(experiments/ssa_probe.py) — 커널 훅·마스크 길이 맞추기·창 뽑기. CPU, 모델·오디오 없이."""
import math, random, sys, os
import numpy as np, torch
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", "experiments")]
import ssa_probe as P


class Layer(torch.nn.Module):
    def forward(self, x=None, att_mask=None):
        return x * 2


def test_fit_mask_pads_left_and_trims_left():
    m = torch.tensor([[1.0, 0.0, 1.0]])
    assert P.fit_mask(m, 5, 1.0).tolist() == [[1.0, 1.0, 1.0, 0.0, 1.0]]
    assert P.fit_mask(m, 2, 1.0).tolist() == [[0.0, 1.0]]


def test_kernel_hook_starts_as_identity_and_uses_masks():
    torch.manual_seed(0); k = P.build_kernels(d=8); lay = Layer(); lay.register_forward_pre_hook(k.hook, with_kwargs=True)
    x = torch.randn(2, 5, 8)
    assert torch.allclose(lay(x=x), x * 2)                                                    # 마스크 없음 → 그대로
    k.mt, k.mb = torch.ones(2, 5), torch.zeros(2, 5)
    assert torch.allclose(lay(x=x), x * 2)                                                    # 0 초기화 → 시작 = 원 모델
    torch.nn.init.normal_(k.t[-1].weight); k.eval()
    y1 = lay(x=x); k.mt = torch.zeros(2, 5); y0 = lay(x=x)
    assert not torch.allclose(y1, x * 2) and not torch.allclose(y1, y0)                     # 대상 활동이 출력을 바꾼다


class FakeTok:
    def decode(self, ids): return " ".join(f"w{i}" for i in ids)


class U:
    def __init__(self, spk, s, e, toks, flags=None): self.speaker, self.start, self.end, self.tokens, self.flags = spk, s, e, toks, flags


class Dlg:
    def __init__(self, utts, dur=40.0): self.utterances, self.duration_s, self.meta, self.conv_id = utts, dur, {}, "x:c1"


def test_window_sample_targets_one_speaker_text_and_masks(monkeypatch, tmp_path):
    d = Dlg([U("a", 1.0, 9.0, [[1, 2.0], [2, 5.0]]), U("b", 4.0, 12.0, [[3, 6.0], [4, 11.0]])])
    w = P.Windows.__new__(P.Windows); w.qtok = FakeTok(); w.win = (12.0, 12.0); w.rng = random.Random(1); w.cross_p = 0.0
    np.save(tmp_path / "m.npy", np.zeros(16000 * 40, np.float16))
    w.sets = [("s", [d], str(tmp_path), 1.0)]; w.W = [1.0]
    import vapasr.data.dialogue_dataset as DD
    monkeypatch.setattr(DD, "build_mono_cache", lambda d_, mono, mix_kw=None: str(tmp_path / "m.npy"))
    for _ in range(20):
        it = w.sample(); K = len(it["mt"])
        assert K == math.ceil(len(it["wav"]) / 16000 / P.FR) or abs(K - len(it["wav"]) / 16000 / P.FR) < 2
        assert it["mt"].sum() > 0 and it["text"] and set(it["text"].split()) <= {"w1", "w2", "w3", "w4"}
        assert not ({"w1", "w2"} & set(it["text"].split()) and {"w3", "w4"} & set(it["text"].split()))   # 한 화자 글자만


def test_bad_span_window_is_skipped():
    d = Dlg([U("a", 1.0, 9.0, [[1, 2.0]]), U("b", 4.0, 12.0, [], flags=None)])
    w = P.Windows.__new__(P.Windows)
    assert w.bad(d, 0.0, 10.0) and not w.bad(d, 13.0, 20.0)


def test_cross_interferer_adds_overlap_and_background_activity(monkeypatch, tmp_path):
    """P2 간섭원 덧섞기: cross_p 1 이면 다른 창이 더해지고(소리가 바뀐다) 그 창의 화자 활동이 배경 활동에 들어간다. 대상 글자는 그대로 한 화자."""
    d = Dlg([U("a", 0.0, 39.0, [[1, 2.0], [2, 5.0], [3, 15.0], [4, 25.0]])], dur=40.0)       # 화자 하나 — 배경 활동은 간섭원에서만 생긴다
    rng = np.random.default_rng(0); np.save(tmp_path / "a.npy", (0.1 * rng.standard_normal(16000 * 40)).astype(np.float16))
    import vapasr.data.dialogue_dataset as DD
    monkeypatch.setattr(DD, "build_mono_cache", lambda d_, mono, mix_kw=None: str(tmp_path / "a.npy"))
    w = P.Windows.__new__(P.Windows); w.qtok = FakeTok(); w.win = (10.0, 10.0); w.rng = random.Random(2); w.cross_p = 1.0; w.sir = (0.0, 0.0)
    w.sets = [("s", [d], str(tmp_path), 1.0)]; w.W = [1.0]
    it = w.sample()
    assert set(it["text"].split()) <= {"w1", "w2", "w3", "w4"} and it["mt"].sum() > 0
    assert it["mb"].sum() > 0                                                                     # 간섭원 활동
    w.cross_p = 0.0; w.rng = random.Random(2); it0 = w.sample()
    assert it0["mb"].sum() == 0 and not np.allclose(it0["wav"][: len(it["wav"])], it["wav"][: len(it0["wav"])])
