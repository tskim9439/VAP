"""원거리 증강(vapasr.data.far_aug, 2026-10-26): 한 방에서 화자마다 다른 RIR·직접음 t=0(지연 없음)·원래 창 세기 유지·stems 없으면 mono 경로. CPU."""
import json, random

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
from vapasr.data.far_aug import FarFieldAugment, from_args  # noqa: E402
from vapasr.data.noise_aug import speech_power  # noqa: E402

SR = 16000


def _bank(tmp_path):
    rirs, rows = [], []
    for k, name in enumerate(["RIRS_NOISES/simulated_rirs/smallroom/Room001/Room001-00001.wav", "RIRS_NOISES/simulated_rirs/smallroom/Room001/Room001-00002.wav",
                              "RIRS_NOISES/simulated_rirs/largeroom/Room009/Room009-00001.wav", "RIRS_NOISES/simulated_rirs/largeroom/Room009/Room009-00002.wav",
                              "RIRS_NOISES/real_rirs_isotropic_noises/air_x.wav"]):
        h = np.zeros(800, np.float32); h[40 + 10 * k] = 1.0; h[200 + 30 * k] = 0.3 + 0.05 * k                       # 직접음 앞에 40+ 표본 무음
        rirs.append(h); rows.append(dict(source="real" if "real" in name else name.split("/")[2].replace("room", "") and f"sim-{name.split('/')[2]}", name=name, i=k, n=800))
    np.save(tmp_path / "rir.npy", np.stack(rirs)); (tmp_path / "rir.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    nz = (np.random.default_rng(0).standard_normal(3 * SR) * 0.1).astype(np.float32); sf.write(tmp_path / "n.wav", nz, SR)
    (tmp_path / "noise.jsonl").write_text(json.dumps(dict(category="noise", path=str(tmp_path / "n.wav"), n=len(nz))) + "\n")
    return str(tmp_path)


def test_rooms_and_pick(tmp_path):
    fa = FarFieldAugment(_bank(tmp_path), p=1.0, p_noise=0.0, p_real=0.0)
    assert sorted(map(sorted, fa.rooms)) == [[0, 1], [2, 3]] and fa.real == [4]
    for seed in range(20):
        ids = fa.pick_rirs(2, random.Random(seed)); assert len(set(ids)) == 2 and (set(ids) == {0, 1} or set(ids) == {2, 3})   # 한 방, 서로 다른 위치
    fr = FarFieldAugment(_bank(tmp_path), p=1.0, p_noise=0.0, p_real=1.0); assert fr.pick_rirs(1, random.Random(0)) == [4]


def test_render_no_delay_keeps_level(tmp_path):
    fa = FarFieldAugment(_bank(tmp_path), p=1.0, p_noise=0.0, p_real=0.0, gain_db=0.0)
    n = 2 * SR; st = np.zeros((3, n), np.float32); st[0, 1000] = 1.0; st[1, 9000] = 0.5                                     # 화자 C 는 창에서 말이 없다
    mono = st.sum(0) + 0.0
    y = fa.render(st, mono, random.Random(3)); assert y.shape == (n,) and y.dtype == np.float32
    assert abs(int(np.argmax(np.abs(y[:5000]))) - 1000) == 0 and abs(int(np.argmax(np.abs(y[5000:]))) + 5000 - 9000) == 0     # 직접음이 원래 자리(지연 없음)
    assert np.isclose(speech_power(y), speech_power(mono), rtol=1e-3)
    ym = fa.mono(mono, random.Random(3)); assert ym.shape == (n,) and np.isclose(speech_power(ym), speech_power(mono), rtol=1e-3)
    assert from_args(None, 0.5) is None and from_args(_bank(tmp_path), 0.0) is None


def test_noise_added_and_clipped(tmp_path):
    fa = FarFieldAugment(_bank(tmp_path), p=1.0, p_noise=1.0, snr=(0.0, 0.0), p_real=0.0)
    x = (np.random.default_rng(1).standard_normal(SR) * 0.9).astype(np.float32)
    y = fa.mono(x, random.Random(0)); assert np.abs(y).max() <= 0.99 + 1e-6 and not np.allclose(y, x)


def test_dataset_hook_uses_stems_then_mono(tmp_path):
    """CstDialogueDataset(far_aug): 화자별 신호 캐시가 있으면 창 구간의 (S, n) 을 render 에, 없으면 mono 경로. far_aug 를 끄면 난수를 더 뽑지 않는다."""
    import torch
    from test_dialogue_dataset import FakeTok
    from test_v2_slots import make_multi
    from vapasr.data.cst_dialogue import CstDialogueDataset
    from vapasr.data.dialogue_dataset import build_stems_cache, build_mono_cache
    q, d = make_multi(tmp_path, n=3, dur=72.0); mono = str(tmp_path / "mono")
    kw = dict(delays=(4,), window_s=(20.0, 20.0), hop_s=5.0, seed=1, max_speakers=4, mono_cache_dir=mono)
    calls = []

    class Fake:
        p = 1.0
        def render(self, st, wav, rng): calls.append(("render", tuple(st.shape), len(wav))); return np.zeros_like(wav)
        def mono(self, wav, rng): calls.append(("mono", len(wav))); return wav

    ds = CstDialogueDataset([q], FakeTok(), far_aug=Fake(), **kw); build_mono_cache(d, mono)
    random.seed(0); it = ds[0]; assert calls[-1][0] == "mono"                                                               # stems 캐시 전
    build_stems_cache(d, mono); ds._stems.clear(); random.seed(0); it = ds[0]
    assert calls[-1][0] == "render" and calls[-1][1] == (3, calls[-1][2]) and float(torch.as_tensor(it["wav"]).abs().sum()) == 0.0
    off = CstDialogueDataset([q], FakeTok(), **kw); random.seed(0); a = off[0]; st = random.getstate()
    random.seed(0); b = CstDialogueDataset([q], FakeTok(), far_aug=None, **kw)[0]
    assert random.getstate() == st and torch.equal(torch.as_tensor(a["wav"]), torch.as_tensor(b["wav"]))
