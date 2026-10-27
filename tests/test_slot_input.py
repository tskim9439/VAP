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


def test_e6_diar_head_loss_mixing_and_encode_stream():
    """E6(2026-10-26): 인코더 탭 위 인과 헤드 — forward 에 Sort·PIT 손실, 표본마다 ρ 로 헤드 예측/흔든 정답을 슬롯 활동 입력으로, encode 는 예측 활동 스트림을 붙이고
    chunk_embed 가 그것을 쓴다(디코드 경로가 그대로 예측을 쓴다)."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("_tl", pathlib.Path(__file__).parent / "test_semcommit_tokens_loss.py"); tl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tl)
    m, _ = tl.tiny_model(); x, _ = tl._batch(m.config.sp_ids); x["labels"] = x["labels"].clone(); x["labels"][0, 4] = -100; x["labels"][0, 8] = -100
    m.add_slot_input(4)
    cfg = dict(n_streams=2, d_in=4, d=16, layers=1, heads=2, ffn=16, n_spk=4, dropout=0.0, lookahead=0)   # d 는 위치 conv 그룹(16)의 배수
    m.add_diar_head(cfg, [0], weight=0.5, threshold=0.5, ramp=(0.0, 1.0)); assert m.config.diar_head_cfg == cfg and m.config.diar_taps == [0]

    class Enc(torch.nn.Module):
        def forward(self, wav, wav_len, K, taps=(), inject=None):
            g = torch.Generator().manual_seed(int(wav.shape[1])); y = torch.randn(wav.shape[0], 1 + len(taps), int(K.max()), 4, generator=g)
            return y if taps else y[:, 0]
    m.encoder = Enc(); del x["feats"]
    x.update(wav=torch.zeros(2, 480), wav_len=torch.tensor([320, 480]), K=torch.tensor([2, 3]))
    act = torch.zeros(2, 3, 4); act[0, :2, 0] = 1; act[1, 1:, 1] = 1; am = torch.tensor([[1., 1., 0.], [1., 1., 1.]]); ai = torch.ones(2, 3, 4)
    seen = []; orig = m.chunk_embed
    def spy(feats, a=None): seen.append(None if a is None else a.clone()); return orig(feats, a)
    m.chunk_embed = spy; m.train()
    out = m(**x, activity=act, activity_mask=am, act_in=ai); assert out.loss_diar is not None and 0.0 <= out.diar_acc <= 1.0 and torch.isfinite(out.loss)
    m.act_pred_rho = 0.0; m(**x, activity=act, activity_mask=am, act_in=ai); assert torch.equal(seen[-1], ai)                     # ρ 0 → 흔든 정답
    m.act_pred_rho = 1.0; m(**x, activity=act, activity_mask=am, act_in=ai)
    f, dlog = m.encode_diar(x["wav"], x["wav_len"], x["K"]); assert torch.equal(seen[-1], m.predict_act(dlog))                   # ρ 1 → 헤드 예측
    m.chunk_embed = orig; m.eval()
    feats = m.encode(x["wav"], x["wav_len"], x["K"]); assert feats.shape == (2, 2, 3, 4)
    assert torch.allclose(m.chunk_embed(feats), m.chunk_embed(feats[:, :1], m.predict_act(dlog)), atol=1e-6)                      # 예측 활동 스트림 = 명시 입력
    assert "diar_head" in m.trainable_state()


def test_slot_input_ties_into_loss_without_activity():
    """E5 학습(2026-10-27): 활동 입력이 없는 행(replay)에서도 adapter_slot 이 손실 그래프에 0 가중으로 묶인다 — DDP(find_unused_parameters=False)가
    쓰이지 않은 파라미터를 기다리다 멈추지 않게(spk_head·diar_head 와 같은 규칙). 값은 그대로다."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("_tl", pathlib.Path(__file__).parent / "test_semcommit_tokens_loss.py"); tl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tl)
    m, _ = tl.tiny_model(); x, _ = tl._batch(m.config.sp_ids); x["labels"] = x["labels"].clone(); x["labels"][0, 4] = -100; x["labels"][0, 8] = -100   # SEM 타깃 빼기(E6 시험과 같다)
    m.train(); base = m(**x).loss.detach()
    m.add_slot_input(4); m.train(); out = m(**x); out.loss.backward()
    assert torch.allclose(out.loss.detach(), base) and m.adapter_slot.weight.grad is not None and not m.adapter_slot.weight.grad.any()
    m.eval(); m.adapter_slot.weight.grad = None; m(**x).loss                                                    # 평가에서는 묶지 않는다
    assert m.adapter_slot.weight.grad is None


def test_e7_encode_long_aosc_matches_encode_before_cache_update():
    """E7(2026-10-27): encode_long(aosc=…) — 조각마다 인코더 탭에서 가지 임베딩을 이어 회의 전체를 AOSC 스트리밍. 캐시를 갱신하기 전(FIFO 안)이면
    E6 의 model.encode(헤드 예측 활동 스트림)와 같은 값이다. segment_plan 은 프레임을 빠짐없이 한 번씩 덮는다."""
    import importlib.util, pathlib
    import numpy as np
    from vapasr.hf.diar import AoscConfig
    from vapasr.hf.long_session import encode_long, segment_plan
    spec = importlib.util.spec_from_file_location("_tl", pathlib.Path(__file__).parent / "test_semcommit_tokens_loss.py"); tl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tl)
    m, _ = tl.tiny_model(); m.add_slot_input(4)
    m.add_diar_head(dict(n_streams=2, d_in=4, d=16, layers=1, heads=2, ffn=16, n_spk=4, dropout=0.0, lookahead=0), [0], weight=0.5, threshold=0.5, ramp=(1.0, 1.0))

    class Enc(torch.nn.Module):
        def forward(self, wav, wav_len, K, taps=(), inject=None):
            g = torch.Generator().manual_seed(int(wav.shape[1])); y = torch.randn(wav.shape[0], 1 + len(taps), int(K.max()), 4, generator=g)
            return y if taps else y[:, 0]
    m.encoder = Enc(); m.eval(); K = 40; wav = np.zeros(K * 1280, np.float32)
    ref = m.encode(torch.from_numpy(wav)[None], torch.tensor([len(wav)]), torch.tensor([K]))[0]
    got = encode_long(m, wav, K, aosc=AoscConfig(fifo=64, period=32, chunk=6))
    assert got.shape == ref.shape == (2, K, 4) and torch.allclose(got, ref, atol=1e-5)
    long = encode_long(m, np.zeros(300 * 1280, np.float32), 300, aosc=AoscConfig(cache=20, fifo=30, period=10, chunk=7))
    assert long.shape == (2, 300, 4) and set(long[1, :, :4].unique().tolist()) <= {0.0, 1.0} and not long[1, :, 4:].any()
    ch = int(getattr(m.config, "encoder_right_context", 0) or 0) + 1; Kb = 10000
    plan = segment_plan(np.zeros(Kb * 1280, np.float32), Kb, ch)
    cov = np.zeros(Kb, int)
    for seg, Ks, a0, c0, n in plan: cov[c0: c0 + n] += 1; assert a0 <= c0 and c0 + n <= a0 + Ks and len(seg) == (Kb - a0) * 1280 or len(seg) == Ks * 1280
    assert (cov == 1).all()


def test_e7_global_slots_dataset_training_only(tmp_path):
    """E7(2026-10-27): global_slots — 학습 rng 에서 그 확률로 대화 전체 도착 순서 슬롯(태그·활동·슬롯 입력이 모두 따른다). 평가(rng 없음)와 0 이면 창 순서·난수 그대로."""
    import random
    from test_dialogue_dataset import FakeTok
    from test_v2_slots import make_multi
    from vapasr.data.cst_dialogue import CstDialogueDataset
    q, d = make_multi(tmp_path); kw = dict(delays=(4,), windows=[("m1", 3.0, 10.0)], max_speakers=4, spk_activity=True)
    w = CstDialogueDataset([q], FakeTok(), **kw); g = CstDialogueDataset([q], FakeTok(), global_slots=1.0, act_input=ActNoise(clean=1.0), **kw)
    assert w.slots(0, random.Random(0))[0] == ["p1", "p2", "p3", "p0"] == g.slots(0)[0]                 # 창 순서(학습 끔·평가)
    assert g.slots(0, random.Random(0))[0] == ["p0", "p1", "p2", "p3"]
    s = g.cst_sequence(0, 4, rng=random.Random(0)); assert s["speakers"] == ["p0", "p1", "p2", "p3"] and g.problems_of(s) == []
    random.seed(0); it = g[0]; a = it["activity"]
    assert it["act_in"].shape == a.shape and torch.equal(it["act_in"], a) and a[:20, 1].all() and not a[:20, 0].any() and a[-5:, 0].all()   # p1 = 슬롯 1(3–5 s), p0 = 슬롯 0(12–13 s)
    z = CstDialogueDataset([q], FakeTok(), global_slots=0.0, **kw); r1, r2 = random.Random(5), random.Random(5)
    assert z.slots(0, r1) == w.slots(0, r2) and r1.random() == r2.random()                            # 0 이면 난수를 더 뽑지 않는다
    with pytest.raises(ValueError): CstDialogueDataset([q], FakeTok(), global_slots=1.5, **kw)


def test_e7_arrival_sort_and_align_slots():
    from vapasr.hf.diar import align_slots, arrival_sort
    t = torch.zeros(1, 10, 4); t[0, 5:, 0] = 1; t[0, 1:4, 2] = 1; t[0, 8:, 3] = 1                         # 창 안 첫 활동: 2(1) → 0(5) → 3(8), 1 은 없음
    srt = arrival_sort(t); assert torch.equal(srt[0, :, 0], t[0, :, 2]) and torch.equal(srt[0, :, 1], t[0, :, 0]) and torch.equal(srt[0, :, 2], t[0, :, 3]) and not srt[0, :, 3].any()
    m = torch.ones(1, 10, dtype=torch.bool); m[0, :4] = False; s2 = arrival_sort(t, m); assert torch.equal(s2[0, :, 0], t[0, :, 0])   # 마스크 밖 활동은 순서에 안 센다
    p = t[..., [3, 0, 1, 2]]                                                                            # 예측 열이 섞인 것 → 정답 자리로
    assert torch.equal(align_slots(p, t), t) and torch.equal(align_slots(t, t), t)


def test_e7_diar_align_forward():
    """diar_align: 헤드 손실은 창 도착 순서로 정렬한 정답, 슬롯 활동 입력(ρ 1)은 정답 자리로 맞춘 헤드 예측."""
    import importlib.util, pathlib
    from vapasr.hf.diar import align_slots, arrival_sort, sort_pit_loss
    spec = importlib.util.spec_from_file_location("_tl", pathlib.Path(__file__).parent / "test_semcommit_tokens_loss.py"); tl = importlib.util.module_from_spec(spec); spec.loader.exec_module(tl)
    m, _ = tl.tiny_model(); x, _ = tl._batch(m.config.sp_ids); x["labels"] = x["labels"].clone(); x["labels"][0, 4] = -100; x["labels"][0, 8] = -100
    m.add_slot_input(4); m.add_diar_head(dict(n_streams=2, d_in=4, d=16, layers=1, heads=2, ffn=16, n_spk=4, dropout=0.0, lookahead=0), [0], weight=0.5, threshold=0.5, ramp=(1.0, 1.0))
    m.config.diar_align = True

    class Enc(torch.nn.Module):
        def forward(self, wav, wav_len, K, taps=(), inject=None):
            g = torch.Generator().manual_seed(int(wav.shape[1])); y = torch.randn(wav.shape[0], 1 + len(taps), int(K.max()), 4, generator=g)
            return y if taps else y[:, 0]
    m.encoder = Enc(); del x["feats"]; x.update(wav=torch.zeros(2, 480), wav_len=torch.tensor([320, 480]), K=torch.tensor([2, 3]))
    act = torch.zeros(2, 3, 4); act[0, 1:, 1] = 1; act[0, :2, 3] = 1; act[1, :, 2] = 1; am = torch.tensor([[1., 1., 0.], [1., 1., 1.]]); ai = torch.zeros(2, 3, 4)
    seen = []; orig = m.chunk_embed
    def spy(feats, a=None): seen.append(None if a is None else a.clone()); return orig(feats, a)
    m.chunk_embed = spy; m.train(); m.act_pred_rho = 1.0
    out = m(**x, activity=act, activity_mask=am, act_in=ai)
    f, dlog = m.encode_diar(x["wav"], x["wav_len"], x["K"]); pred = m.predict_act(dlog); mk = am.bool() & (torch.arange(3)[None] < x["K"][:, None])
    want, _, _ = sort_pit_loss(dlog.float(), arrival_sort(act, mk), mk, float(m.config.diar_alpha))
    assert abs(float(out.loss_diar) - float(want)) < 1e-6 and torch.equal(seen[-1], align_slots(pred, act))
