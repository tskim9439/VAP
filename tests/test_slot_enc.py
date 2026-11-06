"""E15b 슬롯 조건 인코더(2026-11-06, vapasr/hf/slot_enc): CPU.
- slot_branch: NeMo 처럼 층을 키워드(x=, att_mask=, pos_emb=, pad_mask=)로 부르는 가짜 Conformer 에서, 0 초기화면 모든 벌 = 원래 출력, 아니면 벌마다 따로 돌린 것과 같다.
- NemotronOnline.forward(slots=): 0 번 스트림 = slots 없는 출력, 동결 인코더여도 SlotInject 까지 grad.
- 모델: add_slot_enc 0 초기화(슬롯 토큰 0 — E13 과 같은 입력), encode 스트림 꼴, build·tag_embeds 슬롯 토큰, config 로 다시 세운 모델 = 같은 가중, 학습 forward 의 활동 순서·DDP 묶음,
  추측 prefill 디코드 = 한 토큰씩(슬롯 토큰 포함), encode_long 조각 = 통째."""
import importlib.util, math, pathlib
import numpy as np, pytest, torch
import torch.nn as nn

_HERE = pathlib.Path(__file__).parent
DIN, HOP = 16, 1280


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}_se", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _rand_(m, seed=0, scale=0.2):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters(): p.copy_(torch.randn(p.shape, generator=g) * scale)
    return m


class FakeLayer(nn.Module):
    """국소 어텐션(att_mask) + 합성곱(pad_mask) — NeMo ConformerLayer 처럼 키워드로 불린다."""
    def __init__(self, d):
        super().__init__(); self.lin = nn.Linear(d, d); self.conv = nn.Conv1d(d, d, 3, padding=1)
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        q = x + pos_emb[:, : x.shape[1]]
        sc = (q @ x.transpose(1, 2)) / math.sqrt(x.shape[-1])
        h = x + self.lin(sc.masked_fill(att_mask, -1e9).softmax(-1) @ x)
        return h + 0.5 * self.conv(h.transpose(1, 2)).transpose(1, 2).masked_fill(pad_mask[..., None], 0.0)


class FakeNeMoEnc(nn.Module):
    """NeMo ConformerEncoder 흉내: (audio_signal (B, F, T), length) → (h (B, d, T), length). 마스크·위치 임베딩을 만들어 층을 키워드로 부른다(창 ±w)."""
    def __init__(self, f=4, d=DIN, n=4, w=2):
        super().__init__(); self.pre_encode = nn.Linear(f, d); self.layers = nn.ModuleList(FakeLayer(d) for _ in range(n)); self.w = w
    def forward(self, audio_signal, length):
        x = self.pre_encode(audio_signal.transpose(1, 2)); B, T, d = x.shape; t = torch.arange(T)
        pad = t[None] >= length[:, None]; att = pad[:, None, :] | pad[:, :, None] | ((t[:, None] - t[None]).abs() > self.w)[None]
        pos = 0.1 * torch.cos(torch.arange(d, dtype=torch.float32))[None, None].expand(1, T, d)             # 위치와 무관(상대 위치 흉내 — 조각 = 통째가 성립)
        for l in self.layers: x = l(x=x, att_mask=att, pos_emb=pos, pad_mask=pad)
        return x.transpose(1, 2), length


class FakePre(nn.Module):
    """mel 흉내: 파형 (B, L) → 프레임(80 ms = HOP 표본)마다 [평균, 표준편차, 최소, 최대] (B, 4, T), 길이 = L // HOP."""
    def forward(self, input_signal, length):
        B, L = input_signal.shape; T = L // HOP; f = input_signal[:, : T * HOP].reshape(B, T, HOP)
        return torch.stack([f.mean(-1), f.std(-1), f.min(-1).values, f.max(-1).values], 1), length // HOP


def _online():
    from vapasr.features.online import NemotronOnline
    e = NemotronOnline.__new__(NemotronOnline); nn.Module.__init__(e)
    torch.manual_seed(11); e.pre = FakePre(); e.enc = FakeNeMoEnc(); e.spec_aug = None; e.trainable = False
    for p in e.parameters(): p.requires_grad_(False)
    return e


def test_slot_branch_zero_init_and_matches_per_copy_run():
    from vapasr.hf.slot_enc import SlotInject, slot_branch
    torch.manual_seed(0); enc = FakeNeMoEnc(); B, T, N, L = 2, 30, 4, 2
    sig = torch.randn(B, 4, T); ln = torch.tensor([30, 24]); act = (torch.rand(B, T, N) > 0.6).float()
    with torch.no_grad():
        base, _ = enc(sig, ln); inj = SlotInject(DIN, 8)
        with slot_branch(enc.layers, L, inj, act) as st: h, _ = enc(sig, ln)
        R = st["R"]; assert R == N + 1 and h.shape == (B * R, DIN, T)
        hr = h.view(B, R, DIN, T); assert all(torch.allclose(hr[:, r], base, atol=1e-6) for r in range(R))   # 0 초기화 = 원래 출력
        _rand_(inj, 1)
        with slot_branch(enc.layers, L, inj, act): h, _ = enc(sig, ln)
        hr = h.view(B, R, DIN, T); assert torch.allclose(hr[:, 0], base, atol=1e-6) and not torch.allclose(hr[:, 1], base, atol=1e-3)
        # 수동: 아래층 → 주입 → 벌마다 위층(마스크는 그 표본 것)
        x = enc.pre_encode(sig.transpose(1, 2)); t = torch.arange(T); pad = t[None] >= ln[:, None]
        att = pad[:, None, :] | pad[:, :, None] | ((t[:, None] - t[None]).abs() > enc.w)[None]; pos = 0.1 * torch.cos(torch.arange(DIN, dtype=torch.float32))[None, None].expand(1, T, DIN)
        for l in enc.layers[:L]: x = l(x=x, att_mask=att, pos_emb=pos, pad_mask=pad)
        xs = torch.cat([x[:, None], inj(x, act)], 1)
        for r in range(R):
            y = xs[:, r]
            for l in enc.layers[L:]: y = l(x=y, att_mask=att, pos_emb=pos, pad_mask=pad)
            assert torch.allclose(hr[:, r], y.transpose(1, 2), atol=1e-5), r
    assert all(not l._forward_pre_hooks for l in enc.layers)                                         # 훅을 걷었다


def test_nemotron_forward_slots_streams_and_grad_to_inject():
    from vapasr.hf.slot_enc import SlotInject
    e = _online(); B, K, N = 2, 25, 4; torch.manual_seed(1); wav = torch.randn(B, K * HOP); wl = torch.tensor([K * HOP, (K - 3) * HOP]); Kt = torch.tensor([K, K - 3])
    act = (torch.rand(B, K, N) > 0.6).float(); inj = _rand_(SlotInject(DIN, 8), 2)
    base = e(wav, wl, Kt); f = e(wav, wl, Kt, slots=(2, inj, act))
    assert f.shape == (B, N + 1, K, DIN) and torch.allclose(f[:, 0], base, atol=1e-6) and not torch.allclose(f[:, 1], base, atol=1e-3)
    assert not f[1, :, K - 3:].any()                                                                 # 목표 프레임 밖은 0(fit)
    f[:, 1:].sum().backward(); assert inj.w1.weight.grad is not None and inj.w1.weight.grad.abs().sum() > 0   # 동결 인코더여도 주입까지 grad
    assert all(p.grad is None for p in e.parameters())
    with pytest.raises(AssertionError): e(wav, wl, Kt, taps=(1,), slots=(2, inj, act))


def _e15b_model():
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR
    t = _load("test_semcommit_tokens_loss")
    torch.manual_seed(0); tok = t.FakeTok(); sp = t.add_specials(tok)
    blocked = [sp["<EMPTY_AUDIO>"], sp["<SPK_A>"], sp["<SPK_B>"], 7] + [v for k, v in sp.items() if k.startswith("<DELAY_")]
    m = VapAsrForStreamingASR(VapAsrConfig(adapter_d_in=DIN, adapter_d_out=t.D, adapter_hidden=16, sp_ids=sp, special_tokens=list(t.PHASE1_SPECIALS), blocked_ids=blocked, act_hidden=4),
                              thinker=t._Thinker())
    m.add_semantic_tokens(tok, turn=True); m.enable_slots(tok, 4)
    with pytest.raises(AssertionError): m.add_slot_enc(2, 8)                                         # 슬롯 블록·활동 입력이 먼저
    m.add_slot_input(4); m.add_slot_blocks(tok, 6); m.encoder = _online()
    with pytest.raises(AssertionError): m.add_slot_enc(4, 8)                                         # 층은 0–3
    m.add_slot_enc(2, 8); m.add_slot_enc(2, 8)                                                       # 다시 불러도 그대로
    with pytest.raises(AssertionError): m.add_slot_enc(1, 8)
    return m, tok, t


def test_add_slot_enc_zero_init_streams_and_tokens():
    m, tok, t = _e15b_model(); m.eval(); B, K, N = 2, 20, 4; torch.manual_seed(2)
    assert m.config.slot_enc_cfg == dict(layer=2, hidden=8, n=4) and "adapter_slotinj" in m.trainable_state() and "adapter_slotenc" in m.trainable_state()
    assert all(torch.equal(a, b) for (ka, a), (kb, b) in zip(m.adapter_slotenc.state_dict().items(), m.adapter.state_dict().items()) if not ka.startswith(f"net.{len(m.adapter.net) - 1}."))
    wav = torch.randn(B, K * HOP); wl = torch.tensor([K * HOP] * B); Kt = torch.tensor([K] * B); act = (torch.rand(B, K, N) > 0.6).float()
    f = m.encode(wav, wl, Kt, slot_act=act); assert f.shape == (B, 2 + N, K, DIN)
    assert torch.allclose(f[:, 0], m.encoder(wav, wl, Kt), atol=1e-6) and torch.equal(f[:, 1, :, :N], act) and not f[:, 1, :, N:].any()
    x = m.slot_enc_tokens(f); assert x.shape == (B, K, N, t.D) and not x.any()                         # 0 초기화 — 슬롯 토큰 0
    assert m.slot_audio_of(f[0], act[0]).shape == (K, N, t.D) and m.slot_enc_tokens(f[:, :2]) is None
    _rand_(m.adapter_slotinj, 3); _rand_(m.adapter_slotenc, 4)
    f2 = m.encode(wav, wl, Kt, slot_act=act); x2 = m.slot_enc_tokens(f2); assert x2.abs().sum() > 0 and torch.allclose(f2[:, 0], f[:, 0], atol=1e-6)
    a3 = act.clone(); a3[:, :, 2] = 1 - a3[:, :, 2]; x3 = m.slot_enc_tokens(m.encode(wav, wl, Kt, slot_act=a3))
    assert not torch.allclose(x3[..., 2, :], x2[..., 2, :])                                         # 활동에 반응
    X = m.slot_audio_of(f2[0], act[0]); assert torch.allclose(m.tag_embeds(3, act[0], 4, X), m.tag_embeds(3, act[0], 4) + X[3], atol=1e-6)


def test_reload_from_config_keeps_slot_enc():
    from vapasr.hf import VapAsrForStreamingASR
    m, tok, t = _e15b_model(); _rand_(m.adapter_slotinj, 5); _rand_(m.adapter_slotenc, 6)
    m2 = VapAsrForStreamingASR(m.config, thinker=t._Thinker()); m2.load_state_dict(m.state_dict(), strict=False)
    for k, v in m.state_dict().items():
        if k.startswith("adapter_slot"): assert torch.equal(v, m2.state_dict()[k]), k
    assert m2.adapter_slotinj is not None and m2.adapter_slotenc is not None and dict(m2.config.slot_enc_cfg) == dict(layer=2, hidden=8, n=4)


def test_speculative_decode_with_slot_enc_equals_token_by_token():
    from vapasr.hf.batch_decode import BlockState, block_stops
    from vapasr.hf.realtime import realtime_block_decode
    tb = _load("test_slot_blocks")
    m, tok, t = _e15b_model(); m.eval(); _rand_(m.adapter_slotenc, 3)
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight))
    K, D, N = 24, 4, 4; torch.manual_seed(7); feats = torch.randn(2 + N, K, DIN); act = (torch.rand(K, N) > 0.6).float(); feats[1] = torch.nn.functional.pad(act, (0, DIN - N))
    emb = m.get_input_embeddings().weight; prefix = [5, 6, 7]; NA = m.next_audio; words = [50, 60, 70, 151721, m.sp_ids["<EOT>"]]
    pick = lambda h: NA if (h % 10) < 6 else words[(h // 10) % len(words)]; m.config.runaway_cap = 3
    X = m.slot_audio_of(feats, act); assert X.shape == (K, N, t.D) and X.abs().sum() > 0
    lm = tb._FakeLM(emb, pick); out, forced, ticks, rounds, _, nf = realtime_block_decode(m, lm, feats, prefix, D, act=act, max_flush_rounds=4)
    lm2 = tb._FakeLM(emb, pick); lm2.run_all(emb[torch.tensor(prefix)], 0); pos = len(prefix); ce = m.chunk_embed(feats[None])[0]
    st = BlockState(K, D, m.config.runaway_cap, 4, int(6 * K) + 64, tuple(m.slot_ids(D))); stop = block_stops(m, D); x = ce[0]
    while True:
        tid = lm2.run(x[None], pos); pos += 1; kind, v = st.consume(tid, stop)
        if kind == "pad": break
        x = ce[v] if kind == "audio" else m.tag_embeds(v[0], act, D, X)[v[1]] if kind == "tag" else emb[v] if kind == "text" else emb[m.empty_audio]
    assert out == st.emitted and forced == st.forced and rounds == st.flush_rounds and len(out) > 0
    base = realtime_block_decode(m, tb._FakeLM(emb, pick), feats[:2], prefix, D, act=act, max_flush_rounds=4)
    assert base[0] != out                                                                            # 슬롯 토큰이 디코드에 들어간다(스트림이 없으면 예전 디코드)


def test_encode_long_e15b_segments_equal_whole():
    from vapasr.hf.long_session import encode_long
    m, tok, t = _e15b_model(); m.eval(); _rand_(m.adapter_slotinj, 8); m.n3d = object()            # 전문가 자리(활동은 act 로 준다)
    K, N = 70, 4; torch.manual_seed(9); wav = np.random.default_rng(0).standard_normal(K * HOP).astype(np.float32); act = (np.random.default_rng(1).random((K, N)) > 0.6).astype(np.float32)
    whole = m.encode(torch.from_numpy(wav)[None], torch.tensor([K * HOP]), torch.tensor([K]), slot_act=torch.from_numpy(act)[None])[0]
    seg = encode_long(m, wav, K, core=20, ctx=20, tail=16, act=act)                                  # 수용 영역 ±12(층 4 × (어텐션 2 + 합성곱 1)) < ctx·tail
    assert seg.shape == whole.shape == (2 + N, K, DIN) and torch.allclose(seg, whole, atol=1e-4), (seg - whole).abs().max()


def test_forward_e15b_conditions_on_activity_input_and_ties_ddp():
    """학습 forward: 전문가 자리(가짜 n3d_act)로 활동을 먼저 정하고 그 활동으로 인코딩(인코딩 횟수 1), 슬롯 토큰이 태그 자리에 들어가며, 태그 없는 행도 주입·어댑터가 그래프에 묶인다."""
    m, tok, t = _e15b_model(); m.train(); _rand_(m.adapter_slotinj, 1); B, K, N = 1, 12, 4
    calls = []
    class FakeN3D:
        def act(self, wav, wl, Kt, n, train=False): return torch.ones(wav.shape[0], int(Kt.max()), n)
    m.n3d = FakeN3D(); enc0 = m.encode
    def spy(*a, **kw): calls.append(kw.get("slot_act")); return enc0(*a, **kw)
    m.encode = spy
    ids = torch.tensor([[5, 6, 0] + list(m.slot_ids(4)) + [50, 0] + list(m.slot_ids(4))]); L_ = ids.shape[1]
    is_audio = torch.zeros(1, L_, dtype=torch.bool); is_audio[0, 2] = is_audio[0, 8] = True; chunk_of = torch.full((1, L_), -1); chunk_of[0, 2], chunk_of[0, 8] = 0, 1
    tag_chunk = torch.full((1, L_), -1); tag_chunk[0, 3:7], tag_chunk[0, 9:13] = 0, 1
    labels = torch.full((1, L_), -100); labels[0, 7] = 50
    out = m(ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, wav=torch.randn(B, K * HOP), wav_len=torch.tensor([K * HOP]), K=torch.tensor([K]), tag_chunk=tag_chunk)
    assert len(calls) == 1 and calls[0] is not None and calls[0].shape[-1] == N and bool(m._slotx_used)
    out.loss.backward(); assert m.adapter_slotenc.net[-1].weight.grad is not None
    calls.clear(); m.zero_grad()
    out = m(ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, wav=torch.randn(B, K * HOP), wav_len=torch.tensor([K * HOP]), K=torch.tensor([K]))   # 태그 활동 없는 행
    out.loss.backward(); assert m.adapter_slotinj.w1.weight.grad is not None and m.adapter_slotenc.net[0].weight.grad is not None   # DDP 묶음


def test_slot_enc_off_zero_tokens():
    """2026-11-06 진단 --slot-enc-off: 슬롯 소리 토큰이 같은 모양의 0 이 된다(켜면 adapter_slotenc 출력)."""
    from types import SimpleNamespace
    import torch
    from vapasr.hf.modeling_vapasr import Adapter, VapAsrForStreamingASR as VapasrModel
    ad = Adapter(8, 6, 16)
    me = SimpleNamespace(adapter_slotenc=ad, config=SimpleNamespace(slot_enc_cfg={"n": 4}))
    f = torch.randn(2, 6, 5, 8)                                                                   # (B, 2 + N, K, Din)
    on = VapasrModel.slot_enc_tokens(me, f)
    me.slot_enc_off = True
    off = VapasrModel.slot_enc_tokens(me, f)
    assert on.shape == off.shape == (2, 5, 4, 6) and off.dtype == torch.float32
    assert float(off.abs().max()) == 0.0 and float(on.abs().max()) > 0
