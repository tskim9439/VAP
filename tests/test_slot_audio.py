"""E10 슬롯 추출기(2026-10-29): SlotAudio(0 초기화·인과·조각 = 통째·활동에 반응), 모델 연결(add_slot_audio 가 헤드 embed 로 시작, encode 셋째 스트림,
build 의 태그 자리 슬롯 소리 토큰, 섞인 소리 지우기, forward gradient·DDP 묶음), 디코드(추측 prefill = 한 토큰씩, 슬롯 소리·섞인 소리 끄기 포함), encode_long AOSC 셋째 스트림. CPU."""
import importlib.util, pathlib
import numpy as np, pytest, torch

from vapasr.hf.slot_audio import SlotAudio

_HERE = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _rand_(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters(): p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return m


def test_slot_audio_zero_init_causal_and_segmented():
    torch.manual_seed(0); B, K, N = 2, 60, 4
    sa = SlotAudio(n_streams=3, d_enc=6, d_emb=5, d=8, blocks=2, kernel=4, ffn=16, d_out=7)
    enc, emb = torch.randn(B, K, 6), torch.randn(B, K, 5); act = (torch.rand(B, K, N) > 0.6).float()
    assert sa(enc, emb, act).shape == (B, K, N, 7) and not sa(enc, emb, act).any()                     # 출력 0 초기화
    _rand_(sa); y = sa(enc, emb, act)
    t = 33; e2, m2, a2 = enc.clone(), emb.clone(), act.clone(); e2[:, t:] += 5; m2[:, t:] -= 3; a2[:, t:] = 1 - a2[:, t:]
    y2 = sa(e2, m2, a2); assert torch.allclose(y[:, :t], y2[:, :t], atol=1e-5) and not torch.allclose(y[:, t:], y2[:, t:])   # 인과 — t 이후를 바꿔도 앞은 같다
    for seg in (7, 16, 25): assert torch.allclose(sa(enc, emb, act, seg=seg), y, atol=1e-4), seg                       # 조각(문맥 R = 6) = 통째
    a3 = act.clone(); a3[:, :, 1] = 0; assert not torch.allclose(sa(enc, emb, a3)[:, :, 1], y[:, :, 1])                # 활동에 반응
    feats = torch.randn(B, 3, K, 6); w = torch.softmax(sa.mix, 0)
    assert torch.allclose(sa.embed(feats), sa.proj_t(sa.ln_t((feats * w[None, :, None, None]).sum(1))), atol=1e-6)


DIN = 16                                                                                               # 인코더 특징 차원 — 화자 단서 임베딩(헤드 d 16)을 셋째 스트림에 담을 만큼


def _e10_model(mix_drop=0.5):
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR
    t = _load("test_semcommit_tokens_loss")
    torch.manual_seed(0); tok = t.FakeTok(); sp = t.add_specials(tok)
    blocked = [sp["<EMPTY_AUDIO>"], sp["<SPK_A>"], sp["<SPK_B>"], 7] + [v for k, v in sp.items() if k.startswith("<DELAY_")]
    m = VapAsrForStreamingASR(VapAsrConfig(adapter_d_in=DIN, adapter_d_out=t.D, adapter_hidden=16, sp_ids=sp, special_tokens=list(t.PHASE1_SPECIALS), blocked_ids=blocked, act_hidden=4),
                              thinker=t._Thinker())
    m.add_semantic_tokens(tok, turn=True); m.enable_slots(tok, 4); m.add_slot_input(4); m.add_slot_blocks(tok, 6)
    with pytest.raises(AssertionError): m.add_slot_audio(d=8, blocks=2, kernel=3, ffn=16)                              # 헤드 없이는 안 된다
    m.add_diar_head(dict(n_streams=2, d_in=DIN, d=16, layers=1, heads=2, ffn=16, n_spk=4, dropout=0.0, lookahead=0), [0], weight=0.5, threshold=0.5, ramp=(1.0, 1.0))
    m.add_slot_audio(d=8, blocks=2, kernel=3, ffn=16, mix_drop=mix_drop)

    class Enc(torch.nn.Module):
        def forward(self, wav, wav_len, K, taps=(), inject=None):
            g = torch.Generator().manual_seed(int(wav.shape[1])); y = torch.randn(wav.shape[0], 1 + len(taps), int(K.max()), DIN, generator=g)
            return y if taps else y[:, 0]
    m.encoder = Enc(); return m, tok


def test_add_slot_audio_starts_from_head_embed_and_encode_streams():
    m, tok = _e10_model(); sx = m.adapter_slotx
    assert m.config.slot_audio_cfg == dict(n_streams=2, d_enc=DIN, d_emb=16, d=8, blocks=2, kernel=3, ffn=16, d_out=8) and m.config.slot_mix_drop == 0.5
    assert torch.equal(sx.mix, m.diar_head.mix) and torch.equal(sx.proj_t.weight, m.diar_head.proj.weight) and not m.adapter_mixnull.any()
    feats = torch.randn(1, 2, 9, DIN); assert torch.allclose(sx.embed(feats), m.diar_head.embed(feats))                # 헤드 embed 에서 시작
    m.add_slot_audio(d=8, blocks=2, kernel=3, ffn=16)                                                                  # 같은 설정이면 그대로
    with pytest.raises(AssertionError): m.add_slot_audio(d=16, blocks=2, kernel=3, ffn=16)
    m.eval(); K = 9; wav = torch.zeros(1, K * 1280); f = m.encode(wav, torch.tensor([K * 1280]), torch.tensor([K]))
    assert f.shape == (1, 3, K, DIN)
    allf = m.encoder(wav, None, torch.tensor([K]), taps=(0,)); assert torch.allclose(f[0, 2, :, :16], sx.embed(allf)[0], atol=1e-6)        # 셋째 = 추출기 embed
    assert "adapter_slotx" in m.trainable_state() and "adapter_mixnull" in m.trainable_state()


def _block_batch(m, Ks=(5, 3)):
    tb = _load("test_slot_blocks"); return tb._batch(m, Ks)


def test_build_adds_slot_tokens_and_mix_mask():
    m, tok = _e10_model(); m.eval(); b = _block_batch(m); Km = 5; torch.manual_seed(5)
    feats = torch.randn(2, 1, Km, DIN); act = (torch.rand(2, Km, 4) > 0.5).float(); X = torch.randn(2, Km, 4, 8)
    E0 = m.build(feats, b["ids"], b["is_audio"], b["chunk_of"], act_in=act, tag_chunk=b["tag_chunk"])
    E1 = m.build(feats, b["ids"], b["is_audio"], b["chunk_of"], act_in=act, tag_chunk=b["tag_chunk"], slot_x=X)
    for r in range(2):
        for p in range(b["ids"].shape[1]):
            k = int(b["tag_chunk"][r, p])
            if k < 0: assert torch.equal(E1[r, p], E0[r, p]); continue
            s_ = m.slot_ids(4).index(int(b["ids"][r, p])); assert torch.allclose(E1[r, p], E0[r, p] + X[r, k, s_], atol=1e-6)
    with torch.no_grad(): m.adapter_mixnull.fill_(0.7)
    E2 = m.build(feats, b["ids"], b["is_audio"], b["chunk_of"], act_in=act, tag_chunk=b["tag_chunk"], mix_mask=torch.tensor([True, False]))
    ce_off = m.chunk_embed(feats[:1], act[:1], mix_mask=torch.tensor([True]))[0]; want = torch.full_like(ce_off, 0.7) + m.adapter_slot(act[0])
    assert torch.allclose(ce_off, want, atol=1e-6)                                                                   # 섞인 소리 → 빈 임베딩 + 활동 몫
    au0 = b["is_audio"][0]; assert torch.allclose(E2[0][au0], ce_off[b["chunk_of"][0][au0]], atol=1e-6) and torch.equal(E2[1], E0[1])


def test_forward_trains_extractor_and_ties():
    m, tok = _e10_model(mix_drop=1.0); b = _block_batch(m); x = {k: b[k] for k in ("ids", "is_audio", "chunk_of", "labels", "mask")}
    wav = torch.zeros(2, 5 * 1280); wl = torch.tensor([5 * 1280, 3 * 1280]); K = torch.tensor([5, 3]); act = torch.zeros(2, 5, 4); act[0, :3, 0] = 1; act[1, 1:, 1] = 1
    m.train(); m.act_pred_rho = 0.0
    out = m(**x, wav=wav, wav_len=wl, K=K, activity=act, activity_mask=torch.ones(2, 5), act_in=act, tag_chunk=b["tag_chunk"]); out.loss.backward()
    g = m.adapter_slotx.out.weight.grad; assert torch.isfinite(out.loss) and g is not None and g.abs().sum() > 0          # 0 초기화 출력층도 gradient 를 받는다
    assert m.adapter_mixnull.grad is not None and m.adapter_mixnull.grad.abs().sum() > 0                               # mix_drop 1 → 빈 임베딩이 쓰였다
    m.zero_grad(); out = m(**x, wav=wav, wav_len=wl, K=K, activity=act, activity_mask=torch.ones(2, 5), act_in=act); out.loss.backward()   # 태그 없는 행
    assert m.adapter_slotx.out.weight.grad is not None and not m.adapter_slotx.out.weight.grad.any()


@pytest.mark.parametrize("mix_off", [False, True])
def test_speculative_decode_with_slot_audio_equals_token_by_token(mix_off):
    from vapasr.hf.batch_decode import BlockState, block_stops
    from vapasr.hf.realtime import realtime_block_decode
    tb = _load("test_slot_blocks")
    m, tok = _e10_model(); m.eval(); _rand_(m.adapter_slotx, 3)
    with torch.no_grad(): m.adapter_tagact.weight.copy_(torch.randn_like(m.adapter_tagact.weight)); m.adapter_mixnull.fill_(0.3)
    K, D = 24, 4; torch.manual_seed(7); feats = torch.randn(3, K, DIN); act = (torch.rand(K, 4) > 0.6).float(); feats[1] = torch.nn.functional.pad(act, (0, DIN - 4))
    emb = m.get_input_embeddings().weight; prefix = [5, 6, 7]; NA = m.next_audio; words = [50, 60, 70, 151721, m.sp_ids["<EOT>"]]
    pick = lambda h: NA if (h % 10) < 6 else words[(h // 10) % len(words)]; m.config.runaway_cap = 3
    X = m.slot_audio_of(feats, act); assert X.shape == (K, 4, 8) and X.abs().sum() > 0
    lm = tb._FakeLM(emb, pick); out, forced, ticks, rounds, _, nf = realtime_block_decode(m, lm, feats, prefix, D, act=act, max_flush_rounds=4, mix_off=mix_off)
    lm2 = tb._FakeLM(emb, pick); lm2.run_all(emb[torch.tensor(prefix)], 0); pos = len(prefix)
    ce = m.chunk_embed(feats[None], mix_mask=torch.tensor([True]) if mix_off else None)[0]
    st = BlockState(K, D, m.config.runaway_cap, 4, int(6 * K) + 64, tuple(m.slot_ids(D))); stop = block_stops(m, D); x = ce[0]
    while True:
        tid = lm2.run(x[None], pos); pos += 1; kind, v = st.consume(tid, stop)
        if kind == "pad": break
        x = ce[v] if kind == "audio" else m.tag_embeds(v[0], act, D, X)[v[1]] if kind == "tag" else emb[v] if kind == "text" else emb[m.empty_audio]
    assert out == st.emitted and forced == st.forced and rounds == st.flush_rounds and len(out) > 0
    t0 = m.tag_embeds(2, act, D); t1 = m.tag_embeds(2, act, D, X); assert torch.allclose(t1, t0 + X[2], atol=1e-6)
    assert torch.equal(m.tag_embeds(K + 1, act, D, X), m.tag_embeds(K + 1, act, D))                                    # flush 라운드는 슬롯 소리 없음


def test_encode_long_aosc_adds_slot_embedding_stream():
    from vapasr.hf.diar import AoscConfig
    from vapasr.hf.long_session import encode_long
    m, tok = _e10_model(); m.eval(); K = 40; wav = np.zeros(K * 1280, np.float32)
    got = encode_long(m, wav, K, aosc=AoscConfig(fifo=64, period=32, chunk=6))
    ref = m.encode(torch.from_numpy(wav)[None], torch.tensor([len(wav)]), torch.tensor([K]))[0]
    assert got.shape == ref.shape == (3, K, DIN) and torch.allclose(got, ref, atol=1e-5)                                # 캐시 갱신 전이면 encode 와 같다(셋째 스트림 포함)
