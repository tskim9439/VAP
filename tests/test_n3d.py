"""E11 고정 화자 분리 전문가(2026-10-29, vapasr/hf/n3d): 모델 연결(add_n3d 설정, encode 스트림 1 교체, 학습 forward 의 슬롯 활동 입력 = 전문가 활동을 정답 자리로 맞춘 것,
encode_long 은 회의 전체 한 번), 전문가 가중치는 state_dict 밖. 실제 Nemotron 3 Diarization 시험(스트리밍 = NeMo forward, 배치 줄 = 단독)은
VAPASR_N3D_PATH(.nemo)와 NeMo 가 있을 때만. CPU."""
import importlib.util, os, pathlib
import numpy as np, pytest, torch

_HERE = pathlib.Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", _HERE / f"{name}.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _fake_probs(B, K, seed=0):
    g = torch.Generator().manual_seed(seed); return torch.rand(B, K, 8, generator=g)


def _model_with_n3d(tmp_path, probs_fn):
    ts = _load("test_slot_audio"); m, tok = ts._e10_model()
    f = tmp_path / "fake.nemo"; f.write_bytes(b"x")
    m.add_n3d(str(f), chunk=4, train_chunk=16)
    m.n3d.probs = probs_fn                                                                        # 실제 Nemotron 대신(읽지 않는다)
    return m, tok


def test_add_n3d_config_validation_and_state_dict(tmp_path):
    from vapasr.hf.n3d import DEFAULTS
    calls = []; m, _ = _model_with_n3d(tmp_path, lambda *a, **k: calls.append(1))
    assert m.config.n3d_cfg == {**DEFAULTS, "path": str((tmp_path / "fake.nemo").resolve()), "chunk": 4, "train_chunk": 16}
    assert not any("n3d" in k for k in m.state_dict()) and not any("n3d" in n for n, _ in m.named_parameters())   # 체크포인트·DDP 밖
    m.add_n3d(None, chunk=6, threshold=0.4); assert m.config.n3d_cfg["chunk"] == 6 and m.n3d.cfg["threshold"] == 0.4 and m.config.n3d_cfg["path"].endswith("fake.nemo")
    with pytest.raises(AssertionError): m.add_n3d(None, bogus=1)
    ts = _load("test_semcommit_tokens_loss")
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR
    tok = ts.FakeTok(); sp = ts.add_specials(tok)
    bare = VapAsrForStreamingASR(VapAsrConfig(adapter_d_in=16, adapter_d_out=ts.D, adapter_hidden=16, sp_ids=sp, special_tokens=list(ts.PHASE1_SPECIALS), blocked_ids=[7], act_hidden=4),
                                 thinker=ts._Thinker())
    with pytest.raises(AssertionError): bare.add_n3d(str(tmp_path / "fake.nemo"))                                   # 슬롯 활동 입력 없이는 안 된다
    c = VapAsrForStreamingASR(m.config, thinker=ts._Thinker()); assert c.n3d is not None and c.n3d.cfg == m.n3d.cfg   # config 로 다시 만들면 전문가도(읽기는 처음 쓸 때)


def test_encode_replaces_activity_stream(tmp_path):
    K = 9; P = _fake_probs(1, K + 3)
    m, _ = _model_with_n3d(tmp_path, lambda wav, wav_len, Kt, train=False: P[:, : int(Kt.max())])
    m.eval(); wav = torch.zeros(1, K * 1280); wl = torch.tensor([K * 1280]); Kt = torch.tensor([K])
    f0 = m.encode(wav, wl, Kt, n3d=False); f1 = m.encode(wav, wl, Kt)
    assert f0.shape == f1.shape == (1, 3, K, f0.shape[-1])
    assert torch.equal(f1[:, 0], f0[:, 0]) and torch.equal(f1[:, 2], f0[:, 2])                                     # ASR·화자 단서 임베딩 스트림은 그대로
    want = (P[0, :K, :4] > 0.5).float(); assert torch.equal(f1[0, 1, :, :4], want) and not f1[0, 1, :, 4:].any()     # 스트림 1 = 전문가 앞 4 채널 0/1
    one = torch.zeros(1, 1, K, 5); a = torch.ones(1, K + 2, 4); w = m.with_act_stream(one, a)
    assert w.shape == (1, 2, K, 5) and torch.equal(w[0, 1, :, :4], torch.ones(K, 4))                                 # 스트림 하나면 더한다(길이는 K 로)


def test_forward_feeds_expert_activity_aligned_to_reference(tmp_path):
    tb = _load("test_slot_blocks")
    ref = torch.zeros(2, 5, 4); ref[0, :3, 0] = 1; ref[0, 2:, 1] = 1; ref[1, 1:, 2] = 1; ref[1, :2, 3] = 1
    perm = [2, 0, 3, 1]                                                                              # 전문가 채널 = 정답 슬롯을 섞은 것(+ 채널 4–7 잡음 0)
    P = torch.zeros(2, 5, 8); P[..., :4] = ref[..., perm] * 0.9
    m, _ = _model_with_n3d(tmp_path, lambda wav, wav_len, Kt, train=False: P[:, : int(Kt.max())])
    b = tb._batch(m, (5, 3)); x = {k: b[k] for k in ("ids", "is_audio", "chunk_of", "labels", "mask")}
    seen = []; orig = m.build
    def spy(*a, **k): seen.append(k.get("act_in")); return orig(*a, **k)
    m.build = spy
    wav = torch.zeros(2, 5 * 1280); wl = torch.tensor([5 * 1280, 3 * 1280]); K = torch.tensor([5, 3]); jit = torch.full((2, 5, 4), 0.25)
    m.train(); m.act_pred_rho = 1.0
    out = m(**x, wav=wav, wav_len=wl, K=K, activity=ref, activity_mask=torch.ones(2, 5), act_in=jit, tag_chunk=b["tag_chunk"]); out.loss.backward()
    assert torch.isfinite(out.loss) and torch.equal(seen[-1], ref)                                                  # ρ 1: 전문가 활동을 정답 슬롯 자리로 맞춘 것
    m.act_pred_rho = 0.0; m(**x, wav=wav, wav_len=wl, K=K, activity=ref, activity_mask=torch.ones(2, 5), act_in=jit, tag_chunk=b["tag_chunk"])
    assert torch.equal(seen[-1], jit)                                                                                # ρ 0: 흔든 정답
    m(**x, wav=wav, wav_len=wl, K=K, tag_chunk=b["tag_chunk"])                                                       # 활동 라벨 없는 행: 전문가 활동 그대로(맞추지 않는다)
    assert torch.equal(seen[-1], (P[..., :4] > 0.5).float())


def test_encode_long_runs_expert_once_over_session(tmp_path):
    from vapasr.hf.long_session import encode_long
    calls = []; K = 40; P = _fake_probs(1, K + 4, 3)
    def probs(wav, wav_len, Kt, train=False): calls.append((tuple(wav.shape), int(wav_len[0]), int(Kt[0]))); return P[:, : int(Kt.max())]
    m, _ = _model_with_n3d(tmp_path, probs); m.eval(); wav = np.zeros(K * 1280, np.float32)
    got = encode_long(m, wav, K, core=16, ctx=8, tail=4)                                                            # 조각 셋 — 조각은 n3d=False, 전문가는 회의 전체 한 번
    assert calls == [((1, K * 1280), K * 1280, K)]
    assert torch.equal(got[1, :, :4], (P[0, :K, :4] > 0.5).float())
    from vapasr.hf.diar import AoscConfig
    with pytest.raises(AssertionError): encode_long(m, wav, K, aosc=AoscConfig())


_N3D = os.environ.get("VAPASR_N3D_PATH", "")


@pytest.mark.skipif(not (_N3D and os.path.exists(_N3D) and importlib.util.find_spec("nemo") is not None), reason="VAPASR_N3D_PATH(.nemo)·NeMo 가 있을 때만")
def test_real_n3d_streaming_matches_nemo_and_batch_rows():
    from vapasr.hf.n3d import load_n3d, set_streaming, stream_probs
    m, aux = load_n3d(_N3D); assert aux and all(k.startswith("sortformer_modules.activity_head.") for k in aux)
    g = torch.Generator().manual_seed(0); x = (torch.randn(16000 * 12, generator=g) * 0.05).float()
    t = torch.arange(len(x)) / 16000.0; x[: 16000 * 5] += 0.3 * torch.sin(2 * np.pi * 220 * t[: 16000 * 5]) * (torch.sin(2 * np.pi * 3 * t[: 16000 * 5]) > 0)
    p = stream_probs(m, x[None], torch.tensor([len(x)]), 4, 0, dtype="fp32"); set_streaming(m, 4, 0)
    with torch.inference_mode(): q = m(audio_signal=x[None], audio_signal_length=torch.tensor([len(x)]))
    assert p.shape == (1, 150, 8) and torch.allclose(p, q[:, : p.shape[1]], atol=1e-6)
    x2 = x[: 16000 * 7]; W = torch.zeros(2, len(x)); W[0] = x; W[1, : len(x2)] = x2
    pb = stream_probs(m, W, torch.tensor([len(x), len(x2)]), 4, 0, dtype="fp32"); ps = stream_probs(m, x2[None], torch.tensor([len(x2)]), 4, 0, dtype="fp32")
    assert torch.allclose(pb[0], p[0], atol=1e-5) and torch.allclose(pb[1, : ps.shape[1]], ps[0], atol=1e-5) and not pb[1, ps.shape[1]:].any()


@pytest.mark.skipif(not (_N3D and os.path.exists(_N3D) and importlib.util.find_spec("nemo") is not None), reason="VAPASR_N3D_PATH(.nemo)·NeMo 가 있을 때만")
def test_real_n3d_sdpa_matches_flex_attention():
    """use_sdpa(FlexAttention → SDPA, 키 패딩 마스크)가 NeMo 원래 주의와 같은 값 — 스트리밍 확률과 오프라인 층 출력·run_layers."""
    from vapasr.hf.n3d import load_n3d, run_layers, stream_probs
    m0, _ = load_n3d(_N3D, sdpa=False); m1, _ = load_n3d(_N3D)
    assert m1.encoder.sync_max_audio_length is False
    g = torch.Generator().manual_seed(1); x = (torch.randn(16000 * 9, generator=g) * 0.05).float()
    t = torch.arange(len(x)) / 16000.0; x[: 16000 * 4] += 0.3 * torch.sin(2 * np.pi * 180 * t[: 16000 * 4])
    for ch in (4, 16):
        p0 = stream_probs(m0, x[None], torch.tensor([len(x)]), ch, 0, dtype="fp32"); p1 = stream_probs(m1, x[None], torch.tensor([len(x)]), ch, 0, dtype="fp32")
        assert torch.allclose(p0, p1, atol=1e-4), (ch, float((p0 - p1).abs().max()))
    W = torch.zeros(2, len(x)); W[0] = x; W[1, : 16000 * 6] = x[: 16000 * 6]; wl = torch.tensor([len(x), 16000 * 6])
    with torch.inference_mode():
        f0, l0 = m0.preprocessor(input_signal=W, length=wl); f1, l1 = m1.preprocessor(input_signal=W, length=wl)
        e0, n0 = m0.frontend_encoder(processed_signal=f0, processed_signal_length=l0); e1, n1 = m1.frontend_encoder(processed_signal=f1, processed_signal_length=l1)
        assert torch.equal(n0, n1) and torch.allclose(e0[0], e1[0], atol=1e-4) and torch.allclose(e1[1, : int(n1[1])], e0[1, : int(n0[1])], atol=1e-4)
        h0, hl = m1.encoder.pre_encode(f1, l1); h0 = m1.encoder.embed_norm(h0)
        assert torch.allclose(run_layers(m1, h0, hl.long(), 0)[0], e1[0], atol=1e-4)                               # 0 층부터 = 인코더 전체
