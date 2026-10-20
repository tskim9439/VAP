"""V2-a 두 스트림 입력 — NemotronOnline(taps=) 층 출력 hook·자르기, add_enc_tap(0 초기화 → V1 과 같은 출력, 재호출·다른 층), chunk_embed 스트림 검사,
0 초기화에서 학습이 시작되는지, config·state_dict 왕복, adapter lr 그룹 이름. NeMo·Qwen 없이 CPU 에서 수 초."""
import contextlib, types
import pytest, torch, torch.nn as nn
from vapasr.features.online import NemotronOnline


@pytest.fixture(autouse=True)
def _cpu_autocast(monkeypatch):                                                            # CPU 전용 torch(로컬 2.0)는 autocast("cuda") 를 거부한다 — 인코더 forward 의 autocast 를 빈 문맥으로
    if not torch.cuda.is_available(): monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())

V, D, DIN, DM = 64, 8, 4, 4                                                                # 어휘·thinker hidden·인코더 출력(= adapter_d_in)


class _Pre(nn.Module):                                                                     # 오디오 4 샘플 = 특징 프레임 1 (B, 4, T)
    def forward(self, input_signal, length): B, L = input_signal.shape; return input_signal[:, : L // 4 * 4].reshape(B, -1, 4).transpose(1, 2), length // 4


class _Layer(nn.Module):
    def __init__(self): super().__init__(); self.lin = nn.Linear(DM, DM)
    def forward(self, x): return torch.tanh(self.lin(x))


class _Enc(nn.Module):                                                                     # NeMo ConformerEncoder 흉내: layers 출력 (B, T', D), 반환 (B, D, T'), 길이
    def __init__(self, n=3): super().__init__(); self.inp = nn.Linear(4, DM); self.layers = nn.ModuleList(_Layer() for _ in range(n))
    def forward(self, audio_signal, length):
        x = self.inp(audio_signal.transpose(1, 2))
        for l in self.layers: x = l(x)
        return x.transpose(1, 2), length


def fake_online(seed=0):
    torch.manual_seed(seed); e = NemotronOnline.__new__(NemotronOnline); nn.Module.__init__(e)
    e.pre, e.enc, e.spec_aug, e.trainable = _Pre(), _Enc(), None, False; return e.eval()


class _Inner(nn.Module):
    def __init__(s): super().__init__(); s.embed_tokens = nn.Embedding(V, D); s.lin = nn.Linear(D, D)
    def forward(s, inputs_embeds=None, attention_mask=None, **_): return types.SimpleNamespace(last_hidden_state=torch.tanh(s.lin(inputs_embeds)))
class _Thinker(nn.Module):
    def __init__(s): super().__init__(); s.model = _Inner(); s.lm_head = nn.Linear(D, V, bias=False)
    def get_input_embeddings(s): return s.model.embed_tokens
    def set_input_embeddings(s, v): s.model.embed_tokens = v


def tiny(**cfg_kw):
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR
    torch.manual_seed(0); cfg = VapAsrConfig(adapter_d_in=DIN, adapter_d_out=D, adapter_hidden=16, sp_ids={"<NEXT_AUDIO>": 60, "<EMPTY_AUDIO>": 61}, **cfg_kw)
    m = VapAsrForStreamingASR(cfg, thinker=_Thinker()); m.encoder = fake_online(); return m


def _wav(B=2, L=40, seed=1):
    g = torch.Generator().manual_seed(seed); return torch.randn(B, L, generator=g), torch.tensor([L, L - 12])


def test_online_taps_hook_and_trim():
    e = fake_online(); wav, wl = _wav(); K = torch.tensor([10, 9])                         # T' = 10 · 7(두 번째는 길이 7 → 9 중 7 만 채우고 0)
    out = e(wav, wl, K); both = e(wav, wl, K, taps=(0, 2))
    assert out.shape == (2, 10, DM) and both.shape == (2, 3, 10, DM) and torch.equal(both[:, 0], out)
    x = e.enc.inp(e.pre(wav, wl)[0].transpose(1, 2)); l0 = e.enc.layers[0](x)
    assert torch.allclose(both[0, 1], l0[0]) and torch.allclose(both[1, 1, :7], l0[1, :7]) and both[1, 1, 7:].abs().sum() == 0
    assert torch.allclose(both[:, 2], out)                                                  # 마지막 층(2)을 탭하면 0 번 스트림과 같다
    assert all(len(l._forward_hooks) == 0 for l in e.enc.layers)                            # hook 은 호출마다 떼어 낸다
    assert e(wav, wl, torch.tensor([12, 12]), taps=(1,)).shape == (2, 2, 12, DM)            # K 가 T' 보다 길면 0 패딩


def test_add_enc_tap_zero_init_equals_v1():
    m = tiny(); wav, wl = _wav(); K = torch.tensor([10, 7])
    f1 = m.encode(wav, wl, K); assert f1.shape == (2, 1, 10, DIN) and m.adapter_tap is None
    ce1 = m.chunk_embed(f1)
    m.add_enc_tap(1); f2 = m.encode(wav, wl, K)
    assert f2.shape == (2, 2, 10, DIN) and torch.equal(f2[:, 0], f1[:, 0]) and m.config.enc_tap_layer == 1 and m.config.enc_tap_hidden == 16
    assert torch.equal(m.chunk_embed(f2), ce1)                                              # 마지막 Linear 0 → 붙인 직후 LLM 입력이 V1 과 같다
    tap = m.adapter_tap; m.add_enc_tap(1); assert m.adapter_tap is tap                     # 같은 층 재호출(재개)은 그대로
    with pytest.raises(AssertionError, match="이미 탭 층"): m.add_enc_tap(0)
    with pytest.raises(AssertionError, match="특징 스트림"): m.chunk_embed(f1)              # 두 스트림 모델에 한 스트림 특징(캐시·live 경로)은 거부
    with pytest.raises(AssertionError, match="인코더 층"): tiny().add_enc_tap(3)
    names = [n for n, _ in m.named_parameters() if n.startswith("adapter_tap.")]
    assert names and all(n.startswith(("adapter", "spk_head")) for n in names)              # trainer.create_optimizer 의 adapter lr 그룹(이름 접두사)


def test_tap_learns_from_zero_init():
    m = tiny(); m.add_enc_tap(0); wav, wl = _wav(); K = torch.tensor([10, 7]); tgt = torch.randn(2, 10, D, generator=torch.Generator().manual_seed(3))
    opt = torch.optim.SGD(m.adapter_tap.parameters(), lr=0.5)
    def step():
        opt.zero_grad(); ((m.chunk_embed(m.encode(wav, wl, K)) - tgt) ** 2).mean().backward()
        return m.adapter_tap.net[1].weight.grad.abs().sum().item(), m.adapter_tap.net[-1].weight.grad.abs().sum().item()
    g_first, g_last = step(); assert g_last > 0 and g_first == 0                            # 0 초기화: 첫 step 은 마지막 Linear 만 움직인다
    opt.step(); g_first, g_last = step(); assert g_first > 0 and g_last > 0                 # 그다음부터 탭 가지 전체가 배운다


def test_config_and_state_roundtrip(tmp_path):
    from vapasr.hf import VapAsrConfig, VapAsrForStreamingASR
    assert VapAsrConfig.from_dict({"sp_ids": {}}).enc_tap_layer == -1                       # 탭 이전 config(V1 체크포인트)는 끔
    m = tiny(); m.add_enc_tap(2, hidden=12)
    with torch.no_grad():
        for p in m.adapter_tap.parameters(): p.add_(0.1)
    m.config.save_pretrained(tmp_path); c = VapAsrConfig.from_pretrained(tmp_path); assert (c.enc_tap_layer, c.enc_tap_hidden) == (2, 12)
    r = VapAsrForStreamingASR(c, thinker=_Thinker()); r.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("encoder.")}, strict=True)
    r.encoder = m.encoder; wav, wl = _wav(); K = torch.tensor([10, 7])
    assert torch.equal(r.chunk_embed(r.encode(wav, wl, K)), m.chunk_embed(m.encode(wav, wl, K)))
    st = m.trainable_state(); assert set(st["adapter_tap"]) == set(m.adapter_tap.state_dict())
