"""N3D 이식 머리(G1, 2026-10-29, 사용자 "G1 도 구현해놓자" — wiki output-n3d-branch-20261029 4.2 절): 인코더가 하나인 화자 분리 머리.
우리 ASR 인코더(고정) 탭 → GraftAdapter(학습) → Nemotron 3 Diarization(N3D)의 layer 층 위·머리(고정). G0(n3d_graft_probe) 결과 layer 8 을 기본으로 한다.
- GraftAdapter: 탭 스트림 S 개(각 1024)를 이어 붙여 Linear(S·1024 → 512) + 인과 블록 n 개(LN → 깊이별 conv(왼쪽만) → 점별 Linear, LN → FFN — 둘 다 출력 0 초기화).
  Linear 를 능형 회귀 해(set_linear — G0 선형 이식과 같은 꼴)로 두면 붙인 직후 = 선형 이식이다. 출력은 N3D layer 층 입력 꼴이다
  (layer 0 이면 embed_norm 뒤, layer ≥ 1 이면 layer − 1 층 출력 — 잔차 흐름). 인과: 프레임 k 의 출력은 탭 ≤ k 만 본다(우리 탭은 블록 [56,3] 스트리밍 값).
- teacher(m, wav, wav_len, layer): N3D 오프라인 한 번 → (h_layer, 10 ms 확률, 80 ms 프레임 길이) — 학습 교사(창 통째).
- student_offline(m, h, length, layer): ĥ → layer 층 위·머리 → 10 ms 확률(가중치는 고정, gradient 는 ĥ 로 흐른다).
- stream(m, h, length, layer, sil, chunk, rc, …): 스트리밍(AOSC). 캐시·FIFO 가 ĥ 를 담고 걸음마다 [캐시, FIFO, 청크] 위에서 layer 층 위·머리만 돈다 —
  NeMo forward_streaming_step(동기·고해상도)과 같은 갱신(streaming_update). 캐시의 무음 자리(learnable_sil_emb — 0 층 입력 전 꼴)는 layer 층 입력 꼴의
  sil 로 바꿔 넣는다(학습 데이터에서 교사가 모두 무음이라 본 프레임의 h_layer 평균 — graft.pt). layer 0 · ĥ = 교사 h_0 · sil = embed_norm(learnable_sil_emb)
  이면 교사 스트리밍과 같다(tests/test_n3d_graft.py).
- GraftExpert: 평가에 붙는 학생(N3DExpert 와 같은 probs·act 꼴) — graft.pt(어댑터·층·탭·sil·init) + N3D(.nemo) + 인코더(graft.pt 의 init, 고정)."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .n3d import FRAME_FACTOR, load_n3d, run_layers, set_streaming


class _CausalBlock(nn.Module):
    def __init__(self, d: int, kernel: int, ffn: int):
        super().__init__()
        self.k = int(kernel)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, self.k, groups=d); self.pw = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        for lin in (self.pw, self.ff[-1]): nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)        # 붙인 직후 블록은 항등

    def forward(self, x):                                                                         # (B, K, d)
        y = F.conv1d(F.pad(self.ln1(x).transpose(1, 2), (self.k - 1, 0)), self.conv.weight, self.conv.bias, groups=self.conv.groups).transpose(1, 2)
        x = x + self.pw(F.gelu(y))
        return x + self.ff(self.ln2(x))


class GraftAdapter(nn.Module):
    """탭 (B, S, K, d_in) → N3D layer 층 입력 꼴 (B, K, d). 모듈 설명 참고."""

    def __init__(self, n_streams: int, d_in: int = 1024, d: int = 512, blocks: int = 3, kernel: int = 5, ffn: int = 2048):
        super().__init__()
        self.cfg = dict(n_streams=int(n_streams), d_in=int(d_in), d=int(d), blocks=int(blocks), kernel=int(kernel), ffn=int(ffn))
        self.inp = nn.Linear(int(n_streams) * int(d_in), int(d))
        self.blocks = nn.ModuleList([_CausalBlock(int(d), kernel, ffn) for _ in range(int(blocks))])

    def forward(self, feats):
        B, S, K, D = feats.shape
        x = self.inp(feats.permute(0, 2, 1, 3).reshape(B, K, S * D).to(self.inp.weight.dtype))
        for blk in self.blocks: x = blk(x)
        return x

    @torch.no_grad()
    def set_linear(self, W, b) -> None:
        """능형 회귀 해 W (S·d_in, d)(탭을 스트림 순서로 이어 붙인 기저), b (d,) 를 입력 Linear 로."""
        self.inp.weight.copy_(W.T.to(self.inp.weight)); self.inp.bias.copy_(b.to(self.inp.bias))


@torch.no_grad()
def teacher(m, wav, wav_len, layer: int):
    """N3D 오프라인 한 번(창 통째) → (h (B, T, 512) = layer 층 입력, p10 (B, T·8, 8) 10 ms 확률, elen (B,) 80 ms 프레임 길이)."""
    enc = m.encoder; got = {}
    mod = enc.embed_norm if layer == 0 else enc.layers[layer - 1]
    hook = mod.register_forward_hook(lambda mod_, i_, o: got.__setitem__("h", o if torch.is_tensor(o) else o[0]))
    try:
        with torch.autocast(wav.device.type, enabled=False):
            feats, flen = m.preprocessor(input_signal=wav.float(), length=wav_len.long())
        emb, elen = m.frontend_encoder(processed_signal=feats[:, :, : int(flen.max())], processed_signal_length=flen)
        p10 = m.forward_infer(emb, elen)
    finally:
        hook.remove()
    return got["h"], p10, elen


def student_offline(m, h, length, layer: int):
    """ĥ (B, T, 512) → 10 ms 확률 (B, T·8, 8)(N3D layer 층 위·머리 — 가중치 고정, gradient 는 ĥ 로)."""
    return m.forward_infer(run_layers(m, h, length, layer), length)


def pool80(p10, length=None):
    """10 ms 확률 → 80 ms(평균, NeMo downsample_preds — length 를 주면 패딩을 빼고 평균)."""
    from nemo.collections.asr.modules.sortformer_modules import SortformerModules
    return SortformerModules.downsample_preds(p10, FRAME_FACTOR, lengths=None if length is None else length * FRAME_FACTOR)


@torch.no_grad()
def stream(m, h, length, layer: int, sil, chunk: int = 4, rc: int = 0, fifo: int = 264, cache: int = 264, period: int = 222):
    """ĥ (B, T, 512)(인과), length (B,) 80 ms 프레임 → (B, T, 8) 80 ms 확률(도착 순서). NeMo 동기 스트리밍 고리와 같은 갱신(모듈 설명).
    줄마다 길이가 달라도 단독 실행과 같다(패딩 줄은 길이 뒤만 다르다 — n3d.stream_probs 와 같은 규약)."""
    set_streaming(m, chunk, rc, fifo, cache, period)
    sm = m.sortformer_modules; B, T, _ = h.shape; dev = h.device; L = length.to(dev).long()
    state = sm.init_streaming_state(batch_size=B, async_streaming=False, device=dev)
    old = sm.learnable_sil_emb.data; sm.learnable_sil_emb.data = sil.to(device=old.device, dtype=old.dtype)   # 캐시 무음 자리 = layer 층 입력 꼴
    out = []
    try:
        for s0 in range(0, T, int(chunk)):
            s1 = min(T, s0 + int(chunk)); r1 = min(T, s1 + int(rc))
            ch = h[:, s0:r1].float(); cl = (L - s0).clamp(0, r1 - s0)
            x = torch.cat([state.spkcache, state.fifo, ch], 1); xl = state.spkcache.shape[1] + state.fifo.shape[1] + cl
            hr = m.forward_infer(run_layers(m, x, xl, layer), xl)
            p = sm.apply_mask_to_preds(sm.downsample_preds(hr, m.upsample_factor) if m.high_resolution else hr, xl).float()
            state, cp = sm.streaming_update(streaming_state=state, chunk=ch, preds=p, lc=0, rc=r1 - s1)
            out.append(cp)
    finally:
        sm.learnable_sil_emb.data = old
    p = torch.cat(out, 1)[:, :T]
    return p * (torch.arange(T, device=dev)[None] < L[:, None])[..., None]


class GraftExpert:
    """평가용 학생(N3DExpert 와 같은 probs·act 꼴). graft.pt = dict(state, cfg, layer, taps, sil, init, n3d, step) — n3d_graft_train 이 쓴다."""

    def __init__(self, graft_path: str, n3d_path: Optional[str] = None, nemotron_dir: Optional[str] = None, chunk: int = 4, rc: int = 0, threshold: float = 0.5,
                 device: str = "cuda"):
        from .diar import load_encoder
        x = torch.load(graft_path, map_location="cpu")
        self.layer, self.taps, self.sil, self.chunk, self.rc, self.threshold = int(x["layer"]), tuple(x["taps"]), x["sil"].float(), int(chunk), int(rc), float(threshold)
        self.adapter = GraftAdapter(**x["cfg"]); self.adapter.load_state_dict(x["state"]); self.adapter.to(device).eval()
        self.m, _ = load_n3d(n3d_path or x["n3d"], device)
        self.enc, _ = load_encoder(x["init"], nemotron_dir, device)
        self.cfg = dict(graft=graft_path, layer=self.layer, chunk=self.chunk, rc=self.rc, threshold=self.threshold, step=x.get("step"))

    @torch.no_grad()
    def probs(self, wav, wav_len, K, offline: bool = False) -> torch.Tensor:
        """→ (B, max K, 8) 80 ms 확률 — 기본 스트리밍(청크 chunk), offline 이면 창 통째."""
        f = self.enc(wav, wav_len, K, taps=self.taps)
        with torch.autocast(wav.device.type, dtype=torch.bfloat16, enabled=wav.device.type == "cuda"):
            h = self.adapter(f)
            Km = int(K.max()); L = K.to(wav.device).long()
            p = pool80(student_offline(self.m, h, L, self.layer), L) if offline else stream(self.m, h.float(), L, self.layer, self.sil, self.chunk, self.rc)
        p = p.float()[:, :Km]
        return p * (torch.arange(Km, device=p.device)[None] < L[:, None])[..., None]

    def act(self, wav, wav_len, K, n: int, offline: bool = False) -> torch.Tensor:
        return (self.probs(wav, wav_len, K, offline)[..., :n] > self.threshold).float()
