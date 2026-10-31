"""Nemotron 3 Diarization 고정 가지(2026-10-29, 사용자 "Nemotron 3 Diarization 을 고정 가지로 넣되 하나로 통합할 방법도 고안해놔"
— wiki output-n3d-branch-20261029). NVIDIA Nemotron 3.5 Transcribe 의 Parallel Expert Encoder 와 같은 쓰임이다: 학습하지 않는 화자 분리 전문가가
같은 오디오에서 80 ms 프레임마다 화자 활동(도착 순서 8 채널)을 내고, 모델은 그것을 슬롯 활동 입력(E6·E9·E10 경로)으로 쓴다.
- load_n3d(path): .nemo(NeMo 3.0 저장) → SortformerEncLabelModel(eval·동결). 서버·맥 NeMo 3.1.0(6989ae9fc)에는 저장 config 의 손실 대상
  (bce_loss.BCEWithLogitsLoss)과 보조 손실(activity·phantom)이 없어 BCELoss 로 바꾸고 뺀다 — 추론에는 쓰지 않는다. 가중치는 보조 activity_head
  (무음·한 명·겹침 3 분류 학습 손실용 — NeMo main 에서도 화자 확률 계산과 무관)만 빼고 엄격히(strict) 올린다.
- stream_probs(m, wav, wav_len, ...): NeMo forward_streaming 과 같은 고리(AOSC 화자 캐시 + FIFO)로 (B, T, 8) 확률을 낸다. 10 ms 고해상도 출력을
  80 ms 로 평균한다(output_subsampling_factor 8). 다른 점 둘: 분산 all_reduce 를 하지 않는다(평가 rank 마다 길이가 달라 collective 가 어긋나지 않게),
  process_signal 대신 전처리기를 바로 부른다(그 함수는 부를 때마다 torch.cuda.empty_cache 를 한다). 스트리밍이라 진폭 정규화는 원래 없다.
- 청크(80 ms 프레임 단위): 추론 기본 chunk 4 · rc 0 = 우리 인코더 블록([56,3], 320 ms)과 같은 경계 — 블록 오디오가 다 오면 인코더 출력과 같은 때에
  활동이 나온다(지연을 더하지 않는다. NVIDIA 다화자 스트리밍 예제도 화자 분리 청크를 ASR 청크에 맞춘다). 학습은 기본 chunk 16(고리 수 1/4) —
  모델 카드의 지연별 DER 차이(AMI SDM 0.32 s 12.95 · 1.04 s 12.80)가 작아 학습·추론 활동 품질 차이가 작다.
- use_sdpa(로드 기본): 주의를 FlexAttention(torch.compile — 길이마다 다시 컴파일) 대신 같은 계산의 SDPA 로, 분산 all_reduce(sync_max_audio_length)는 끈다.
- N3DExpert: 모델에 붙는 고정 전문가. 가중치는 체크포인트에 넣지 않는다(nn.Module 이 아니라 state_dict·parameters·DDP 밖) — config 의 경로
  (없으면 환경 변수 VAPASR_N3D_PATH)에서 처음 쓸 때 그 장치로 읽는다.
  흡수(2026-10-31, VapAsr.absorb_n3d — wiki output-diarizer-replacement-20261031 길 3)면 m 은 모델 안 가지(n3d_net — 체크포인트에 함께 저장, build_n3d 로 틀을 만든다),
  mel 은 우리 인코더 전처리기(pre — 두 .nemo 의 전처리 설정이 같다)로 만든다. 출력은 고정 전문가와 같다.
"""
import io, math, os, random, tarfile
from typing import Optional

import numpy as np
import torch

FRAME_FACTOR = 8                                                                                  # 10 ms 특징 8 개 = 80 ms(우리 인코더 프레임)
DEFAULTS = dict(chunk=4, rc=0, train_chunk=16, train_rc=0, fifo=264, cache=264, period=222, threshold=0.5, dtype="bf16")


def use_sdpa(m) -> None:
    """N3D 인코더의 주의를 FlexAttention 대신 SDPA 로(인스턴스만 — 다른 NeMo 모듈은 그대로). NeMo TransformerEncoder 는 CUDA 에서
    torch.compile(flex_attention, dynamic=True)을 쓰는데, 스트리밍처럼 길이가 바뀌면 블록 수마다 다시 컴파일한다(2026-10-29 서버 GPU: 첫 forward 617 s,
    26 s 창 활동 55 s — 다시 컴파일 한도 8 을 넘었다). N3D 는 attn_mode full + RoPE 라 마스크가 키 패딩(kv < 길이, _make_padding_mod)뿐이어서 SDPA 의
    키 패딩 마스크와 같은 계산이다. 함께 encoder.sync_max_audio_length 를 끈다 — 켜 두면 forward 마다 분산 all_reduce 를 해서 rank 마다 고리 수가 다른
    학습·평가에서 collective 가 어긋난다(우리 ASR 인코더에 2026-09-08 고친 것과 같은 문제)."""
    import types
    import torch.nn.functional as F
    enc = m.encoder; attns = [layer.attn for layer in enc.layers]
    assert enc.attn_mode == "full" and enc.self_attention_model == "rope" and type(enc.pre_encode).__name__ == "FeatureStacking", "N3D(feature_stacking·rope·full) 전용"
    assert all(a._uses_rope and not a._uses_rel_pos for a in attns)

    def attn_forward(self, x, block_mask=None, pos_emb=None):
        B, T, _ = x.shape; H, D = self.n_heads, self.head_dim
        q, k, v = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm: q = self.q_norm(q).to(v.dtype); k = self.k_norm(k).to(v.dtype)
        q, k = self.rope(q, k)
        keep = (torch.arange(T, device=x.device)[None] < self._vap_len.to(x.device)[:, None])[:, None, None, :]
        return self.out_proj(F.scaled_dot_product_attention(q, k, v, attn_mask=keep).transpose(1, 2).reshape(B, T, self.d_model))

    def forward_internal(self, audio_signal, length, bypass_pre_encode=False):                    # NeMo forward_internal 의 N3D 경로(블록 마스크만 뺐다)
        if length is None:
            length = audio_signal.new_full((audio_signal.size(0),), audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1), dtype=torch.int64)
        if bypass_pre_encode: x = audio_signal
        else: x, length = self.pre_encode(audio_signal, length)
        length = length.to(torch.int64)
        if self.xscale: x = x * self.xscale
        x = self.embed_norm(self.dropout_pre_encoder(x))
        for a in attns: a._vap_len = length
        for layer in self.layers: x = layer(x)
        x = self.final_norm(x)
        if self.out_proj is not None: x = self.out_proj(x)
        return x.transpose(1, 2), length

    for a in attns: a.forward = types.MethodType(attn_forward, a)
    enc.forward_internal = types.MethodType(forward_internal, enc); enc.sync_max_audio_length = False


def run_layers(m, x, length, start: int = 0):
    """use_sdpa 를 건 N3D 에서 start 층 입력 x (B, T, 512)(embed_norm 뒤 = 0 층 입력) → 층 start… 뒤 머리 입력 (B, T, 192)(final_norm·encoder_proj) — 이식 시험용."""
    enc = m.encoder
    for layer in enc.layers: layer.attn._vap_len = length
    for layer in enc.layers[start:]: x = layer(x)
    x = enc.final_norm(x)
    if enc.out_proj is not None: x = enc.out_proj(x)
    return m.sortformer_modules.encoder_proj(x) if m.sortformer_modules.encoder_proj is not None else x


def build_n3d(cfg=None, path: Optional[str] = None):
    """N3D 틀(가중치는 무작위 초기화) → (SortformerEncLabelModel, 설정). cfg(손본 NeMo 설정 — 흡수 체크포인트의 n3d_config.yaml)가 없으면 path(.nemo)의 설정을 읽어 손본다
    (모듈 설명의 손실 대상·보조 손실). 생성자의 random.seed(42) 가 학습 난수를 건드리지 않게 상태를 되돌린다."""
    from omegaconf import open_dict
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.utils import logging as nlog
    lvl = nlog.getEffectiveLevel(); nlog.setLevel(nlog.ERROR)
    st = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        if cfg is None:
            cfg = SortformerEncLabelModel.restore_from(path, return_config=True)
            with open_dict(cfg):
                if "BCEWithLogitsLoss" in str(cfg.loss.get("_target_", "")): cfg.loss = {"_target_": "nemo.collections.asr.losses.bce_loss.BCELoss", "reduction": "mean"}
                for k in ("activity_loss", "phantom_loss"): cfg.pop(k, None)
                for k in ("train_ds", "validation_ds", "test_ds"):
                    if k in cfg and cfg[k] is not None: cfg[k].manifest_filepath = None
        m = SortformerEncLabelModel(cfg=cfg)
    finally:
        random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2]); nlog.setLevel(lvl)
    return m, cfg


def load_n3d(path: str, device: str = "cpu", sdpa: bool = True):
    """.nemo → (SortformerEncLabelModel(eval, requires_grad False), 뺀 보조 키 목록). 틀은 build_n3d(난수 상태를 되돌린다).
    sdpa(기본): use_sdpa — FlexAttention 다시 컴파일과 분산 all_reduce 를 피한다(끄면 NeMo 그대로 — 대조 시험용)."""
    m, _ = build_n3d(path=path)
    with tarfile.open(path) as t:
        name = next(n for n in t.getnames() if n.endswith("model_weights.ckpt"))
        sd = torch.load(io.BytesIO(t.extractfile(name).read()), map_location="cpu", weights_only=True)
    aux = sorted(k for k in sd if k.startswith("sortformer_modules.activity_head."))
    for k in aux: sd.pop(k)
    m.load_state_dict(sd, strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    if sdpa: use_sdpa(m)
    return m.to(device), aux


def set_streaming(m, chunk: int, rc: int, fifo: int = 264, cache: int = 264, period: int = 222) -> None:
    """스트리밍 설정(80 ms 프레임 단위). 바뀔 때만 검사한다."""
    sm = m.sortformer_modules; want = (int(chunk), int(rc), int(fifo), int(cache), int(period), FRAME_FACTOR)
    if getattr(m, "_vap_stream", None) == want: return
    sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_len, sm.spkcache_update_period = want[:5]
    sm.chunk_left_context = 0; m.output_subsampling_factor = FRAME_FACTOR; m._check_streaming_parameters(); m._vap_stream = want


@torch.no_grad()
def stream_probs(m, wav, wav_len, chunk: int = 4, rc: int = 0, fifo: int = 264, cache: int = 264, period: int = 222, dtype: str = "bf16", pre=None) -> torch.Tensor:
    """wav (B, L) 16 kHz, wav_len (B,) 샘플 수 → (B, T, 8) float32 화자 확률(80 ms, 도착 순서). 줄마다 길이 뒤는 0 — 스트리밍이라 줄의 출력은
    그 줄 길이 안의 오디오(와 청크 경계까지)만 본다. B 줄은 같은 걸음으로 가지만(sync) 캐시·FIFO 갱신은 줄마다 따로다.
    pre(흡수): mel 을 이 전처리기(우리 인코더 것 — 설정이 같다)로 만든다. 없으면 N3D 자신의 것."""
    set_streaming(m, chunk, rc, fifo, cache, period)
    dev = next(m.parameters()).device; sm = m.sortformer_modules; B = wav.shape[0]
    with torch.autocast(dev.type, enabled=False):
        feats, flen = (m.preprocessor if pre is None else pre)(input_signal=wav.to(dev).float(), length=wav_len.to(dev).long())
    feats = feats[:, :, : int(flen.max())]
    state = sm.init_streaming_state(batch_size=B, async_streaming=False, device=dev)
    total = torch.zeros((B, 0, sm.n_spk), device=dev); off = torch.zeros((B,), dtype=torch.long, device=dev)
    use_amp = dev.type == "cuda" and dtype != "fp32"
    with torch.autocast(dev.type, dtype=torch.bfloat16 if dtype == "bf16" else torch.float16, enabled=use_amp):
        for _, ch, ln, lo, ro in sm.streaming_feat_loader(feat_seq=feats, feat_seq_length=flen, feat_seq_offset=off):
            state, total = m.forward_streaming_step(processed_signal=ch, processed_signal_length=ln, streaming_state=state, total_preds=total, left_offset=lo, right_offset=ro)
    T = math.ceil(feats.shape[2] / FRAME_FACTOR); out = total[:, :T].float()
    valid = torch.arange(out.shape[1], device=dev)[None] < torch.div(flen + FRAME_FACTOR - 1, FRAME_FACTOR, rounding_mode="floor")[:, None]
    return out * valid[..., None]


class N3DExpert:
    """고정 전문가(체크포인트 밖). cfg = DEFAULTS 키 + path. probs(wav, wav_len, K, train) → (B, max K, 8) — 프레임 수를 우리 K 에 맞춘다(넘치면 자르고 모자라면 0)."""

    def __init__(self, cfg: dict):
        self.cfg = {**DEFAULTS, **dict(cfg)}; self.m = None; self.aux = None
        self.pre = None                                                                           # 흡수(VapAsr.absorb_n3d): m 은 모델 안 가지 n3d_net, pre 는 우리 인코더 전처리기

    def path(self) -> str:
        p = os.environ.get("VAPASR_N3D_PATH") or self.cfg.get("path")
        assert p and os.path.exists(p), f"Nemotron 3 Diarization .nemo 가 없다: {p!r}(config n3d_cfg.path 또는 VAPASR_N3D_PATH)"
        return p

    def load(self, device):
        if self.m is None: self.m, self.aux = load_n3d(self.path(), "cpu")
        if next(self.m.parameters()).device != torch.device(device): self.m = self.m.to(device)
        if self.m.training: self.m.eval()                                                          # 흡수 가지는 모델 train() 을 따라 바뀌므로 늘 eval 로(dropout 없음)
        return self.m

    def probs(self, wav, wav_len, K, train: bool = False) -> torch.Tensor:
        m = self.load(wav.device); c = self.cfg; pre = self.pre
        ch, rc = (c["train_chunk"], c["train_rc"]) if train else (c["chunk"], c["rc"])
        was = pre is not None and pre.training                                                    # 공유 전처리기도 eval 로(학습 중 dither 없이 — 고정 전문가와 같은 mel)
        if was: pre.eval()
        try: p = stream_probs(m, wav, wav_len, ch, rc, c["fifo"], c["cache"], c["period"], c["dtype"], pre=pre)
        finally:
            if was: pre.train()
        Km = int(K.max()); p = p[:, :Km] if p.shape[1] >= Km else torch.nn.functional.pad(p, (0, 0, 0, Km - p.shape[1]))
        return p * (torch.arange(Km, device=p.device)[None] < K.to(p.device)[:, None])[..., None]

    def act(self, wav, wav_len, K, n: int, train: bool = False) -> torch.Tensor:
        """→ (B, max K, n) 0/1 — 임계(cfg threshold, Nemotron 3.5 Transcribe 처럼 0.5)를 넘은 앞 n 채널(도착 순서 = 우리 슬롯 순서)."""
        return (self.probs(wav, wav_len, K, train)[..., :n] > float(self.cfg["threshold"])).float()
