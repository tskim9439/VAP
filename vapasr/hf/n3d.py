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
- N3DExpert: 모델에 붙는 고정 전문가. 가중치는 체크포인트에 넣지 않는다(nn.Module 이 아니라 state_dict·parameters·DDP 밖) — config 의 경로
  (없으면 환경 변수 VAPASR_N3D_PATH)에서 처음 쓸 때 그 장치로 읽는다.
"""
import io, math, os, random, tarfile
from typing import Optional

import numpy as np
import torch

FRAME_FACTOR = 8                                                                                  # 10 ms 특징 8 개 = 80 ms(우리 인코더 프레임)
DEFAULTS = dict(chunk=4, rc=0, train_chunk=16, train_rc=0, fifo=264, cache=264, period=222, threshold=0.5, dtype="bf16")


def load_n3d(path: str, device: str = "cpu"):
    """.nemo → (SortformerEncLabelModel(eval, requires_grad False), 뺀 보조 키 목록). 생성자의 random.seed(42) 가 학습 난수를 건드리지 않게 상태를 되돌린다."""
    from omegaconf import open_dict
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.utils import logging as nlog
    lvl = nlog.getEffectiveLevel(); nlog.setLevel(nlog.ERROR)
    st = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        cfg = SortformerEncLabelModel.restore_from(path, return_config=True)
        with open_dict(cfg):
            if "BCEWithLogitsLoss" in str(cfg.loss.get("_target_", "")): cfg.loss = {"_target_": "nemo.collections.asr.losses.bce_loss.BCELoss", "reduction": "mean"}
            for k in ("activity_loss", "phantom_loss"): cfg.pop(k, None)
            for k in ("train_ds", "validation_ds", "test_ds"):
                if k in cfg and cfg[k] is not None: cfg[k].manifest_filepath = None
        m = SortformerEncLabelModel(cfg=cfg)
        with tarfile.open(path) as t:
            name = next(n for n in t.getnames() if n.endswith("model_weights.ckpt"))
            sd = torch.load(io.BytesIO(t.extractfile(name).read()), map_location="cpu", weights_only=True)
        aux = sorted(k for k in sd if k.startswith("sortformer_modules.activity_head."))
        for k in aux: sd.pop(k)
        m.load_state_dict(sd, strict=True)
    finally:
        random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2]); nlog.setLevel(lvl)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m.to(device), aux


def set_streaming(m, chunk: int, rc: int, fifo: int = 264, cache: int = 264, period: int = 222) -> None:
    """스트리밍 설정(80 ms 프레임 단위). 바뀔 때만 검사한다."""
    sm = m.sortformer_modules; want = (int(chunk), int(rc), int(fifo), int(cache), int(period), FRAME_FACTOR)
    if getattr(m, "_vap_stream", None) == want: return
    sm.chunk_len, sm.chunk_right_context, sm.fifo_len, sm.spkcache_len, sm.spkcache_update_period = want[:5]
    sm.chunk_left_context = 0; m.output_subsampling_factor = FRAME_FACTOR; m._check_streaming_parameters(); m._vap_stream = want


@torch.no_grad()
def stream_probs(m, wav, wav_len, chunk: int = 4, rc: int = 0, fifo: int = 264, cache: int = 264, period: int = 222, dtype: str = "bf16") -> torch.Tensor:
    """wav (B, L) 16 kHz, wav_len (B,) 샘플 수 → (B, T, 8) float32 화자 확률(80 ms, 도착 순서). 줄마다 길이 뒤는 0 — 스트리밍이라 줄의 출력은
    그 줄 길이 안의 오디오(와 청크 경계까지)만 본다. B 줄은 같은 걸음으로 가지만(sync) 캐시·FIFO 갱신은 줄마다 따로다."""
    set_streaming(m, chunk, rc, fifo, cache, period)
    dev = next(m.parameters()).device; sm = m.sortformer_modules; B = wav.shape[0]
    with torch.autocast(dev.type, enabled=False):
        feats, flen = m.preprocessor(input_signal=wav.to(dev).float(), length=wav_len.to(dev).long())
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

    def path(self) -> str:
        p = os.environ.get("VAPASR_N3D_PATH") or self.cfg.get("path")
        assert p and os.path.exists(p), f"Nemotron 3 Diarization .nemo 가 없다: {p!r}(config n3d_cfg.path 또는 VAPASR_N3D_PATH)"
        return p

    def load(self, device):
        if self.m is None: self.m, self.aux = load_n3d(self.path(), "cpu")
        if next(self.m.parameters()).device != torch.device(device): self.m = self.m.to(device)
        return self.m

    def probs(self, wav, wav_len, K, train: bool = False) -> torch.Tensor:
        m = self.load(wav.device); c = self.cfg
        ch, rc = (c["train_chunk"], c["train_rc"]) if train else (c["chunk"], c["rc"])
        p = stream_probs(m, wav, wav_len, ch, rc, c["fifo"], c["cache"], c["period"], c["dtype"])
        Km = int(K.max()); p = p[:, :Km] if p.shape[1] >= Km else torch.nn.functional.pad(p, (0, 0, 0, Km - p.shape[1]))
        return p * (torch.arange(Km, device=p.device)[None] < K.to(p.device)[:, None])[..., None]

    def act(self, wav, wav_len, K, n: int, train: bool = False) -> torch.Tensor:
        """→ (B, max K, n) 0/1 — 임계(cfg threshold, Nemotron 3.5 Transcribe 처럼 0.5)를 넘은 앞 n 채널(도착 순서 = 우리 슬롯 순서)."""
        return (self.probs(wav, wav_len, K, train)[..., :n] > float(self.cfg["threshold"])).float()
