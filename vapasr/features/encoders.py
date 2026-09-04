"""Frozen encoder 공통 인터페이스. encode(wav (2,T) @16k float32) → np.float16 (2, T', D).

이름            frame_hz  D     lookahead(ms, 감사 실측)   비고
cpc             50        256   0                          원 VAP EncoderCPC (causal)
nemotron-c0     12.5      1024  ≤80                        FastConformer att_context [56,0]
nemotron-c1     12.5      1024  ≤160                       [56,1]
wavlm-base      50        768   ∞ (창 20 s 내 양방향)       비인과 참조 — 20 s 독립 창
wavlm-large     50        1024  ∞ (창 20 s 내 양방향)       비인과 참조
qwen-aut-cc1s   13.0*     1024  0–800 (평균 420)            chunked-causal 1 s 블록, 좌측 8 s 창 (학습 블록 크기)
qwen-aut-causal 13.0*     1024  ≤80                        프레임 causal, 좌측 8 s 창 (OOD, WER +23 %)
  * AuT 는 1 s chunk 당 13 프레임을 낸다 (12.5 가 아님). 1 s 경계에 정렬되므로 초 단위 세그먼트에서 exact.
긴 파일은 SEG_S 초 세그먼트 + 좌측 CTX_S 겹침으로 처리하고 겹침 구간 출력은 버린다.
주의 — 층을 쌓으면 유효 수용장이 층수 × 층당 좌측 context 로 늘어난다: Nemotron 56 frame × 24 층 = 1,344 frame ≈ 107 s.
2026-09-03 진단: 20 s 겹침에서는 세그먼트 시작 후 ~100 s 가 무분할과 달랐다(rel max 3.7). 그래서 CTX_S = 120 s.
세그먼트 경계는 80 ms 격자(1280 샘플)에 맞춘다 — 40 ms 어긋나면 전 프레임이 바뀐다.
"""
import os, sys, math
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np, torch

SR = 16000
SEG_S, CTX_S = 300.0, 120.0
GRID = 1280   # 80 ms @16k — 세그먼트 경계 정렬 단위 (12.5 Hz 와 50 Hz 모두의 공배수)

@dataclass
class Encoder:
    name: str; frame_hz: float; dim: int; lookahead_ms: str; causal: bool
    _fn: Callable[[torch.Tensor], torch.Tensor]        # (B, T) wav → (B, T', D) float32 (device)  — frontend 가 있으면 (B, F, Tf) feats → (B, T', D)
    seg_s: float = SEG_S; ctx_s: float = CTX_S; independent_windows: bool = False
    frontend: Optional[Callable[[torch.Tensor], torch.Tensor]] = None   # (B, T) wav → (B, F, Tf) 파일 단위 1회 (utterance 정규화가 있는 프론트엔드용)
    feat_hz: float = 100.0; feat_grid: int = 100                        # frontend 프레임율과 세그먼트 경계 격자(프레임)
    device: str = "cuda"
    module: Optional[torch.nn.Module] = None                            # .to(device) 용 (있으면)

    def to(self, device: str):
        self.device = device
        if self.module is not None: self.module.to(device)
        return self

    @torch.inference_mode()
    def encode(self, wav: np.ndarray, dtype=np.float16) -> np.ndarray:
        """wav (2, T) → (2, T', D) dtype(기본 fp16). 채널을 배치로 묶어 한 번에."""
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).to(self.device)
        n_frames_total = int(round(x.shape[1] / SR * self.frame_hz)); outs = []
        if self.frontend is not None:                       # 특징을 파일 단위로 한 번 만들고 특징 축에서 자른다
            x = self.frontend(x); rate, grid = self.feat_hz, self.feat_grid
        else:
            rate, grid = float(SR), GRID
        T = x.shape[-1]; seg, ctx = int(self.seg_s * rate) // grid * grid, int(self.ctx_s * rate) // grid * grid
        start = 0
        while start < T:
            end = min(T, start + seg)
            if self.independent_windows:
                h = self._fn(x[..., start:end]); keep_from = 0
            else:
                a = max(0, start - ctx); h = self._fn(x[..., a:end]); keep_from = int(round((start - a) / rate * self.frame_hz))
            # 모델별 패딩으로 출력이 1~2 프레임 길 수 있다 → 세그먼트가 담당하는 구간 길이만큼만 남긴다 (경계 정렬)
            expect = int(round((end - start) / rate * self.frame_hz))
            outs.append(h[:, keep_from: keep_from + expect].float().cpu()); start = end
        h = torch.cat(outs, 1)[:, :n_frames_total]
        return h.numpy().astype(dtype)

def _pad_to(x, n):  # 프레임 수 정합 (모델별 반올림 차이 보정)
    if x.shape[1] >= n: return x[:, :n]
    return torch.nn.functional.pad(x, (0, 0, 0, n - x.shape[1]))

# ───────────────────────────── 로더들 ─────────────────────────────
def _load_cpc():
    import glob
    repo = os.path.join(os.environ.get("DATA_ROOT", "/data3/tskim"), "third_party", "VoiceActivityProjection"); cwd = os.getcwd(); os.chdir(repo)
    from vap.model import VapConfig, VapGPT, load_older_state_dict
    m = VapGPT(VapConfig()); m.load_state_dict(load_older_state_dict(sorted(glob.glob("example/VAP_*.ckpt"))[-1]), strict=False); os.chdir(cwd)
    enc = m.encoder.cuda().eval().float()
    def fn(x): return enc(x)                                  # (B, T', 256)
    return Encoder("cpc", 50.0, 256, "0", True, fn, ctx_s=60.0, module=enc)   # GRU 상태 잔차: 20 s 에서 rel 7e-3 → 60 s

def _load_nemotron(right: int):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu").cuda().eval().float()
    m.encoder.set_default_att_context_size([56, right]); pre, enc = m.preprocessor, m.encoder
    def fn(x):
        L = torch.full((x.shape[0],), x.shape[1], device=x.device)
        f, fl = pre(input_signal=x, length=L); h, hl = enc(audio_signal=f, length=fl)   # (B, D, T')
        return h.transpose(1, 2)
    return Encoder(f"nemotron-c{right}", 12.5, 1024, f"≤{80*(right+1)}", True, fn, seg_s=270.0, ctx_s=120.0)   # seg+ctx < pos_emb_max_len(5000 frame = 400 s)

def _load_wavlm(size: str):
    from transformers import WavLMModel
    mid = {"base": "microsoft/wavlm-base-plus", "large": "microsoft/wavlm-large"}[size]
    m = WavLMModel.from_pretrained(mid).cuda().eval().float()
    def fn(x):
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-7)     # Wav2Vec2FeatureExtractor 정규화와 동일
        return m(x).last_hidden_state                                             # (B, T', D) @50 Hz
    return Encoder(f"wavlm-{size}", 50.0, m.config.hidden_size, "∞ (20 s 창)", False, fn, seg_s=20.0, independent_windows=True)

def _load_qwen(mode: str):
    import torch.nn as nn
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.qwen_aut_mask import patch_aut
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cuda", max_new_tokens=8)
    root = next(v for v in vars(m).values() if isinstance(v, nn.Module)); enc = root.thinker.audio_tower.float().eval(); fe = m.processor.feature_extractor
    mod = sys.modules[type(enc).__module__]; gl = getattr(enc, "_get_feat_extract_output_lengths", None) or mod._get_feat_extract_output_lengths
    # 마스크: 학습 블록(8 s)에 맞춰 좌측 7 블록(=8 s 창)으로 제한 → 층당 수용장 8 s, 18 층 = 144 s → ctx 150 s 면 exact
    patch_aut(enc, mode={"cc1s": "chunked-causal", "causal": "causal"}[mode], block_fbank_frames=100, left_ctx_blocks=7)
    def frontend(x):                                           # 파일 단위 1회: Whisper FE 는 utterance max 정규화라 세그먼트별로 하면 스케일이 달라진다
        feats = []
        for ch in x:
            o = fe(ch.cpu().numpy(), sampling_rate=SR, return_tensors="pt", padding="longest", truncation=False, return_attention_mask=True)
            Tm = int(o.attention_mask[0].sum()); feats.append(o.input_features[0, :, :Tm])
        n = min(f.shape[-1] for f in feats); return torch.stack([f[:, :n] for f in feats]).cuda()     # (2, 128, Tf) @100 Hz
    def fn(f):                                                 # (2, 128, Tf) → (2, T', D). AuT forward 는 단일 시퀀스 → 채널별
        outs = []
        for ch in f:
            L = torch.tensor([ch.shape[-1]], device="cuda"); r = enc(ch.contiguous(), feature_lens=L, aftercnn_lens=gl(L))
            h = r.last_hidden_state if hasattr(r, "last_hidden_state") else r[0]; outs.append(h.reshape(-1, h.shape[-1]))
        n = min(o.shape[0] for o in outs); return torch.stack([o[:n] for o in outs])
    la = {"cc1s": "0–800 (평균 420)", "causal": "≤80"}[mode]
    # 실제 출력률 13 Hz: 1 s conv chunk(100 fbank 프레임) 당 ceil(100/8)=13 프레임 (기술 보고서의 "12.5 Hz" 와 다름 — 2026-09-03 실측 304 s → 3954)
    return Encoder(f"qwen-aut-{mode}", 13.0, 1024, la, mode == "causal", fn, seg_s=240.0, ctx_s=150.0, frontend=frontend, feat_hz=100.0, feat_grid=100)

def _load_fbank():
    """사전학습 없음 floor: 80-mel log fbank @50 Hz (hop 320). probe 가 스스로 학습한 양의 기준선."""
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, win_length=400, hop_length=320, n_mels=80).cuda()
    def fn(x): return torch.log(mel(x) + 1e-6).transpose(1, 2)          # (B, T', 80)
    return Encoder("fbank", 50.0, 80, "0 (창 25 ms)", True, fn, seg_s=600.0, ctx_s=1.0, module=mel)

def _load_qwen_block8s():
    """U0.5 증류 타깃: AuT 를 학습 분포(8 s 블록 양방향, 블록 간 독립)로 실행. 비인과 — 타깃 전용. 8 s 격자 정렬 세그먼트로 exact."""
    import torch.nn as nn
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from experiments.qwen_aut_mask import patch_aut
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cuda", max_new_tokens=8)
    root = next(v for v in vars(m).values() if isinstance(v, nn.Module)); enc = root.thinker.audio_tower.float().eval(); fe = m.processor.feature_extractor
    mod = sys.modules[type(enc).__module__]; gl = getattr(enc, "_get_feat_extract_output_lengths", None) or mod._get_feat_extract_output_lengths
    patch_aut(enc, mode="block", block_fbank_frames=800)
    def frontend(x):
        feats = []
        for ch in x:
            o = fe(ch.cpu().numpy(), sampling_rate=SR, return_tensors="pt", padding="longest", truncation=False, return_attention_mask=True)
            Tm = int(o.attention_mask[0].sum()); feats.append(o.input_features[0, :, :Tm])
        n = min(f.shape[-1] for f in feats); return torch.stack([f[:, :n] for f in feats]).cuda()
    def fn(f):
        outs = []
        for ch in f:
            L = torch.tensor([ch.shape[-1]], device="cuda"); r = enc(ch.contiguous(), feature_lens=L, aftercnn_lens=gl(L))
            h = r.last_hidden_state if hasattr(r, "last_hidden_state") else r[0]; outs.append(h.reshape(-1, h.shape[-1]))
        n = min(o.shape[0] for o in outs); return torch.stack([o[:n] for o in outs])
    return Encoder("qwen-aut-block8s", 13.0, 1024, "≤8000 (블록)", False, fn, seg_s=240.0, ctx_s=0.0, independent_windows=True, frontend=frontend, feat_hz=100.0, feat_grid=800)

ENCODERS: Dict[str, Callable[[], Encoder]] = {
    "qwen-aut-block8s": _load_qwen_block8s, "fbank": _load_fbank, "cpc": _load_cpc, "nemotron-c0": lambda: _load_nemotron(0), "nemotron-c1": lambda: _load_nemotron(1),
    "wavlm-base": lambda: _load_wavlm("base"), "wavlm-large": lambda: _load_wavlm("large"),
    "qwen-aut-cc1s": lambda: _load_qwen("cc1s"), "qwen-aut-causal": lambda: _load_qwen("causal"),
}
def load_encoder(name: str) -> Encoder:
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # 세그먼트 간 수치 일관성
    return ENCODERS[name]()
