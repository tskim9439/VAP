"""학습 루프 안에서 오디오 → Nemotron FastConformer [56,0] 특징을 만드는 온라인 인코더 (특징 캐시 대체, 2026-09-07).
지금은 동결(eval, requires_grad False)이지만 set_trainable(True) 로 풀어 encoder 까지 학습할 수 있다.
출력은 캐시 규약과 같다: 12.5 Hz, D=1024, 스트림당 K = round(duration·12.5) 프레임(모델 패딩으로 1–2 프레임 길면 자르고 짧으면 0 패딩)."""
import os, glob
from typing import Optional
import torch, torch.nn as nn

FRAME_HZ, SR, DIM = 12.5, 16000, 1024
def frames_for(duration_s: float) -> int: return int(round(duration_s * FRAME_HZ))

class NemotronOnline(nn.Module):
    def __init__(self, right_context: int = 0, path: Optional[str] = None):
        super().__init__()
        import nemo.collections.asr as nemo_asr
        local = sorted(glob.glob(os.path.join(path or os.environ.get("MXC_NEMOTRON_DIR", os.environ.get("NEMOTRON_DIR", "")), "*.nemo")))
        m = nemo_asr.models.ASRModel.restore_from(local[0], map_location="cpu") if local else nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu")
        m.encoder.set_default_att_context_size([56, right_context]); self.pre, self.enc = m.preprocessor, m.encoder.float()
        del m; self.trainable = False; self.set_trainable(False)
    def set_trainable(self, flag: bool):
        self.trainable = flag
        for p in self.enc.parameters(): p.requires_grad_(flag)
        self.enc.train(flag); return self
    def train(self, mode: bool = True):                    # 동결 중에는 dropout 등이 켜지지 않도록 eval 유지
        super().train(mode); self.enc.train(mode and self.trainable); return self
    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """wav (B, L) float32, wav_len (B,) 샘플 수, K (B,) 목표 프레임 수 → (B, max K, D) float32. 인코더는 fp32 로 돈다(autocast 밖)."""
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad(): f, fl = self.pre(input_signal=wav.float(), length=wav_len)
            ctx = torch.no_grad() if not self.trainable else torch.enable_grad()
            with ctx: h, hl = self.enc(audio_signal=f, length=fl)              # (B, D, T')
        h = h.transpose(1, 2); Km = int(K.max()); out = h.new_zeros(h.shape[0], Km, h.shape[2])
        for i in range(h.shape[0]):
            k, t = int(K[i]), int(hl[i]); n = min(k, t); out[i, :n] = h[i, :n]
        return out
