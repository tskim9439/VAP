"""학습 루프 안에서 오디오 → Nemotron FastConformer [56,0] 특징을 만드는 온라인 인코더 (특징 캐시 대체, 2026-09-07).
지금은 동결(eval, requires_grad False)이지만 set_trainable(True) 로 풀어 encoder 까지 학습할 수 있다.
출력은 캐시 규약과 같다: 12.5 Hz, D=1024, 스트림당 K = round(duration·12.5) 프레임(모델 패딩으로 1–2 프레임 길면 자르고 짧으면 0 패딩)."""
import os, glob
from typing import Optional
import torch, torch.nn as nn

FRAME_HZ, SR, DIM = 12.5, 16000, 1024
def frames_for(duration_s: float) -> int: return int(round(duration_s * FRAME_HZ))


def _encoder_cache_path(nemo_path: str) -> Optional[str]:
    """인코더+전처리기만 담은 캐시(.pt) 경로. VAPASR_STAGE_DIR(로컬 디스크) 아래, .nemo 의 크기·mtime 으로 키를 만든다. 스테이지 디렉토리가 없으면 None."""
    root = os.environ.get("VAPASR_STAGE_DIR", "")
    if not root: return None
    try: os.makedirs(root, exist_ok=True)
    except OSError: return None
    st = os.stat(nemo_path); return os.path.join(root, f"nemotron-encoder-{st.st_size}-{int(st.st_mtime)}.pt")

def _load_encoder_cache(nemo_path: str):
    """restore_from(.nemo) 은 2.4 GB tar 를 풀고 디코더·joint 까지 만들어 로컬 디스크에서도 60 s 가 걸린다(2026-09-09). 캐시가 있으면 인코더·전처리기만 config 로 만들어 state_dict 를 얹는다(수 초)."""
    cp = _encoder_cache_path(nemo_path)
    if not cp or not os.path.exists(cp): return None
    try:
        import torch
        from omegaconf import OmegaConf
        from nemo.core.classes.common import Serialization
        d = torch.load(cp, map_location="cpu", weights_only=False)
        pre = Serialization.from_config_dict(OmegaConf.create(d["pre_cfg"])); enc = Serialization.from_config_dict(OmegaConf.create(d["enc_cfg"]))
        pre.load_state_dict(d["pre_state"]); enc.load_state_dict(d["enc_state"]); return pre, enc.float()
    except Exception as e:
        print(f"nemotron encoder 캐시 무시({type(e).__name__}: {e}) → restore_from", flush=True); return None

def _save_encoder_cache(nemo_path: str, m) -> None:
    cp = _encoder_cache_path(nemo_path)
    if not cp or os.path.exists(cp): return
    try:
        import torch
        from omegaconf import OmegaConf
        tmp = f"{cp}.{os.getpid()}.tmp"
        torch.save({"pre_cfg": OmegaConf.to_container(m.cfg.preprocessor, resolve=True), "enc_cfg": OmegaConf.to_container(m.cfg.encoder, resolve=True),
                    "pre_state": m.preprocessor.state_dict(), "enc_state": m.encoder.state_dict()}, tmp)
        os.replace(tmp, cp); print(f"nemotron encoder 캐시 저장 → {cp}", flush=True)
    except Exception as e: print(f"nemotron encoder 캐시 저장 실패({type(e).__name__}: {e})", flush=True)

class NemotronOnline(nn.Module):
    def __init__(self, right_context: int = 0, path: Optional[str] = None):
        super().__init__()
        local = sorted(glob.glob(os.path.join(path or os.environ.get("MXC_NEMOTRON_DIR", os.environ.get("NEMOTRON_DIR", "")), "*.nemo")))
        cached = _load_encoder_cache(local[0]) if local else None
        if cached is not None: self.pre, self.enc = cached
        else:
            import nemo.collections.asr as nemo_asr
            m = nemo_asr.models.ASRModel.restore_from(local[0], map_location="cpu") if local else nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu")
            self.pre, self.enc = m.preprocessor, m.encoder.float()
            if local: _save_encoder_cache(local[0], m)
            del m
        self.enc.set_default_att_context_size([56, right_context]); self.trainable = False; self.set_trainable(False)
        # NeMo ConformerEncoder.forward 는 update_max_seq_length 에서 torch.distributed.all_reduce(MAX) 를 기본 그룹에 날린다(rank 간 pos-enc 버퍼 길이 동기화).
        # 학습에서는 모든 rank 가 step 마다 한 번씩 불러 맞지만, 평가(rank 별 스트림 수가 다름)나 rank 0 단독 호출에서는 collective 가 어긋나 교착·NCCL 오류가 난다
        # (job 65963·65965·66066·66103·66201·66227·66260 의 평가 행, 2026-09-08 스택 덤프로 확인). 로컬 길이만 보고 버퍼를 키우도록 바꾼다.
        enc = self.enc
        if hasattr(enc, "sync_max_audio_length"): enc.sync_max_audio_length = False           # NeMo 3.x 공식 스위치
        else:                                                                                 # 구버전: 로컬 판정으로 대체
            def _update_max_seq_length_local(seq_length: int, device=None):
                if seq_length > enc.max_audio_length: enc.set_max_audio_length(seq_length)
            enc.update_max_seq_length = _update_max_seq_length_local
    def set_trainable(self, flag: bool):
        self.trainable = flag
        for p in self.enc.parameters(): p.requires_grad_(flag)
        self.enc.train(flag)
        # NeMo 의 fused Triton 서브샘플링(dw_striding) 커널은 backward 에서 autotune 벤치마크를 돌리다 메모리가 빠듯하면 "Triton Error [CUDA]: out of memory" 로 죽는다(2026-09-09 실측).
        # 인코더를 학습할 때는 순수 PyTorch 경로로 돌린다(fuse_triton=False). 동결·추론은 그대로.
        conv = getattr(getattr(self.enc, "pre_encode", None), "conv", None)
        if flag and conv is not None and getattr(conv, "fuse_triton", False):
            conv.fuse_triton = False; print("nemotron encoder 학습: 서브샘플링 Triton 커널 끔(fuse_triton=False)", flush=True)
        return self
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
