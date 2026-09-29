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

def _load_encoder_cache_file(cp: str):
    """캐시 파일 경로를 직접 주고 (pre, enc) 를 만든다. 실패하면 None."""
    try:
        import torch
        from omegaconf import OmegaConf
        from nemo.core.classes.common import Serialization
        d = torch.load(cp, map_location="cpu", weights_only=False)
        pre = Serialization.from_config_dict(OmegaConf.create(d["pre_cfg"])); enc = Serialization.from_config_dict(OmegaConf.create(d["enc_cfg"]))
        pre.load_state_dict(d["pre_state"]); enc.load_state_dict(d["enc_state"]); return pre, enc.float()
    except Exception as e:
        print(f"nemotron encoder 캐시 무시({type(e).__name__}: {e}) → restore_from", flush=True); return None

def _load_encoder_cache(nemo_path: str):
    """restore_from(.nemo) 은 2.4 GB tar 를 풀고 디코더·joint 까지 만들어 로컬 디스크에서도 60 s 가 걸린다(2026-09-09). 캐시가 있으면 인코더·전처리기만 config 로 만들어 state_dict 를 얹는다(수 초)."""
    cp = _encoder_cache_path(nemo_path)
    if not cp or not os.path.exists(cp): return None
    return _load_encoder_cache_file(cp)

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

class SpecAugment(nn.Module):
    """SpecAugment(Park et al. 2019) — log-mel (B, F, T) 에 주파수·시간 마스크. 모듈이 training 일 때만 적용한다(추론·평가는 그대로).
    마스크 값 0 = NeMo SpectrogramAugmentation 기본값. Nemotron 사전학습 설정(.nemo model_config: freq 2×27, time 10×0.05, preprocessor normalize NA)이
    그렇게 학습했으므로 인코더에는 분포 안의 입력이다. time_width < 1 이면 발화(유효 프레임 fl) 길이 비율, ≥ 1 이면 프레임 수. 폭은 [0, 최대] 균등, 시작은 유효 구간 안 균등.
    마스크를 한 번에 만든다(배치·마스크 수만큼 커널을 띄우지 않는다). 난수는 torch 기본 생성기(학습 시드를 따른다)."""
    PRESETS = {"off": None, "light": (2, 27, 5, 0.05), "nemo": (2, 27, 10, 0.05)}

    def __init__(self, freq_masks: int = 2, freq_width: int = 27, time_masks: int = 5, time_width: float = 0.05):
        super().__init__()
        self.freq_masks, self.freq_width, self.time_masks, self.time_width = int(freq_masks), int(freq_width), int(time_masks), float(time_width)

    @classmethod
    def from_spec(cls, spec: str) -> Optional["SpecAugment"]:
        """'off' | 'light' | 'nemo' | 'Fm,Fw,Tm,Tw' → 모듈 또는 None(끔)."""
        if spec in cls.PRESETS: p = cls.PRESETS[spec]; return None if p is None else cls(*p)
        fm, fw, tm, tw = spec.split(","); return cls(int(fm), int(fw), int(tm), float(tw))

    def extra_repr(self): return f"freq {self.freq_masks}×{self.freq_width}, time {self.time_masks}×{self.time_width:g}"

    def forward(self, f: torch.Tensor, fl: torch.Tensor) -> torch.Tensor:
        if not self.training or (self.freq_masks <= 0 and self.time_masks <= 0): return f
        B, F, T = f.shape; dev = f.device; keep = torch.ones(B, F, T, dtype=torch.bool, device=dev)
        if self.freq_masks > 0 and self.freq_width > 0:
            w = torch.randint(0, self.freq_width + 1, (B, self.freq_masks), device=dev)
            s = (torch.rand(B, self.freq_masks, device=dev) * (F - w + 1).clamp(min=1)).long()
            ix = torch.arange(F, device=dev)[None, None]; hit = ((ix >= s[..., None]) & (ix < (s + w)[..., None])).any(1)       # (B, F)
            keep &= ~hit[:, :, None]
        if self.time_masks > 0:
            L = fl.to(dev).long().clamp(min=0, max=T)
            wmax = (L.float() * self.time_width).floor().long() if self.time_width < 1 else torch.full_like(L, int(self.time_width))
            w = (torch.rand(B, self.time_masks, device=dev) * (wmax[:, None] + 1)).long()                                      # [0, wmax]
            s = (torch.rand(B, self.time_masks, device=dev) * (L[:, None] - w + 1).clamp(min=1)).long()
            ix = torch.arange(T, device=dev)[None, None]; hit = ((ix >= s[..., None]) & (ix < (s + w)[..., None])).any(1)       # (B, T)
            keep &= ~hit[:, None, :]
        return f.masked_fill(~keep, 0.0)


class NemotronOnline(nn.Module):
    def __init__(self, right_context: int = 0, path: Optional[str] = None):
        super().__init__()
        local = sorted(glob.glob(os.path.join(path or os.environ.get("MXC_NEMOTRON_DIR", os.environ.get("NEMOTRON_DIR", "")), "*.nemo")))
        direct = os.environ.get("VAPASR_ENCODER_CACHE", "")                                   # .nemo 없이 인코더 캐시(.pt)만 있는 곳(로컬 Mac 등)
        cached = _load_encoder_cache(local[0]) if local else (_load_encoder_cache_file(direct) if direct and os.path.exists(direct) else None)
        if not local and cached is None: raise FileNotFoundError(f".nemo 도 인코더 캐시(VAPASR_ENCODER_CACHE={direct!r})도 없음")
        if cached is not None: self.pre, self.enc = cached
        else:
            import nemo.collections.asr as nemo_asr
            m = nemo_asr.models.ASRModel.restore_from(local[0], map_location="cpu") if local else nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu")
            self.pre, self.enc = m.preprocessor, m.encoder.float()
            if local: _save_encoder_cache(local[0], m)
            del m
        self.enc.set_default_att_context_size([56, right_context]); self.trainable = False; self.set_trainable(False)
        self.spec_aug: Optional[SpecAugment] = None                                           # set_spec_augment 로 켠다(학습 중에만 적용)
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
    def set_spec_augment(self, spec: str = "off") -> Optional[SpecAugment]:
        """'off' | 'light' | 'nemo' | 'Fm,Fw,Tm,Tw'. 인코더 동결 여부와 무관하게 입력 특징에 걸린다(동결이어도 adapter·thinker 에는 증강)."""
        self.spec_aug = SpecAugment.from_spec(spec)
        if self.spec_aug is not None: self.spec_aug.train(self.training)
        return self.spec_aug
    def train(self, mode: bool = True):                    # 동결 중에는 dropout 등이 켜지지 않도록 eval 유지
        super().train(mode); self.enc.train(mode and self.trainable); return self
    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """wav (B, L) float32, wav_len (B,) 샘플 수, K (B,) 목표 프레임 수 → (B, max K, D) float32. 인코더는 fp32 로 돈다(autocast 밖)."""
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad(): f, fl = self.pre(input_signal=wav.float(), length=wav_len)
            if self.spec_aug is not None: f = self.spec_aug(f, fl)                              # 모듈이 training 일 때만(추론·평가는 그대로)
        # 동결 인코더는 fp32(C2 이후 특징과 완전 동일). 학습할 때는 bf16 autocast — fp32 는 활성화 메모리가 2 배라 KO 배치 48 이 H200 에서 OOM(2026-09-09 실측: bs24 87 GB)
        with torch.autocast("cuda", enabled=bool(self.trainable), dtype=torch.bfloat16):
            ctx = torch.no_grad() if not self.trainable else torch.enable_grad()
            with ctx: h, hl = self.enc(audio_signal=f, length=fl)              # (B, D, T')
        h = h.float()
        h = h.transpose(1, 2); Km = int(K.max()); out = h.new_zeros(h.shape[0], Km, h.shape[2])
        for i in range(h.shape[0]):
            k, t = int(K[i]), int(hl[i]); n = min(k, t); out[i, :n] = h[i, :n]
        return out
