"""VapAsrConfig — VAP-ASR 스트리밍 모델의 HF 설정.

thinker 는 Qwen3-ASR 의 thinker 설정(dict) 을 그대로 품는다(오디오 타워는 쓰지 않는다). 인코더는 Nemotron [56,0] (.nemo 경로는 환경에 따라 다르므로
config 에는 '출처 이름' 만 두고 실제 경로는 env MXC_NEMOTRON_DIR / NEMOTRON_DIR 또는 from_pretrained(encoder_path=…) 로 준다).
특수 토큰 id(sp_ids) 와 chunk 규약(80 ms, 12.5 Hz) 은 학습·디코드 양쪽의 정본이다."""
from typing import Dict, List, Optional
from transformers import PretrainedConfig

class VapAsrConfig(PretrainedConfig):
    model_type = "vapasr"

    def __init__(self, thinker: Optional[dict] = None, thinker_name_or_path: str = "Qwen/Qwen3-ASR-0.6B",
                 adapter_d_in: int = 1024, adapter_d_out: int = 1024, adapter_hidden: int = 2048,
                 encoder_type: str = "nemotron", encoder_name: str = "nvidia/nemotron-3.5-asr-streaming-0.6b", encoder_left_context: int = 56, encoder_right_context: int = 0,
                 encoder_trainable: bool = False, sp_ids: Optional[Dict[str, int]] = None, special_tokens: Optional[List[str]] = None,
                 chunk_s: float = 0.08, frame_hz: float = 12.5, delays: List[int] = (2, 3, 4, 6), next_weight: float = 0.3, next_weight_ko: float = 0.15,
                 full_ft: bool = True, lora_r: int = 0, runaway_cap: int = 64, max_flush_rounds: int = 8, blocked_ids: Optional[List[int]] = None, audio_pad_id: Optional[int] = None,
                 prefix_ids: Optional[List[int]] = None, **kw):
        self.thinker = thinker or {}; self.thinker_name_or_path = thinker_name_or_path
        self.adapter_d_in, self.adapter_d_out, self.adapter_hidden = adapter_d_in, adapter_d_out, adapter_hidden
        self.encoder_type, self.encoder_name = encoder_type, encoder_name; self.encoder_left_context, self.encoder_right_context = encoder_left_context, encoder_right_context
        self.encoder_trainable = encoder_trainable
        self.sp_ids = dict(sp_ids or {}); self.special_tokens = list(special_tokens or [])
        self.chunk_s, self.frame_hz, self.delays = chunk_s, frame_hz, list(delays); self.next_weight, self.next_weight_ko = next_weight, next_weight_ko
        self.full_ft, self.lora_r, self.runaway_cap, self.max_flush_rounds = full_ft, lora_r, runaway_cap, max_flush_rounds
        self.blocked_ids = list(blocked_ids or []); self.audio_pad_id = audio_pad_id; self.prefix_ids = list(prefix_ids or [])   # 디코드에 tokenizer 없이도 쓰도록 id 를 보관
        super().__init__(**kw)

    # thinker 의 텍스트 설정에서 자주 쓰는 값
    @property
    def text_config(self) -> dict: return self.thinker.get("text_config", {})
    @property
    def hidden_size(self) -> int: return int(self.text_config.get("hidden_size", self.adapter_d_out))
    @property
    def vocab_size(self) -> int: return int(self.text_config.get("vocab_size", 151936))
