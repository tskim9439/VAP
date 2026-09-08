"""HF(transformers) 친화 패키지 — VAP-ASR 스트리밍 모델을 PreTrainedModel/PretrainedConfig 로 감싸고, Trainer 서브클래스로 학습한다 (2026-09-08).

  configuration_vapasr : VapAsrConfig (thinker 설정·adapter·인코더·특수 토큰 id·chunk 규약)
  modeling_vapasr      : VapAsrForStreamingASR — encoder(Nemotron, 동결/해동) → adapter → Qwen3-ASR thinker. forward(loss) · stream_decode · from_qwen · from_legacy
  liger                : thinker 에 Liger 커널(RMSNorm·SwiGLU·RoPE) 적용
  data                 : 다중 manifest 라운드로빈 로더(언어별 버킷 배치, rank 분할)
  trainer              : VapAsrTrainer(Trainer) — compute_loss·옵티마이저 그룹·분산 스트리밍 평가·선점 콜백
"""
from .configuration_vapasr import VapAsrConfig
from .modeling_vapasr import VapAsrForStreamingASR, VapAsrOutput
