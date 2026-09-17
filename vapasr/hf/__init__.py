"""HF(transformers) 친화 패키지 — VAP-ASR 스트리밍 모델을 PreTrainedModel/PretrainedConfig 로 감싸고, Trainer 서브클래스로 학습한다 (2026-09-08).

  configuration_vapasr : VapAsrConfig (thinker 설정·adapter·인코더·특수 토큰 id·chunk 규약)
  modeling_vapasr      : VapAsrForStreamingASR — encoder(Nemotron, 동결/해동) → adapter → Qwen3-ASR thinker. forward(loss) · stream_decode · from_qwen · from_legacy
  liger                : thinker 에 Liger 커널(RMSNorm·SwiGLU·RoPE) 적용
  data                 : 다중 manifest 라운드로빈 로더(언어별 버킷 배치, rank 분할)
  trainer              : VapAsrTrainer(Trainer) — compute_loss·옵티마이저 그룹·분산 스트리밍 평가·선점 콜백
"""
from .configuration_vapasr import VapAsrConfig
from .modeling_vapasr import VapAsrForStreamingASR, VapAsrOutput

# AutoConfig/AutoModel 에 등록 → from_pretrained(dir) 를 Auto 클래스로도 쓸 수 있다
from transformers import AutoConfig, AutoModel
AutoConfig.register("vapasr", VapAsrConfig); AutoModel.register(VapAsrConfig, VapAsrForStreamingASR)

def load_tokenizer(path: str):
    """HF 산출물 디렉토리의 tokenizer. checkpoint-N 에 tokenizer 파일이 없으면 config 의 thinker_name_or_path(Qwen 디렉토리)에서 읽고 특수 토큰을 다시 붙인다."""
    import os, json
    from transformers import AutoTokenizer
    from ..uslm.interleave_data import add_specials
    cfg = json.load(open(os.path.join(path, "config.json")))
    src = path if os.path.exists(os.path.join(path, "tokenizer_config.json")) else cfg.get("thinker_name_or_path", "Qwen/Qwen3-ASR-0.6B")
    tok = AutoTokenizer.from_pretrained(src); sp = add_specials(tok)
    if cfg.get("phase2_registry"):                                                   # Phase 2 산출물: lane 3–6·ONSET·EOT 도 붙이고 registry 와 대조
        from ..data.dialogue_tokens import add_phase2_specials
        sp = add_phase2_specials(tok)
        want = {**cfg.get("sp_ids", {}), **cfg["phase2_registry"]}
    else: want = cfg.get("sp_ids", sp)
    bad = {k: (sp.get(k), v) for k, v in want.items() if sp.get(k) != v}
    assert not bad, f"tokenizer 특수 토큰 id 가 config 와 다름: {bad}"
    return tok
