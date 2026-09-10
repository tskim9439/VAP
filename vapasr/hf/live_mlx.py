"""MLX thinker 백엔드(Apple silicon): mlx-lm 의 Qwen3 디코더에 오디오 청크 임베딩·토큰 임베딩을 `input_embeddings` 로 넣고 KV cache 를 이어 간다.
LiveSession(backend="mlx") 가 쓴다. 인코더·adapter 는 PyTorch(MPS) 그대로."""
import time, numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

class MlxThinker:
    def __init__(self, path: str):
        self.model, self.tok = mlx_load(path); self.embed = self.model.model.embed_tokens          # nn.Embedding 또는 QuantizedEmbedding(q8) — 모듈 호출로 조회
        self.vocab = int(self.model.args.vocab_size); self.dtype = self.embed(mx.array([0])).dtype
        self.cache = None; self.reset()
    @property
    def emb_w(self):                                    # (vocab, D) 크기 정보용(호환)
        class _S: shape = (self.vocab,)
        return _S()
    def reset(self): self.cache = make_prompt_cache(self.model)
    def embed_ids(self, ids) -> mx.array: return self.embed(mx.array(list(ids)))                  # (T, D)
    def embed_id(self, tid: int) -> mx.array: return self.embed(mx.array([tid]))[0]
    def step(self, e: mx.array) -> mx.array:
        """e (T, D) 또는 (D,) → 마지막 위치 logits (V,) float32"""
        if e.ndim == 1: e = e[None]
        logits = self.model(mx.zeros((1, e.shape[0]), dtype=mx.int32), cache=self.cache, input_embeddings=e[None].astype(self.dtype))
        return logits[0, -1].astype(mx.float32)
