"""final(오프라인) 모드 디코드 — 같은 VapAsr 모델이 인코더 특징 전체를 Qwen3-ASR 원 형식(vapasr/data/offline_seq)으로 보고 전사를 greedy 로 낸다.

스트리밍 중 계산한 인과 인코더 특징(청크 k 의 프레임)을 그대로 넣을 수 있어, <SEM_END> 로 끊은 구간을 같은 모델로 다시 디코드하는 2-pass('self')에도 쓴다.
막는 토큰: 스트리밍 특수 토큰(<NEXT_AUDIO>·<EMPTY_AUDIO>·<DELAY_d>·화자 태그 등 config.sp_ids), 이벤트(<SEM_END>·턴 종료), 오디오·대화 형식 토큰.
<|im_end|> 가 나오면 멈춘다(최대 max_new_tokens, 기본 K//2 + 32 — 12.5 Hz 청크당 말 토큰은 0.3 개 안팎).
배치 없이 한 발화씩(B=1, KV 캐시) — mRoPE 위치를 패딩과 섞지 않으려는 단순화."""
from typing import Dict, List, Optional
import torch

from ..data.offline_seq import OfflineFormat


def offline_blocked_ids(model, fmt: OfflineFormat) -> List[int]:
    c = model.config; ids = set(int(v) for v in (c.sp_ids or {}).values())
    ids |= set(int(v) for v in (getattr(c, "sem_registry", None) or {}).values())
    ids |= {fmt.audio_start, fmt.audio_end, fmt.audio_pad}
    tok = fmt.tok
    for t in ("<|im_start|>", "<asr_text>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0: ids.add(i)
    ids.discard(fmt.im_end)
    return sorted(ids)


@torch.no_grad()
def offline_decode(model, tok, feats: torch.Tensor, lang: str, max_new_tokens: Optional[int] = None, fmt: Optional[OfflineFormat] = None,
                   blocked: Optional[torch.Tensor] = None) -> List[int]:
    """feats (K, Din) 인코더 특징 → 텍스트 토큰 id 목록(<|im_end|> 제외)."""
    from transformers import DynamicCache
    fmt = fmt or OfflineFormat(tok); dev = feats.device; K = int(feats.shape[0])
    if blocked is None: blocked = torch.tensor(offline_blocked_ids(model, fmt), device=dev)
    emb = model.get_input_embeddings(); ids = fmt.header(K, lang); a0 = len(fmt.pre) + 1
    e = emb(torch.tensor(ids, device=dev)).clone()
    e[a0:a0 + K] = model.chunk_embed(feats[None, None])[0].to(e.dtype)
    cache = DynamicCache()
    def step(x):
        return model.thinker(inputs_embeds=x.view(1, -1, x.shape[-1]), past_key_values=cache, use_cache=True).logits[0, -1].float()
    logits = step(e); out = []; limit = (K // 2 + 32) if max_new_tokens is None else int(max_new_tokens)
    for _ in range(limit):
        logits[blocked] = float("-inf"); t = int(logits.argmax())
        if t == fmt.im_end: break
        out.append(t); logits = step(emb.weight[t])
    return out


@torch.no_grad()
def offline_transcribe(model, tok, wav: torch.Tensor, lang: str, **kw) -> List[int]:
    """wav (T,) 16 kHz → 인코더(학습과 같은 청크 수 K = round(길이·12.5)) → offline_decode."""
    K = torch.tensor([int(round(wav.shape[0] / 16000 * model.config.frame_hz))], device=wav.device)
    feats = model.encode(wav[None], torch.tensor([wav.shape[0]], device=wav.device), K)[0, 0]
    return offline_decode(model, tok, feats, lang, **kw)
