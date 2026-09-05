"""Stage 1 mono 모델 — Nemotron [56,0] 특징 (1,K,1024) → adapter → Qwen3-ASR thinker(LoRA). `model.InterleavedASR` 의 mono 판.

두 채널 merge·화자별 오디오 토큰·<SPK_x> 방출이 없다. 그 외(특수 토큰 여유 행 + grad mask, <NEXT_AUDIO> 가중 CE, KV-cache 스트리밍 디코드)는 동일.
전부 random init (plans/stage1-mono-pilot.md §5): 기존 U0.5 파라미터를 읽지 않는다.
"""
import math
from typing import Dict, List
import torch, torch.nn as nn, torch.nn.functional as F
from .model import Adapter

class MonoInterleavedASR(nn.Module):
    def __init__(self, thinker, tokenizer, adapter: Adapter, sp_ids: Dict[str, int], lora_r: int = 16, lora_alpha: int = 32,
                 lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
        super().__init__()
        self.tok = tokenizer; self.adapter = adapter; self.sp_ids = dict(sp_ids)
        for p in thinker.parameters(): p.requires_grad_(False)
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            thinker = get_peft_model(thinker, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=list(lora_targets), bias="none"))
        self.thinker = thinker
        c = thinker.config if not hasattr(thinker, "base_model") else thinker.base_model.model.config
        self.audio_pad = c.audio_token_id
        emb = self._embed(); W = emb.weight; self.special_rows = sorted(self.sp_ids.values())
        assert max(self.special_rows) < W.shape[0], "임베딩 행렬에 특수 토큰 여유 행 없음 → resize 필요"
        with torch.no_grad():
            mu = W[: len(tokenizer) - len(self.sp_ids)].float().mean(0)
            for r in self.special_rows: W[r] = (mu + 0.02 * torch.randn_like(mu)).to(W.dtype)
        W.requires_grad_(True); mask = torch.zeros(W.shape[0], 1, dtype=W.dtype, device=W.device); mask[self.special_rows] = 1
        W.register_hook(lambda g: g * mask.to(g.device, g.dtype))                       # 특수 토큰 행만 갱신 (lm_head 는 tied)
        self.next_audio = self.sp_ids["<NEXT_AUDIO>"]
        blocked = [self.audio_pad, c.audio_start_token_id, c.audio_end_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|im_start|>"),
                   tokenizer.convert_tokens_to_ids("<asr_text>"), self.sp_ids["<EMPTY_AUDIO>"], self.sp_ids["<SPK_A>"], self.sp_ids["<SPK_B>"]] \
                  + [v for k, v in self.sp_ids.items() if k.startswith("<DELAY_")]
        self.register_buffer("blocked", torch.tensor(sorted(set(blocked))), persistent=False)

    def _embed(self):
        m = self.thinker.base_model.model if hasattr(self.thinker, "base_model") else self.thinker
        return m.get_input_embeddings()

    def chunk_embed(self, feats):                       # (B,1,K,Din) → (B,K,D)
        return self.adapter(feats[:, 0].float())

    def build(self, feats, ids, is_audio, chunk_of):
        emb = self._embed(); E = emb(ids); ce = self.chunk_embed(feats).to(E.dtype); D = E.shape[-1]
        g = torch.gather(ce, 1, chunk_of.clamp(min=0)[..., None].expand(-1, -1, D))
        return torch.where(is_audio[..., None], g, E)

    def forward(self, feats, ids, is_audio, chunk_of, labels, mask, next_weight: float = 1.0):
        """next_weight < 1: <NEXT_AUDIO>(= RNN-T blank) 위치 CE 를 낮춰 라벨 불균형(70–85 %)을 완화."""
        out = self.thinker(inputs_embeds=self.build(feats, ids, is_audio, chunk_of), attention_mask=mask)
        logits = out.logits[:, :-1].float(); tgt = labels[:, 1:]
        tok_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="none").view_as(tgt)
        na = tgt == self.next_audio; tx = (tgt != -100) & ~na
        w = torch.where(na, tok_loss.new_tensor(next_weight), tok_loss.new_tensor(1.0)) * (tgt != -100)
        loss = (tok_loss * w).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            top1 = (logits.argmax(-1) == tgt) & tx
            parts = dict(loss_next=tok_loss[na].mean().item() if na.any() else 0.0, loss_text=tok_loss[tx].mean().item() if tx.any() else 0.0,
                         top1_text=(top1.sum() / tx.sum().clamp(min=1)).item())
        return loss, parts

    @torch.inference_mode()
    def stream_decode(self, feats, prefix_ids: List[int], max_per_chunk: int = 4, next_bias: float = 0.0):
        """feats (1,K,Din) → [(chunk k, token id)], forced_next 횟수. chunk k 의 방출 시각 = (k+1)·80 ms. KV cache 로 위치당 forward 1 회."""
        from transformers import DynamicCache
        emb = self._embed(); dev = feats.device; cache = DynamicCache(); ce = self.chunk_embed(feats[None])[0].to(emb.weight.dtype)   # (K,D)
        def step(e):
            out = self.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True); return out.logits[0, -1].float()
        step(emb(torch.tensor(prefix_ids, device=dev))); e_next = emb.weight[self.next_audio]; out, forced = [], 0
        for k in range(ce.shape[0]):
            logits = step(ce[k]); n = 0
            while True:
                logits[self.blocked] = float("-inf"); logits[self.next_audio] -= next_bias
                tid = int(logits.argmax())
                if tid == self.next_audio or n >= max_per_chunk:
                    forced += int(tid != self.next_audio); step(e_next); break
                out.append((k, tid)); n += 1; logits = step(emb.weight[tid])
        return out, forced

    def trainable_state(self):
        W = self._embed().weight.detach().cpu()
        return dict(adapter=self.adapter.state_dict(), lora={k: v for k, v in self.thinker.state_dict().items() if "lora" in k},
                    special_rows={int(r): W[r].clone() for r in self.special_rows}, sp_ids=self.sp_ids, mono=True)

    def load_trainable_state(self, st):
        self.adapter.load_state_dict(st["adapter"])
        if st.get("lora"): self.thinker.load_state_dict(st["lora"], strict=False)
        with torch.no_grad():
            W = self._embed().weight
            for r, v in st.get("special_rows", {}).items(): W[int(r)] = v.to(device=W.device, dtype=W.dtype)
