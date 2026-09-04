"""Nemotron 특징 → adapter → Qwen3-ASR thinker(LoRA). 프롬프트는 Qwen3-ASR 원본 형식을 그대로 재현한다.

  <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>[<|audio_pad|>×N]<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage {Lang}<asr_text>{text}<|im_end|>
adapter 출력을 <|audio_pad|> 위치의 inputs_embeds 에 scatter 한다 (thinker.get_audio_features 우회).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Optional, Dict

class Adapter(nn.Module):
    def __init__(self, d_in=1024, d_out=1024, h=2048):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))
    def forward(self, x): return self.net(x)

class AdapterThinkerASR(nn.Module):
    def __init__(self, thinker, tokenizer, adapter: Adapter, lora_r: int = 16, lora_alpha: int = 32, lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
        super().__init__()
        self.tok = tokenizer; self.adapter = adapter
        for p in thinker.parameters(): p.requires_grad_(False)
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            thinker = get_peft_model(thinker, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=list(lora_targets), bias="none"))
        self.thinker = thinker
        c = thinker.config if not hasattr(thinker, "base_model") else thinker.base_model.model.config
        self.audio_pad = c.audio_token_id; self.audio_start = c.audio_start_token_id; self.audio_end = c.audio_end_token_id
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._pre = tokenizer("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        self._mid = tokenizer("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]

    def _embed(self):   # 텍스트 임베딩 테이블
        m = self.thinker.base_model.model if hasattr(self.thinker, "base_model") else self.thinker
        return m.get_input_embeddings()

    def build(self, feats, lens, langs: List[str], texts: Optional[List[str]] = None):
        """→ inputs_embeds (B,L,D), attention_mask, labels (텍스트+im_end 만, 나머지 -100), prompt_len 리스트"""
        emb = self._embed(); dev = feats.device; B = feats.shape[0]; seqs, labs = [], []
        a_all = self.adapter(feats)                                       # (B, T, D)
        for i in range(B):
            n = int(lens[i]); force = self.tok(f"language {langs[i]}<asr_text>", add_special_tokens=False)["input_ids"]
            ids = self._pre + [self.audio_start] + [self.audio_pad] * n + [self.audio_end] + self._mid + force
            tgt = (self.tok(texts[i], add_special_tokens=False)["input_ids"] + [self.im_end]) if texts is not None else []
            all_ids = torch.tensor(ids + tgt, device=dev); e = emb(all_ids)
            pos = torch.nonzero(all_ids == self.audio_pad).flatten(); e = e.clone(); e[pos] = a_all[i, :n].to(e.dtype)
            lab = torch.full((len(all_ids),), -100, device=dev, dtype=torch.long)
            if tgt: lab[len(ids):] = torch.tensor(tgt, device=dev)
            seqs.append(e); labs.append(lab)
        L = max(s.shape[0] for s in seqs); D = seqs[0].shape[1]
        E = torch.zeros(B, L, D, device=dev, dtype=seqs[0].dtype); M = torch.zeros(B, L, device=dev, dtype=torch.long); Y = torch.full((B, L), -100, device=dev, dtype=torch.long)
        for i, (e, l) in enumerate(zip(seqs, labs)): E[i, : e.shape[0]] = e; M[i, : e.shape[0]] = 1; Y[i, : e.shape[0]] = l     # right-pad
        return E, M, Y

    def forward(self, feats, lens, langs, texts):
        E, M, Y = self.build(feats, lens, langs, texts)
        out = self.thinker(inputs_embeds=E, attention_mask=M)
        logits = out.logits[:, :-1].float(); tgt = Y[:, 1:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def transcribe(self, feats, lens, langs, max_new_tokens=128):
        E, M, _ = self.build(feats, lens, langs, None)
        # left-pad 로 바꿔 generate (decoder-only 배치 생성 규약)
        B, L, D = E.shape; E2 = torch.zeros_like(E); M2 = torch.zeros_like(M)
        for i in range(B):
            n = int(M[i].sum()); E2[i, L - n:] = E[i, :n]; M2[i, L - n:] = 1
        out = self.thinker.generate(inputs_embeds=E2, attention_mask=M2, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, eos_token_id=self.im_end, pad_token_id=self.im_end)
        seqs = out.sequences if hasattr(out, "sequences") else out
        return [self.tok.decode(s, skip_special_tokens=True).strip() for s in seqs]


# ----------------------------------------------------------------------------------------------------------------------
# U1 — interleaved streaming ASR (두 화자 joint chunk token + <NEXT_AUDIO> 가변 방출)
# ----------------------------------------------------------------------------------------------------------------------
class InterleavedASR(nn.Module):
    """시퀀스: prefix, [AUDIO_k] (<SPK_x>) tok … <NEXT_AUDIO>, … — `vapasr.uslm.interleave_data` 규약.
    chunk 임베딩 = merge([adapter(f_A); adapter(f_B)]) (merge 는 [½I ½I] 로 초기화 → U0.5 adapter 그대로 재사용, 화자 구분은 학습).
    특수 토큰(<NEXT_AUDIO> 등)은 thinker 임베딩 행렬의 여유 행을 쓰고, 그 행만 학습(grad mask; lm_head 는 tied)."""
    def __init__(self, thinker, tokenizer, adapter: Adapter, sp_ids: Dict[str, int], lora_r: int = 16, lora_alpha: int = 32,
                 lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
        super().__init__()
        self.tok = tokenizer; self.adapter = adapter; self.sp_ids = dict(sp_ids); D = adapter.net[-1].out_features
        self.merge = nn.Linear(2 * D, D)
        with torch.no_grad():
            self.merge.weight.zero_(); self.merge.weight[:, :D] += 0.5 * torch.eye(D); self.merge.weight[:, D:] += 0.5 * torch.eye(D)
            self.merge.weight[:, D:] += 0.01 * torch.randn(D, D) / math.sqrt(D); self.merge.bias.zero_()   # 화자 대칭 깨기
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
        W.register_hook(lambda g: g * mask.to(g.device, g.dtype))                       # 특수 토큰 행만 갱신
        self.next_audio = self.sp_ids["<NEXT_AUDIO>"]; self.spk_ids = (self.sp_ids["<SPK_A>"], self.sp_ids["<SPK_B>"])
        blocked = [self.audio_pad, c.audio_start_token_id, c.audio_end_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|im_start|>"),
                   tokenizer.convert_tokens_to_ids("<asr_text>"), self.sp_ids["<EMPTY_AUDIO>"]] + [v for k, v in self.sp_ids.items() if k.startswith("<DELAY_")]
        self.register_buffer("blocked", torch.tensor(sorted(set(blocked))), persistent=False)

    def _embed(self):
        m = self.thinker.base_model.model if hasattr(self.thinker, "base_model") else self.thinker
        return m.get_input_embeddings()

    def chunk_embed(self, feats):                       # (B,2,K,Din) → (B,K,D)
        a = self.adapter(feats.float()); return self.merge(torch.cat([a[:, 0], a[:, 1]], -1))

    def build(self, feats, ids, is_audio, chunk_of):
        emb = self._embed(); E = emb(ids); ce = self.chunk_embed(feats).to(E.dtype); D = E.shape[-1]
        g = torch.gather(ce, 1, chunk_of.clamp(min=0)[..., None].expand(-1, -1, D))
        return torch.where(is_audio[..., None], g, E)

    def forward(self, feats, ids, is_audio, chunk_of, labels, mask):
        E = self.build(feats, ids, is_audio, chunk_of)
        out = self.thinker(inputs_embeds=E, attention_mask=mask)
        logits = out.logits[:, :-1].float(); tgt = labels[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        with torch.no_grad():                            # 진단: NEXT_AUDIO 위치 / 텍스트 위치 손실 분리
            tok_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100, reduction="none").view_as(tgt)
            na = tgt == self.next_audio; tx = (tgt != -100) & ~na
            parts = dict(loss_next=tok_loss[na].mean().item() if na.any() else 0.0, loss_text=tok_loss[tx].mean().item() if tx.any() else 0.0)
        return loss, parts

    @torch.inference_mode()
    def stream_decode(self, feats, prefix_ids: List[int], max_per_chunk: int = 4):
        """feats (2,K,Din) 를 chunk 별로 넣고 greedy 로 방출. → [(chunk k, token id, speaker)], forced_next 횟수.
        chunk k 의 방출 시각 = (k+1)·80 ms (chunk 오디오를 다 본 뒤). KV cache 로 위치당 forward 1 회."""
        from transformers import DynamicCache
        emb = self._embed(); dev = feats.device; cache = DynamicCache()
        ce = self.chunk_embed(feats[None])[0].to(emb.weight.dtype)                      # (K,D)
        def step(e):
            out = self.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True); return out.logits[0, -1].float()
        step(emb(torch.tensor(prefix_ids, device=dev)))
        e_next = emb.weight[self.next_audio]; out, spk, forced = [], 0, 0
        for k in range(ce.shape[0]):
            logits = step(ce[k]); n = 0
            while True:
                logits[self.blocked] = float("-inf"); tid = int(logits.argmax())
                if tid == self.next_audio or n >= max_per_chunk:
                    forced += int(tid != self.next_audio); step(e_next); break
                if tid in self.spk_ids: spk = self.spk_ids.index(tid)
                else: n += 1
                out.append((k, tid, spk)); logits = step(emb.weight[tid])
        return out, forced

    def trainable_state(self):
        W = self._embed().weight.detach().cpu()
        return dict(adapter=self.adapter.state_dict(), merge=self.merge.state_dict(), lora={k: v for k, v in self.thinker.state_dict().items() if "lora" in k},
                    special_rows={int(r): W[r].clone() for r in self.special_rows}, sp_ids=self.sp_ids)

    def load_trainable_state(self, st, strict_lora: bool = True):
        self.adapter.load_state_dict(st["adapter"])
        if "merge" in st: self.merge.load_state_dict(st["merge"])
        if st.get("lora"):
            missing, unexpected = self.thinker.load_state_dict(st["lora"], strict=False)
            if strict_lora: assert not unexpected, unexpected[:3]
        if st.get("special_rows"):
            with torch.no_grad():
                W = self._embed().weight
                for r, v in st["special_rows"].items(): W[int(r)] = v.to(device=W.device, dtype=W.dtype)
