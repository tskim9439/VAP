"""Nemotron 특징 → adapter → Qwen3-ASR thinker(LoRA). 프롬프트는 Qwen3-ASR 원본 형식을 그대로 재현한다.

  <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>[<|audio_pad|>×N]<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage {Lang}<asr_text>{text}<|im_end|>
adapter 출력을 <|audio_pad|> 위치의 inputs_embeds 에 scatter 한다 (thinker.get_audio_features 우회).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Optional

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
