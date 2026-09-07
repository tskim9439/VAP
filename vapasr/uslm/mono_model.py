"""Stage 1 mono 모델 — Nemotron [56,0] 특징 (1,K,1024) → adapter → Qwen3-ASR thinker(LoRA). `model.InterleavedASR` 의 mono 판.

두 채널 merge·화자별 오디오 토큰·<SPK_x> 방출이 없다. 그 외(특수 토큰 여유 행 + grad mask, <NEXT_AUDIO> 가중 CE, KV-cache 스트리밍 디코드)는 동일.
전부 random init (plans/stage1-mono-pilot.md §5): 기존 U0.5 파라미터를 읽지 않는다.
"""
import math
from typing import Dict, List, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from .model import Adapter

class MonoInterleavedASR(nn.Module):
    def __init__(self, thinker, tokenizer, adapter: Adapter, sp_ids: Dict[str, int], lora_r: int = 16, lora_alpha: int = 32,
                 lora_targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), full_ft: bool = False, encoder: Optional[nn.Module] = None):
        """full_ft=True: thinker 0.6B 전체를 학습(LoRA 없음, fp32 master + autocast bf16). 임베딩은 전 행 학습(grad mask 없음).
        encoder: 온라인 특징 인코더(NemotronOnline). 주면 forward/stream_decode 가 wav 를 받아 특징을 직접 만든다(동결/학습은 encoder.set_trainable)."""
        super().__init__()
        self.tok = tokenizer; self.adapter = adapter; self.sp_ids = dict(sp_ids); self.full_ft = full_ft; self.encoder = encoder
        if full_ft:
            thinker = thinker.float()
            for p in thinker.parameters(): p.requires_grad_(True)
        else:
            for p in thinker.parameters(): p.requires_grad_(False)
            if lora_r > 0:
                from peft import LoraConfig, get_peft_model
                thinker = get_peft_model(thinker, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules=list(lora_targets), bias="none"))
        self.thinker = thinker
        c = thinker.base_model.model.config if hasattr(thinker, "peft_config") else thinker.config   # HF 모델도 base_model 속성이 있어 PEFT 판별은 peft_config 로
        self.audio_pad = c.audio_token_id
        emb = self._embed(); W = emb.weight; self.special_rows = sorted(self.sp_ids.values())
        assert max(self.special_rows) < W.shape[0], "임베딩 행렬에 특수 토큰 여유 행 없음 → resize 필요"
        with torch.no_grad():
            mu = W[: len(tokenizer) - len(self.sp_ids)].float().mean(0)
            for r in self.special_rows: W[r] = (mu + 0.02 * torch.randn_like(mu)).to(W.dtype)
        W.requires_grad_(True)
        if not full_ft:                                                                  # LoRA 모드: 특수 토큰 행만 갱신 (lm_head 는 tied). full FT 는 전 행 학습
            mask = torch.zeros(W.shape[0], 1, dtype=W.dtype, device=W.device); mask[self.special_rows] = 1
            W.register_hook(lambda g: g * mask.to(g.device, g.dtype))
        self.next_audio = self.sp_ids["<NEXT_AUDIO>"]
        blocked = [self.audio_pad, c.audio_start_token_id, c.audio_end_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|im_start|>"),
                   tokenizer.convert_tokens_to_ids("<asr_text>"), self.sp_ids["<EMPTY_AUDIO>"], self.sp_ids["<SPK_A>"], self.sp_ids["<SPK_B>"]] \
                  + [v for k, v in self.sp_ids.items() if k.startswith("<DELAY_")]
        self.register_buffer("blocked", torch.tensor(sorted(set(blocked))), persistent=False)

    def _lm(self):                                      # PEFT 래퍼를 벗긴 Qwen3ASRThinkerForConditionalGeneration (.model 디코더, .lm_head)
        return self.thinker.base_model.model if hasattr(self.thinker, "peft_config") else self.thinker
    def _embed(self): return self._lm().get_input_embeddings()

    def chunk_embed(self, feats):                       # (B,1,K,Din) → (B,K,D)
        return self.adapter(feats[:, 0].float())
    def encode(self, wav, wav_len, K):                  # (B,L) wav → (B,1,Kmax,Din) — 온라인 인코더
        assert self.encoder is not None, "encoder 없음(캐시 모드)"; return self.encoder(wav, wav_len, K)[:, None]

    def build(self, feats, ids, is_audio, chunk_of):
        emb = self._embed(); E = emb(ids); ce = self.chunk_embed(feats).to(E.dtype); D = E.shape[-1]
        g = torch.gather(ce, 1, chunk_of.clamp(min=0)[..., None].expand(-1, -1, D))
        return torch.where(is_audio[..., None], g, E)

    def forward(self, feats, ids, is_audio, chunk_of, labels, mask, next_weight: float = 1.0, wav_len=None, K=None):
        """next_weight < 1: <NEXT_AUDIO>(= RNN-T blank) 위치 CE 를 낮춰 라벨 불균형(70–85 %)을 완화. wav_len 이 있으면 feats 는 wav (B,L) 이고 encoder 로 특징을 만든다."""
        if wav_len is not None: feats = self.encode(feats, wav_len, K)
        # lm_head 는 라벨 위치에서만 계산한다. 전 위치 (B,L,152k) fp32 logits 는 KO bs 48 에서 >100 GB 라 140 GB GPU 에서도 OOM (job 65258).
        lm = self._lm(); h = lm.model(inputs_embeds=self.build(feats, ids, is_audio, chunk_of), attention_mask=mask).last_hidden_state
        tgt = labels[:, 1:]; sel = tgt != -100; t = tgt[sel]                                       # (N,)
        logits = lm.lm_head(h[:, :-1][sel]).float()                                                 # (N,V)
        tok_loss = F.cross_entropy(logits, t, reduction="none")
        na = t == self.next_audio; tx = ~na
        w = torch.where(na, tok_loss.new_tensor(next_weight), tok_loss.new_tensor(1.0))
        loss = (tok_loss * w).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            top1 = (logits.argmax(-1) == t) & tx
            parts = dict(loss_next=tok_loss[na].mean().item() if na.any() else 0.0, loss_text=tok_loss[tx].mean().item() if tx.any() else 0.0,
                         top1_text=(top1.sum() / tx.sum().clamp(min=1)).item())
        return loss, parts

    @torch.inference_mode()
    def stream_decode(self, feats, prefix_ids: List[int], max_per_chunk: int = 0, next_bias: float = 0.0, max_flush_rounds: int = 8, runaway_cap: int = 64):
        """feats (1,K,Din) → ([(chunk k, token id)], forced_next 횟수, tick_ms 목록, flush 라운드 수).
        chunk k 의 방출 시각 = (k+1)·80 ms. 스트림 끝에서는 <EMPTY_AUDIO> 를 입력해 flush 라운드를 돌린다(학습 규약과 동일):
        라운드마다 <NEXT_AUDIO> 까지(M 제한 없음, runaway_cap 은 폭주 방지), 토큰 없이 <NEXT_AUDIO> 가 나오면 종료. flush 토큰의 chunk 번호는 K + 라운드. KV cache 로 위치당 forward 1 회."""
        import time
        from transformers import DynamicCache
        if feats.dim() == 1:                            # wav (T,) → 온라인 인코더([56,0] 인과라 전체 인코딩 = 스트리밍 출력)
            K = torch.tensor([int(round(feats.shape[0] / 16000 * 12.5))], device=feats.device)
            feats = self.encode(feats[None], torch.tensor([feats.shape[0]], device=feats.device), K)[0]
        emb = self._embed(); dev = feats.device; cache = DynamicCache(); ce = self.chunk_embed(feats[None])[0].to(emb.weight.dtype)   # (K,D)
        def step(e):
            out = self.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True); return out.logits[0, -1].float()
        step(emb(torch.tensor(prefix_ids, device=dev))); e_next = emb.weight[self.next_audio]; e_empty = emb.weight[self.sp_ids["<EMPTY_AUDIO>"]]
        out, forced, ticks = [], 0, []
        def emit_round(k, logits):
            nonlocal forced; n = 0
            while True:
                logits[self.blocked] = float("-inf"); logits[self.next_audio] -= next_bias
                tid = int(logits.argmax())
                if tid == self.next_audio or n >= (max_per_chunk or runaway_cap):          # max_per_chunk 0 = 제한 없음(폭주 방지 cap 만)
                    forced += int(tid != self.next_audio); step(e_next); return n
                out.append((k, tid)); n += 1; logits = step(emb.weight[tid])
        K = ce.shape[0]
        for k in range(K):
            t = time.time(); emit_round(k, step(ce[k])); torch.cuda.synchronize(); ticks.append((time.time() - t) * 1000)
        rounds = 0
        for r in range(max_flush_rounds):
            rounds += 1
            if emit_round(K + r, step(e_empty)) == 0: break
        return out, forced, ticks, rounds

    def trainable_state(self):
        W = self._embed().weight.detach().cpu()
        enc = {"encoder": {k: v.detach().cpu() for k, v in self.encoder.enc.state_dict().items()}} if (self.encoder is not None and getattr(self.encoder, "trainable", False)) else {}
        if self.full_ft:   # thinker 전체(bf16 로 저장, 0.6B ≈ 1.2 GB)
            return dict(adapter=self.adapter.state_dict(), thinker={k: v.detach().to(torch.bfloat16).cpu() for k, v in self.thinker.state_dict().items()}, sp_ids=self.sp_ids, mono=True, full_ft=True, **enc)
        return dict(adapter=self.adapter.state_dict(), lora={k: v for k, v in self.thinker.state_dict().items() if "lora" in k},
                    special_rows={int(r): W[r].clone() for r in self.special_rows}, sp_ids=self.sp_ids, mono=True)

    def load_trainable_state(self, st):
        self.adapter.load_state_dict(st["adapter"])
        if st.get("encoder") and self.encoder is not None: self.encoder.enc.load_state_dict(st["encoder"])
        if st.get("thinker"): self.thinker.load_state_dict({k: v.to(next(self.thinker.parameters()).dtype) for k, v in st["thinker"].items()}, strict=False); return
        if st.get("lora"): self.thinker.load_state_dict(st["lora"], strict=False)
        with torch.no_grad():
            W = self._embed().weight
            for r, v in st.get("special_rows", {}).items(): W[int(r)] = v.to(device=W.device, dtype=W.dtype)
