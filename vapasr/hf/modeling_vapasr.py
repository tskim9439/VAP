"""VapAsrForStreamingASR — HF PreTrainedModel 판 mono 스트리밍 ASR (vapasr/uslm/mono_model.MonoInterleavedASR 와 수치 동일).

구성: encoder(Nemotron [56,0], 온라인 특징; 기본 동결) → adapter(LN·MLP) → thinker(Qwen3-ASR thinker, 오디오 타워 제거, full FT).
시퀀스·손실 규약은 mono_model 과 같다: 라벨 위치에서만 lm_head 를 계산하고 <NEXT_AUDIO> 는 next_weight 로 가중.
저장: adapter + thinker 만 safetensors 에 쓴다(인코더는 동결이면 .nemo 에서 다시 만들고, 학습했으면 함께 저장). lm_head 는 임베딩과 tied.
생성: from_qwen(Qwen3-ASR 디렉토리) 로 새 모델, from_legacy(ckpt-last.pt) 로 기존 산출물 변환, from_pretrained(dir) 로 재로드."""
import os, time, math
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from .configuration_vapasr import VapAsrConfig

class Adapter(nn.Module):                                   # uslm/model.Adapter 와 동일 구조(키 이름 net.* 유지 → 기존 ckpt 호환)
    def __init__(self, d_in=1024, d_out=1024, h=2048):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))
    def forward(self, x): return self.net(x)

@dataclass
class VapAsrOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    loss_next: Optional[float] = None       # <NEXT_AUDIO> 위치 평균 CE (가중 전)
    loss_text: Optional[float] = None       # 텍스트 위치 평균 CE
    top1_text: Optional[float] = None       # 텍스트 위치 top-1 정확도
    n_labels: Optional[int] = None

class VapAsrForStreamingASR(PreTrainedModel):
    config_class = VapAsrConfig
    base_model_prefix = "vapasr"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["thinker.lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^thinker\.audio_tower\."]
    _keys_to_ignore_on_load_missing = [r"^encoder\."]

    def __init__(self, config: VapAsrConfig, thinker: Optional[nn.Module] = None):
        super().__init__(config)
        self.adapter = Adapter(config.adapter_d_in, config.adapter_d_out, config.adapter_hidden)
        if thinker is None:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRThinkerConfig
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
            thinker = Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRThinkerConfig.from_dict(dict(config.thinker)))
        if hasattr(thinker, "audio_tower"): del thinker.audio_tower               # 오디오 타워는 쓰지 않는다(Nemotron 이 대신)
        self.thinker = thinker
        self.encoder: Optional[nn.Module] = None                                  # attach_encoder 로 붙인다(무거운 NeMo 복원은 __init__ 밖에서)
        self.sp_ids = dict(config.sp_ids); self.next_audio = self.sp_ids.get("<NEXT_AUDIO>"); self.empty_audio = self.sp_ids.get("<EMPTY_AUDIO>")
        self.register_buffer("blocked", torch.tensor(sorted(set(config.blocked_ids)), dtype=torch.long), persistent=False)
        self.post_init()

    # ── HF 규약(tie·임베딩)
    def get_input_embeddings(self): return self.thinker.get_input_embeddings()
    def set_input_embeddings(self, v): self.thinker.set_input_embeddings(v)
    def get_output_embeddings(self): return self.thinker.lm_head
    def _init_weights(self, module):                                              # thinker 는 자체 초기화, adapter 는 기본(nn) 초기화 → 아무것도 덮지 않는다
        return

    # ── 인코더
    def attach_encoder(self, path: Optional[str] = None, trainable: Optional[bool] = None, state_dict: Optional[dict] = None):
        from ..features.online import NemotronOnline
        enc = NemotronOnline(right_context=self.config.encoder_right_context, path=path)
        if state_dict: enc.enc.load_state_dict(state_dict)
        tr = self.config.encoder_trainable if trainable is None else trainable; enc.set_trainable(tr); self.config.encoder_trainable = tr
        self.encoder = enc.to(next(self.adapter.parameters()).device); return self
    def encode(self, wav, wav_len, K):                                            # (B,L) → (B,1,Kmax,Din)
        assert self.encoder is not None, "encoder 없음 — attach_encoder() 먼저"; return self.encoder(wav, wav_len, K)[:, None]
    def chunk_embed(self, feats): return self.adapter(feats[:, 0].float())        # (B,1,K,Din) → (B,K,D)

    # ── 학습 forward
    def build(self, feats, ids, is_audio, chunk_of):
        emb = self.get_input_embeddings(); E = emb(ids); ce = self.chunk_embed(feats).to(E.dtype); D = E.shape[-1]
        g = torch.gather(ce, 1, chunk_of.clamp(min=0)[..., None].expand(-1, -1, D))
        return torch.where(is_audio[..., None], g, E)

    def forward(self, ids=None, is_audio=None, chunk_of=None, labels=None, mask=None, wav=None, wav_len=None, K=None, feats=None, next_weight: Optional[float] = None, return_dict: bool = True, **_):
        """wav(+wav_len, K) 또는 feats 중 하나. labels 는 -100 이 손실 제외. next_weight 기본 config.next_weight."""
        if wav is not None: feats = self.encode(wav, wav_len, K)
        nw = self.config.next_weight if next_weight is None else float(next_weight)
        h = self.thinker.model(inputs_embeds=self.build(feats, ids, is_audio, chunk_of), attention_mask=mask).last_hidden_state
        tgt = labels[:, 1:]; sel = tgt != -100; t = tgt[sel]
        logits = self.thinker.lm_head(h[:, :-1][sel]).float()                       # 라벨 위치만 (전 위치 fp32 logits 는 OOM)
        tok_loss = F.cross_entropy(logits, t, reduction="none")
        na = t == self.next_audio; tx = ~na
        w = torch.where(na, tok_loss.new_tensor(nw), tok_loss.new_tensor(1.0)); loss = (tok_loss * w).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            top1 = (logits.argmax(-1) == t) & tx
            out = VapAsrOutput(loss=loss, loss_next=tok_loss[na].mean().item() if na.any() else 0.0, loss_text=tok_loss[tx].mean().item() if tx.any() else 0.0,
                               top1_text=(top1.sum() / tx.sum().clamp(min=1)).item(), n_labels=int(t.numel()))
        return out if return_dict else (loss,)

    # ── 스트리밍 디코드 (mono_model.stream_decode 와 동일)
    @torch.inference_mode()
    def stream_decode(self, feats, prefix_ids: List[int], max_per_chunk: int = 0, next_bias: float = 0.0, max_flush_rounds: Optional[int] = None, runaway_cap: Optional[int] = None, max_total_per_chunk: float = 6.0):
        """feats (K,Din) 또는 wav (T,) → ([(chunk k, token id)], forced_next 횟수, tick_ms 목록, flush 라운드 수).
        폭주 방지: 청크당 runaway_cap(config, 기본 8) 토큰 · 스트림 전체 max_total_per_chunk×K 토큰(부분 학습 모델의 free-running 디코드가 수만 step 으로 늘어지던 문제)."""
        from transformers import DynamicCache
        max_flush_rounds = self.config.max_flush_rounds if max_flush_rounds is None else max_flush_rounds; runaway_cap = self.config.runaway_cap if runaway_cap is None else runaway_cap
        if feats.dim() == 1:
            K = torch.tensor([int(round(feats.shape[0] / 16000 * self.config.frame_hz))], device=feats.device)
            feats = self.encode(feats[None], torch.tensor([feats.shape[0]], device=feats.device), K)[0]
        emb = self.get_input_embeddings(); dev = feats.device; cache = DynamicCache(); ce = self.chunk_embed(feats[None])[0].to(emb.weight.dtype)
        def step(e):
            out = self.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True); return out.logits[0, -1].float()
        step(emb(torch.tensor(prefix_ids, device=dev))); e_next = emb.weight[self.next_audio]; e_empty = emb.weight[self.empty_audio]
        out, forced, ticks = [], 0, []; K = ce.shape[0]; max_total = int(max_total_per_chunk * K) + 64; cap = min(max_per_chunk, runaway_cap) if max_per_chunk else runaway_cap
        def emit_round(k, logits):
            nonlocal forced; n = 0
            while True:
                logits[self.blocked] = float("-inf"); logits[self.next_audio] -= next_bias
                tid = int(logits.argmax())
                if tid == self.next_audio or n >= cap or len(out) >= max_total:
                    forced += int(tid != self.next_audio); step(e_next); return n
                out.append((k, tid)); n += 1; logits = step(emb.weight[tid])
        for k in range(K):
            t = time.time(); emit_round(k, step(ce[k])); torch.cuda.synchronize(); ticks.append((time.time() - t) * 1000)
        rounds = 0
        for r in range(max_flush_rounds):
            rounds += 1
            if emit_round(K + r, step(e_empty)) == 0: break
        return out, forced, ticks, rounds

    # ── 저장/로드
    def save_pretrained(self, save_directory, *args, state_dict=None, **kw):
        """동결 인코더는 저장하지 않는다(.nemo 에서 재구성). 학습한 인코더(encoder_trainable) 는 함께 저장."""
        sd = state_dict if state_dict is not None else self.state_dict()
        if not self.config.encoder_trainable: sd = {k: v for k, v in sd.items() if not k.startswith("encoder.")}
        return super().save_pretrained(save_directory, *args, state_dict=sd, **kw)

    @classmethod
    def from_pretrained(cls, path, *args, encoder_path: Optional[str] = None, load_encoder: bool = True, **kw):
        m = super().from_pretrained(path, *args, **kw)
        if load_encoder:
            enc_sd = None
            if m.config.encoder_trainable:                                        # 저장된 인코더 가중치를 safetensors 에서 읽어 붙인다
                from safetensors.torch import load_file
                import glob
                enc_sd = {}
                for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                    for k, v in load_file(f).items():
                        if k.startswith("encoder.enc."): enc_sd[k[len("encoder.enc."):]] = v
            m.attach_encoder(encoder_path, state_dict=enc_sd or None)
        return m

    @classmethod
    def from_qwen(cls, qwen_dir: str, tokenizer=None, encoder_path: Optional[str] = None, load_encoder: bool = True, seed: int = 0, **cfg_kw):
        """Qwen3-ASR 디렉토리에서 새 모델. 특수 토큰을 tokenizer 에 추가하고 그 임베딩 행을 평균+잡음으로 초기화(mono_model 과 동일). adapter 는 random."""
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
        from transformers import AutoTokenizer
        from ..uslm.interleave_data import add_specials, SPECIAL_TOKENS
        tok = tokenizer or AutoTokenizer.from_pretrained(qwen_dir)
        sp_ids = add_specials(tok)
        full = Qwen3ASRForConditionalGeneration.from_pretrained(qwen_dir, dtype=torch.float32); thinker = full.thinker; del full
        c = thinker.config; pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        blocked = [c.audio_token_id, c.audio_start_token_id, c.audio_end_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|im_start|>"),
                   tok.convert_tokens_to_ids("<asr_text>"), sp_ids["<EMPTY_AUDIO>"], sp_ids["<SPK_A>"], sp_ids["<SPK_B>"]] + [v for k, v in sp_ids.items() if k.startswith("<DELAY_")]
        config = VapAsrConfig(thinker=c.to_dict(), thinker_name_or_path=qwen_dir, sp_ids=sp_ids, special_tokens=list(SPECIAL_TOKENS), blocked_ids=sorted(set(blocked)),
                              audio_pad_id=c.audio_token_id, prefix_ids=pre, **cfg_kw)
        m = cls(config, thinker=thinker)
        W = m.get_input_embeddings().weight; rows = sorted(sp_ids.values()); assert max(rows) < W.shape[0], "임베딩에 특수 토큰 여유 행 없음"
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            mu = W[: len(tok) - len(sp_ids)].float().mean(0)
            for r in rows: W[r] = (mu + 0.02 * torch.randn(mu.shape, generator=g)).to(W.dtype)
        for p in m.thinker.parameters(): p.requires_grad_(True)
        if load_encoder: m.attach_encoder(encoder_path)
        return m, tok

    @classmethod
    def from_legacy(cls, ckpt_path: str, qwen_dir: str, tokenizer=None, encoder_path: Optional[str] = None, load_encoder: bool = True, **cfg_kw):
        """experiments/s1_train_mono.py 산출물(ckpt-last.pt / ckpt-<step>.pt, full FT) → HF 모델."""
        st = torch.load(ckpt_path, map_location="cpu"); st = st["model"] if "model" in st and "adapter" not in st else st
        assert st.get("full_ft") and st.get("thinker"), "full FT ckpt 만 변환(LoRA 는 미지원)"
        m, tok = cls.from_qwen(qwen_dir, tokenizer, encoder_path, load_encoder=False, **cfg_kw)
        assert dict(st["sp_ids"]) == m.config.sp_ids, f"특수 토큰 id 불일치: {st['sp_ids']} vs {m.config.sp_ids}"
        m.adapter.load_state_dict(st["adapter"])
        missing, unexpected = m.thinker.load_state_dict({k: v.float() for k, v in st["thinker"].items()}, strict=False)
        missing = [k for k in missing if not k.startswith("audio_tower")]; assert not missing and not unexpected, (missing[:5], unexpected[:5])
        if st.get("encoder"): m.config.encoder_trainable = True
        if load_encoder: m.attach_encoder(encoder_path, state_dict=st.get("encoder"))
        return m, tok

    # ── 기존 학습 코드와의 호환(load_trainable_state 형식)
    def trainable_state(self):
        enc = {"encoder": {k: v.detach().cpu() for k, v in self.encoder.enc.state_dict().items()}} if (self.encoder is not None and self.config.encoder_trainable) else {}
        return dict(adapter=self.adapter.state_dict(), thinker={k: v.detach().to(torch.bfloat16).cpu() for k, v in self.thinker.state_dict().items()}, sp_ids=self.sp_ids, mono=True, full_ft=True, **enc)
