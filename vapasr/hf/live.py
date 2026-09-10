"""실시간(마이크) 스트리밍 인식 세션 — 학습 규약(80 ms 청크·<DELAY_δ>)을 그대로 따르되, 오디오를 받는 대로 인코더(cache-aware)와 thinker(KV cache)를 한 청크씩 전진시킨다.

StreamingEncoder: Nemotron 3.5 streaming 인코더(chunked_limited, att_context [56,0], 서브샘플링 8×)를 NeMo cache_aware_stream_step 으로 80 ms 마다 한 프레임씩 낸다.
  오프라인 전체 인코딩(NemotronOnline.forward, 학습 특징)과 프레임 단위로 같아야 한다 — experiments/probe_live_encoder.py 로 검증(2026-09-10).
LiveSession: infer.iter_stream 과 같은 디코드 규약(prefix → 청크마다 [AUDIO_k] 임베딩 → 텍스트 토큰… <NEXT_AUDIO>)을 feed()/finish() 로 쪼갠 것.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np, torch

SR, CHUNK_S, HOP = 16000, 0.08, 160          # mel hop 10 ms, 청크 = 8 mel 프레임 = 1 인코더 프레임
def _sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize()
    elif dev.type == "mps": torch.mps.synchronize()
MEL_TAIL = 256                                # STFT center 패딩(n_fft/2): 프레임 t 는 t·HOP+256 샘플까지 있어야 확정

class StreamingEncoder:
    """NemotronOnline(동결) 을 감싸 오디오를 누적 입력받고 확정된 인코더 프레임 (D,) 을 순서대로 낸다."""
    def __init__(self, nem):
        self.pre, self.enc = nem.pre, nem.enc; self.enc.eval()
        if self.enc.streaming_cfg is None: self.enc.setup_streaming_params()
        self.cfg = self.enc.streaming_cfg; self.dev = next(self.enc.parameters()).device; self.reset()
    def reset(self):
        self.cache = self.enc.get_initial_cache_state(batch_size=1, device=self.dev)      # (last_channel, last_time, last_channel_len)
        self.wav = torch.zeros(0, device=self.dev); self.consumed = 0; self.step = 0; self.frames: List[torch.Tensor] = []
    @torch.inference_mode()
    def _mel(self, upto_frames: int) -> torch.Tensor:
        """지금까지의 오디오 전체로 mel (1,128,T) 를 계산(GPU 에서 수 ms; 60 s 까지 문제없음). 시작 반사 패딩이 오프라인과 같아진다."""
        f, _ = self.pre(input_signal=self.wav[None], length=torch.tensor([self.wav.shape[0]], device=self.dev)); return f[:, :, :upto_frames]
    @torch.inference_mode()
    def feed(self, pcm: np.ndarray, final: bool = False) -> List[torch.Tensor]:
        """pcm float32 @16 kHz 를 붙이고 새로 확정된 인코더 프레임 목록을 돌려준다. final=True 면 끝 반사 패딩까지 써서 남은 프레임을 모두 낸다."""
        if len(pcm): self.wav = torch.cat([self.wav, torch.as_tensor(np.asarray(pcm, dtype=np.float32), device=self.dev)])
        n = self.wav.shape[0]
        if n < 400: return []                                                             # 한 프레임(25 ms)도 안 되는 오디오(정지 직후 등) → preemphasis 가 빈 텐서로 죽는다
        total = n // HOP + 1                                                              # 오프라인 mel 프레임 수(center=True)
        avail = total if final else max(0, (n - MEL_TAIL) // HOP + 1)                    # 확정 프레임 수
        cs0, cs = self.cfg.chunk_size if isinstance(self.cfg.chunk_size, list) else (self.cfg.chunk_size,) * 2
        pc = self.cfg.pre_encode_cache_size[1] if isinstance(self.cfg.pre_encode_cache_size, list) else self.cfg.pre_encode_cache_size
        out = []; mel = None
        while True:
            need = cs0 if self.step == 0 else cs
            if self.consumed + need > avail: break
            if mel is None: mel = self._mel(avail)
            if self.step == 0: x = mel[:, :, :cs0]
            else:
                lo = max(0, self.consumed - pc); x = mel[:, :, lo: self.consumed + cs]
                if self.consumed - lo < pc: x = torch.cat([torch.zeros(1, x.shape[1], pc - (self.consumed - lo), device=self.dev), x], -1)
            drop = 0 if self.step == 0 else self.cfg.drop_extra_pre_encoded
            enc, enc_len, c1, c2, c3 = self.enc.cache_aware_stream_step(processed_signal=x, processed_signal_length=torch.tensor([x.shape[-1]], device=self.dev),
                                                                        cache_last_channel=self.cache[0], cache_last_time=self.cache[1], cache_last_channel_len=self.cache[2],
                                                                        keep_all_outputs=False, drop_extra_pre_encoded=drop)
            self.cache = (c1, c2, c3); fr = enc[0, :, : int(enc_len[0])].transpose(0, 1).float()   # (t, D)
            for i in range(fr.shape[0]): self.frames.append(fr[i]); out.append(fr[i])
            self.consumed += cs0 if self.step == 0 else cs; self.step += 1
        return out

@dataclass
class LiveEvent:
    k: int; ids: List[int]; forced: bool; tick_ms: float; text: str          # text = 지금까지의 전사(후처리)
    enc_ms: float = 0.0; dec_ms: float = 0.0                                  # 청크 처리 시간 분해(인코더 프레임 / thinker 디코드)

class LiveSession:
    """model(VapAsrForStreamingASR, 인코더 부착·eval) + tok. feed(pcm) → 청크마다 LiveEvent. delay 기본 4(사용자 지정 가능)."""
    def __init__(self, model, tok, lang: str = "Korean", delay: int = 4, next_bias: float = 0.0, runaway_cap: Optional[int] = None, max_flush_rounds: int = 8):
        from transformers import DynamicCache
        from .infer import prefix_ids
        self.model, self.tok, self.lang, self.delay, self.next_bias = model, tok, lang, delay, next_bias
        self.cap = model.config.runaway_cap if runaway_cap is None else runaway_cap; self.max_flush = max_flush_rounds
        self.dev = next(model.parameters()).device; self.emb = model.get_input_embeddings(); self.cache = DynamicCache()
        self.enc = StreamingEncoder(model.encoder); self.ids: List[int] = []; self.k = 0; self.done = False
        self.e_next = self.emb.weight[model.next_audio]; self.e_empty = self.emb.weight[model.empty_audio]
        self.autocast = self.emb.weight.dtype == torch.float32 and self.dev.type == "cuda"   # 가중치가 bf16/fp16 이거나 MPS 면 autocast 없이
        self.block_add = torch.zeros(self.emb.weight.shape[0], device=self.dev); self.block_add[model.blocked] = float("-inf")   # 금지 토큰 마스크(팬시 인덱싱 대신 덧셈: 토큰당 수 ms 절약)
        self.block_add[model.next_audio] -= next_bias
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            self._step(self.emb(torch.tensor(prefix_ids(model, tok, lang, delay), device=self.dev)))
    def _step(self, e):
        out = self.model.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=self.cache, use_cache=True); return out.logits[0, -1].float()
    def _emit_round(self, logits):
        """텍스트 토큰을 <NEXT_AUDIO> 가 나올 때까지 뽑는다. 마지막 <NEXT_AUDIO> 입력은 바로 넣지 않고 보류(self._pending)했다가 다음 청크 임베딩과 한 번의 forward 로 합친다
        — KV cache 가 보는 시퀀스는 동일하고 thinker 호출이 청크당 1 회 준다(M4 MPS 디코더 53 → ~35 ms)."""
        ids, forced = [], False
        while True:
            tid = int((logits + self.block_add).argmax())
            if tid == self.model.next_audio or len(ids) >= self.cap:
                forced = tid != self.model.next_audio; self._pending = True; return ids, forced
            ids.append(tid); logits = self._step(self.emb.weight[tid])
    def _step_chunk(self, e):
        """보류된 <NEXT_AUDIO> 임베딩이 있으면 [e_next, e] 두 위치를 한 번에 넣고 마지막 위치 logits 를 돌려준다."""
        if getattr(self, "_pending", False): self._pending = False; return self._step(torch.stack([self.e_next, e]))
        return self._step(e)
    def text(self) -> str:
        import re; return re.sub(r"\s+", " ", self.tok.decode(self.ids, skip_special_tokens=True)).strip()
    @torch.inference_mode()
    def feed(self, pcm: np.ndarray) -> List[LiveEvent]:
        """마이크 PCM(float32 16 kHz, 길이 자유)을 넣고 새로 처리된 청크의 이벤트를 돌려준다."""
        if self.done: return []
        t0 = time.time(); frames = self.enc.feed(pcm)
        _sync(self.dev)
        enc_ms = (time.time() - t0) * 1000 / max(1, len(frames)); events = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            for fr in frames:
                t = time.time(); ce = self.model.chunk_embed(fr[None, None, None])[0, 0].to(self.emb.weight.dtype)   # (1,1,1,Din) → (D,)
                ids, forced = self._emit_round(self._step_chunk(ce)); self.ids += ids
                _sync(self.dev)
                dec_ms = (time.time() - t) * 1000; events.append(LiveEvent(self.k, ids, forced, enc_ms + dec_ms, self.text(), enc_ms, dec_ms)); self.k += 1
        return events
    @torch.inference_mode()
    def finish(self) -> List[LiveEvent]:
        """남은 오디오 프레임을 확정하고 <EMPTY_AUDIO> flush 라운드로 미방출 토큰을 뽑는다."""
        events = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            for fr in self.enc.feed(np.zeros(0, np.float32), final=True):
                t = time.time(); ce = self.model.chunk_embed(fr[None, None, None])[0, 0].to(self.emb.weight.dtype); ids, forced = self._emit_round(self._step_chunk(ce)); self.ids += ids
                events.append(LiveEvent(self.k, ids, forced, (time.time() - t) * 1000, self.text())); self.k += 1
            for r in range(self.max_flush):
                t = time.time(); ids, forced = self._emit_round(self._step_chunk(self.e_empty)); self.ids += ids
                events.append(LiveEvent(self.k + r, ids, forced, (time.time() - t) * 1000, self.text()))
                if not ids: break
        self.done = True; return events
