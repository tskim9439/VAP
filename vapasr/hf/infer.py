"""추론 유틸 — 오디오 파일 하나를 스트리밍 디코드하고 (a) 후처리 텍스트, (b) 후처리 없는 원시 토큰 출력(특수 토큰·청크 경계 포함)을 돌려준다.
노트북(IPykernel) 에서 셀 단위로 쓰도록 설계: load_model → load_audio → iter_stream(청크마다 yield) 또는 transcribe(한 번에).

  from vapasr.hf.infer import load_model, load_audio, transcribe, iter_stream, render_table
  model, tok = load_model("/soundai/Model/VAPASR/hf-C2/final")           # 또는 checkpoint-N 디렉토리, 기존 ckpt-last.pt
  wav = load_audio("/path/to/a.wav")                                       # 16 kHz mono float32 (다른 SR 은 리샘플)
  res = transcribe(model, tok, wav, lang="Korean", delay=2)
  res.text · res.raw · res.table(pandas) · res.chunks[k]
디코드 규약은 modeling_vapasr.stream_decode 와 동일(청크당 1 오디오 임베딩 → greedy → <NEXT_AUDIO> 로 청크 종료, 끝에 <EMPTY_AUDIO> flush)."""
import os, re, time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple
import numpy as np, torch

CHUNK_S = 0.08

# ───────────────────────────── 로드 ─────────────────────────────
def load_model(path: str, device: str = "cuda", encoder_path: Optional[str] = None, liger: bool = False, qwen_dir: Optional[str] = None, dtype: Optional[torch.dtype] = None):
    """HF 디렉토리(final/ 또는 checkpoint-N/) 또는 기존 ckpt-last.pt → (model.eval() on device, tokenizer)."""
    from . import VapAsrForStreamingASR, load_tokenizer
    qwen_dir = qwen_dir or os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")
    if os.path.isfile(path): model, tok = VapAsrForStreamingASR.from_legacy(path, qwen_dir, encoder_path=encoder_path)
    else:
        from .stage import stage_dir                     # NFS 위 체크포인트·.nemo 는 로컬 tmpfs 로 먼저 복사(VAPASR_STAGE_DIR="" 이면 끔)
        enc_src = encoder_path or os.environ.get("MXC_NEMOTRON_DIR", os.environ.get("NEMOTRON_DIR", ""))
        model = VapAsrForStreamingASR.from_pretrained(stage_dir(path), encoder_path=stage_dir(enc_src) if enc_src else None); tok = load_tokenizer(path)
    if liger:
        from .liger import apply_liger_to_thinker; apply_liger_to_thinker(model)
    model = model.to(device).eval()
    if model.encoder is not None: model.encoder.set_trainable(False)               # 추론: 학습된 인코더(E2 등)라도 동결·eval·fp32 경로로(autocast 는 cuda 학습 전용)
    if dtype is not None: model.thinker.to(dtype)                                 # 실시간 디코드(adapter 는 작아서 fp32 유지: chunk_embed 가 fp32 입력): fp32 가중치 + autocast 는 토큰마다 캐스팅해 느리다(H200 에서 thinker step ~25 ms → bf16 가중치 ~10 ms). 인코더는 fp32 유지
    return model, tok

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """flac/wav/mp3… → float32 mono @16 kHz. .pcm 은 16 kHz 16-bit raw 로 간주(KsponSpeech/NIKL)."""
    if path.endswith(".pcm"):
        from ..data.streams import load_utt_audio; return load_utt_audio(path)
    import soundfile as sf
    x, s = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
    if s != sr:
        import torchaudio.functional as AF; x = AF.resample(torch.from_numpy(x), s, sr).numpy()
    return np.ascontiguousarray(x, dtype=np.float32)

# ───────────────────────────── 디코드(제너레이터) ─────────────────────────────
@dataclass
class Chunk:
    k: int                      # 청크 번호(0 부터). k ≥ K 는 flush 라운드
    t0: float; t1: float        # 시간창(초). flush 는 (K·0.08, K·0.08)
    ids: List[int]              # 이 청크에서 방출된 텍스트 토큰 id (<NEXT_AUDIO> 제외)
    pieces: List[str]           # tokenizer 조각(바이트 조각 포함, 후처리 없음)
    forced: bool                # 상한에 걸려 <NEXT_AUDIO> 를 강제했는가
    tick_ms: float              # 이 청크 처리 시간

@dataclass
class Result:
    chunks: List[Chunk]; K: int; lang: str; delay: int; prefix: str
    @property
    def ids(self) -> List[int]: return [t for c in self.chunks for t in c.ids]
    @property
    def emitted(self) -> List[Tuple[int, int]]: return [(c.k, t) for c in self.chunks for t in c.ids]
    def text(self, tok, normalize: bool = True) -> str:
        """후처리 텍스트: 특수 토큰 제거 후 decode, 공백 정리(normalize=True 면 asr-tn 채점 정규화까지)."""
        s = tok.decode(self.ids, skip_special_tokens=True); s = re.sub(r"\s+", " ", s).strip()
        if not normalize: return s
        from ..data.textnorm import score_en, score_ko
        return score_ko(s, True) if self.lang == "Korean" else score_en(s)
    def raw(self, tok, with_audio: bool = True) -> str:
        """후처리 없는 원시 시퀀스: [AUDIO_k] 자리·방출 토큰(조각 그대로)·<NEXT_AUDIO>·<EMPTY_AUDIO> 를 모델이 낸 순서대로."""
        out = []
        for c in self.chunks:
            head = (f"[AUDIO_{c.k}]" if c.k < self.K else "<EMPTY_AUDIO>") if with_audio else ""
            out.append(head + "".join(c.pieces) + ("<NEXT_AUDIO>" if not c.forced else "<NEXT_AUDIO!>"))
        return "".join(out)
    def table(self, tok, skip_empty: bool = True):
        """pandas DataFrame: 청크 | 시간창 | 조각(원시) | 텍스트(이 청크) | 누적 텍스트 | tick ms"""
        import pandas as pd
        rows, acc = [], []
        for c in self.chunks:
            if skip_empty and not c.ids: continue
            acc += c.ids
            rows.append(dict(chunk=(c.k if c.k < self.K else f"flush{c.k - self.K}"), time=f"{c.t0*1000:.0f}–{c.t1*1000:.0f} ms" if c.k < self.K else "end", n=len(c.ids),
                             pieces=" ".join(repr(p) for p in c.pieces), text=tok.decode(c.ids, skip_special_tokens=True), cumulative=re.sub(r"\s+", " ", tok.decode(acc, skip_special_tokens=True)).strip(),
                             forced=c.forced, tick_ms=round(c.tick_ms, 1)))
        return pd.DataFrame(rows)
    def stats(self) -> dict:
        ticks = np.array([c.tick_ms for c in self.chunks if c.k < self.K]); n_tok = len(self.ids)
        return dict(K=self.K, audio_s=round(self.K * CHUNK_S, 2), tokens=n_tok, tok_per_chunk=round(n_tok / max(1, self.K), 3), forced=sum(c.forced for c in self.chunks),
                    flush_rounds=sum(1 for c in self.chunks if c.k >= self.K), tick_p50_ms=float(np.percentile(ticks, 50)) if len(ticks) else None, tick_p99_ms=float(np.percentile(ticks, 99)) if len(ticks) else None,
                    realtime_factor=round(float(ticks.sum() / 1000) / max(1e-6, self.K * CHUNK_S), 3) if len(ticks) else None)

def prefix_ids(model, tok, lang: str, delay: int) -> List[int]:
    return list(model.config.prefix_ids) + tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [model.config.sp_ids[f"<DELAY_{delay}>"]]

@torch.inference_mode()
def iter_stream(model, tok, wav: np.ndarray, lang: str = "English", delay: int = 2, next_bias: float = 0.0, runaway_cap: Optional[int] = None, max_flush_rounds: int = 8, max_total_per_chunk: float = 6.0) -> Iterator[Chunk]:
    """청크마다 Chunk 를 yield. 노트북에서 `for c in iter_stream(...): print(...)` 로 한 청크씩 볼 수 있다. 규약은 stream_decode 와 동일."""
    from transformers import DynamicCache
    dev = next(model.parameters()).device; cap = model.config.runaway_cap if runaway_cap is None else runaway_cap
    def sync():
        if dev.type == "cuda": torch.cuda.synchronize()
        elif dev.type == "mps": torch.mps.synchronize()
    use_ac = dev.type == "cuda" and model.get_input_embeddings().weight.dtype == torch.float32      # bf16/fp16 가중치·MPS/CPU 는 autocast 없이
    w = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(dev); K = int(round(w.shape[0] / 16000 / CHUNK_S))
    feats = model.encode(w[None], torch.tensor([w.shape[0]], device=dev), torch.tensor([K], device=dev))[0]
    emb = model.get_input_embeddings(); cache = DynamicCache(); ce = model.chunk_embed(feats[None])[0].to(emb.weight.dtype)
    def step(e):
        out = model.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True); return out.logits[0, -1].float()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
        step(emb(torch.tensor(prefix_ids(model, tok, lang, delay), device=dev)))
        e_next = emb.weight[model.next_audio]; e_empty = emb.weight[model.empty_audio]; total = 0; max_total = int(max_total_per_chunk * K) + 64
        def emit_round(logits):
            nonlocal total; ids, forced = [], False
            while True:
                logits[model.blocked] = float("-inf"); logits[model.next_audio] -= next_bias; tid = int(logits.argmax())
                if tid == model.next_audio or len(ids) >= cap or total >= max_total:
                    forced = tid != model.next_audio; step(e_next); return ids, forced
                ids.append(tid); total += 1; logits = step(emb.weight[tid])
        for k in range(K):
            t = time.time(); ids, forced = emit_round(step(ce[k])); sync()
            yield Chunk(k, k * CHUNK_S, (k + 1) * CHUNK_S, ids, tok.convert_ids_to_tokens(ids), forced, (time.time() - t) * 1000)
        for r in range(max_flush_rounds):
            t = time.time(); ids, forced = emit_round(step(e_empty)); sync()
            yield Chunk(K + r, K * CHUNK_S, K * CHUNK_S, ids, tok.convert_ids_to_tokens(ids), forced, (time.time() - t) * 1000)
            if not ids: break

def transcribe(model, tok, wav: np.ndarray, lang: str = "English", delay: int = 2, **kw) -> Result:
    chunks = list(iter_stream(model, tok, wav, lang, delay, **kw)); K = int(round(len(wav) / 16000 / CHUNK_S))
    return Result(chunks, K, lang, delay, tok.decode(prefix_ids(model, tok, lang, delay)))
