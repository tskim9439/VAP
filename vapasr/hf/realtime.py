"""실시간 스트리밍 경로(2026-10-12): NeMo 캐시 재사용 인코더 step + 고정 KV 캐시·CUDA graph thinker 디코드.

평가 경로(VapAsrForStreamingASR.stream_decode: HF DynamicCache, 토큰마다 eager forward)와 같은 출력을 목표로 하되, 한 스트림
실시간 처리에서 CPU 쪽 비용(Python·커널 실행)을 없앤다.
- GraphThinker: thinker 언어모델(Qwen3 와 같은 구조 — q/k RMSNorm, GQA, RoPE θ=1e6; 위치가 1 차원이라 interleaved MRoPE 는 보통 RoPE 와
  같다)을 고정 길이 KV 캐시로 돌린다. 레이어의 Linear·RMSNorm·MLP 모듈은 thinker 것을 그대로 부르고, 어텐션 core(SDPA + 위치 마스크)·KV 쓰기·
  RoPE 표만 여기서 한다. 입력 위치 1·2 개짜리 forward 를 CUDA graph 로 잡고, 차단 토큰 −∞·argmax 까지 graph 안에서 끝낸다.
- realtime_decode: stream_decode 와 같은 규칙(청크마다 audio 위치 → argmax 가 <NEXT_AUDIO> 거나 청크 상한·전체 상한이면 끝, 아니면
  토큰을 내고 그 토큰을 넣는다; 입력 끝 뒤 <EMPTY_AUDIO> flush 라운드). 다른 점은 하나: stream_decode 는 <NEXT_AUDIO> 를 넣는 forward 의
  logits 를 버리므로, 그것을 다음 audio 위치(또는 flush 의 <EMPTY_AUDIO>)와 한 forward(위치 2 개)로 합친다 — 출력 규칙은 같다.
- StreamingEncoder: Nemotron 인코더를 NeMo cache_aware_stream_step 으로 320 ms(4 프레임) 청크마다 돌린다(NeMo CacheAwareStreamingAudioBuffer 와
  같은 특징 자르기: 첫 청크 25 mel 프레임, 이후 32 + 앞 9 프레임 캐시, 첫 step 외 drop_extra_pre_encoded). set_streaming_cuda_graphs 로
  정상 상태 step 을 CUDA graph 로 돈다. mel 은 파일 전체에서 한 번 계산한다(NeMo 스트리밍 시뮬레이션과 같다).
"""
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class GraphThinker:
    """thinker(Qwen3ASRThinkerForConditionalGeneration)의 언어모델을 고정 KV 캐시(max_len)로 돌린다. forward(x (q,D), pos (q,)) → 다음 토큰 id."""

    def __init__(self, thinker, blocked_ids, max_len: int = 2048, graph_lens=(1, 2), all_lens=()):
        tm = thinker.model; cfg = tm.config
        self.layers, self.norm, self.lm_head, self.embed = tm.layers, tm.norm, thinker.lm_head, tm.embed_tokens.weight
        self.H, self.KV, self.HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        dev, dt = self.embed.device, self.embed.dtype
        self.max_len, self.dtype, self.device = max_len, dt, dev
        L = len(self.layers)
        self.k = torch.zeros(L, 1, self.KV, max_len, self.HD, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        rot = tm.rotary_emb
        assert float(getattr(rot, "attention_scaling", 1.0)) == 1.0
        freqs = torch.arange(max_len, device=dev, dtype=torch.float32)[:, None] * rot.inv_freq.float()[None, :]   # 위치마다 (HD/2): HF 의 inv_freq @ pos 와 같은 곱
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos, self.sin = emb.cos().to(dt), emb.sin().to(dt)
        self.ar = torch.arange(max_len, device=dev)
        self.block = torch.zeros(self.embed.shape[0], dtype=torch.bool, device=dev)
        self.block[torch.as_tensor(list(blocked_ids), dtype=torch.long, device=dev)] = True
        self.graphs, self.graphs_all = {}, {}
        self._capture(graph_lens)
        self._capture(all_lens, every=True)                                                  # E9 run_all(위치마다 argmax) — 기본 없음

    @torch.inference_mode()
    def _forward(self, x, pos):
        """x (q, D), pos (q,) long → 다음 토큰 id (0-dim long, 차단 토큰 −∞ 후 argmax). KV 캐시의 pos 자리에 쓴다."""
        logits = self.lm_head(self.norm(self._hidden(x, pos)[:, -1]))[0].float()
        return logits.masked_fill(self.block, float("-inf")).argmax()

    @torch.inference_mode()
    def _forward_all(self, x, pos):
        """_forward 와 같고 위치마다 다음 토큰 id (q,) long 을 돌려준다(E9 추측 prefill — realtime_block_decode)."""
        logits = self.lm_head(self.norm(self._hidden(x, pos)))[0].float()
        return logits.masked_fill(self.block, float("-inf")).argmax(-1)

    @torch.inference_mode()
    def _hidden(self, x, pos):
        """x (q, D), pos (q,) long → 마지막 층 hidden (1, q, D)(norm 전). KV 캐시의 pos 자리에 쓴다."""
        q = x.shape[0]
        h = x[None]
        cos, sin = self.cos[pos][None, None], self.sin[pos][None, None]                # (1,1,q,HD)
        mask = (self.ar[None, :] <= pos[:, None])[None, None]                            # (1,1,q,max_len): 자기 위치까지
        for li, layer in enumerate(self.layers):
            a = layer.self_attn
            r = h; hn = layer.input_layernorm(h)
            qs = a.q_norm(a.q_proj(hn).view(1, q, self.H, self.HD)).transpose(1, 2)
            ks = a.k_norm(a.k_proj(hn).view(1, q, self.KV, self.HD)).transpose(1, 2)
            vs = a.v_proj(hn).view(1, q, self.KV, self.HD).transpose(1, 2)
            qs = qs * cos + _rotate_half(qs) * sin
            ks = ks * cos + _rotate_half(ks) * sin
            self.k[li].index_copy_(2, pos, ks); self.v[li].index_copy_(2, pos, vs)
            o = F.scaled_dot_product_attention(qs, self.k[li], self.v[li], attn_mask=mask, scale=a.scaling, enable_gqa=True)
            h = r + a.o_proj(o.transpose(1, 2).reshape(1, q, self.H * self.HD))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h

    def _capture(self, lens, every: bool = False):
        fwd, store = (self._forward_all, self.graphs_all) if every else (self._forward, self.graphs)
        for q in lens:
            xb = torch.zeros(q, self.embed.shape[1], device=self.device, dtype=self.dtype)
            pb = torch.arange(q, device=self.device)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): fwd(xb, pb)                                               # 준비(할당기·커널 선택)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = fwd(xb, pb)
            store[q] = (g, xb, pb, out)

    def run(self, x, start: int) -> int:
        """x (q, D) 를 위치 start.. 에 넣고 다음 토큰 id 를 돌려준다(q 가 잡아 둔 길이면 graph, 아니면 eager — prefix 등)."""
        q = x.shape[0]
        assert start + q <= self.max_len, f"KV 캐시 길이 {self.max_len} 를 넘는다"
        if q in self.graphs:
            g, xb, pb, out = self.graphs[q]
            xb.copy_(x); pb.copy_(torch.arange(start, start + q, device=self.device)); g.replay()
            return int(out)
        return int(self._forward(x, torch.arange(start, start + q, device=self.device)))

    def run_all(self, x, start: int) -> List[int]:
        """run 과 같고 위치마다 다음 토큰 id 목록(E9 추측 prefill — 잡아 둔 길이(all_lens)면 graph, 아니면 eager)."""
        q = x.shape[0]
        assert start + q <= self.max_len, f"KV 캐시 길이 {self.max_len} 를 넘는다"
        if q in self.graphs_all:
            g, xb, pb, out = self.graphs_all[q]
            xb.copy_(x); pb.copy_(torch.arange(start, start + q, device=self.device)); g.replay()
            return out.tolist()
        return self._forward_all(x, torch.arange(start, start + q, device=self.device)).tolist()


@torch.inference_mode()
def realtime_decode(model, gt: GraphThinker, feats, prefix_ids: List[int], max_flush_rounds: Optional[int] = None,
                    max_per_chunk: int = 0, max_total_per_chunk: float = 6.0, on_chunk=None, retag=None):
    """stream_decode 와 같은 규칙의 디코드. feats (1,K,Din) 또는 (K,Din) → (out [(k, tid)], forced, step_ms [K], flush_rounds, flush_ms).
    step_ms[k] = 오디오 프레임 k 를 넣고 그 프레임의 토큰을 다 낼 때까지의 GPU 동기 시간(ms).
    on_chunk(k, 오디오 k 가 놓일 위치): 청크 k 라운드 직전에 부른다(long_session.LongThinker 의 문맥 창 — 없으면 예전 그대로).
    retag(k, tid) → tid(2026-10-28, E7 회의 통째 — long_session.make_retag): 낼 토큰을 바꾼다(화자 태그를 활동으로). 바꾼 토큰을 그대로 다시 넣는다."""
    cfg = model.config
    max_flush_rounds = cfg.max_flush_rounds if max_flush_rounds is None else max_flush_rounds
    runaway_cap = cfg.runaway_cap
    if feats.dim() == 2: feats = feats[None]
    emb = gt.embed
    ce = model.chunk_embed(feats[None])[0].to(emb.dtype)                                 # (K, D) — stream_decode 와 같은 adapter·형변환
    e_next, e_empty = emb[model.next_audio], emb[model.empty_audio]
    K = ce.shape[0]; max_total = int(max_total_per_chunk * K) + 64
    cap = min(max_per_chunk, runaway_cap) if max_per_chunk else runaway_cap
    pos = 0; tid = gt.run(emb[torch.as_tensor(prefix_ids, device=emb.device)], 0); pos = len(prefix_ids)   # prefix: 마지막 logits 는 쓰지 않는다
    out, forced, ticks, pending = [], 0, [], False

    def feed(x):
        nonlocal pos
        t = gt.run(x, pos); pos += x.shape[0]; return t

    def emit_round(k, x):
        nonlocal forced, pending
        n = 0; tid = feed(torch.stack([e_next, x]) if pending else x[None])           # 앞 라운드의 <NEXT_AUDIO> 를 이 위치와 함께 넣는다
        pending = False
        while True:
            if tid == model.next_audio or n >= cap or len(out) >= max_total:
                forced += int(tid != model.next_audio); pending = True; return n
            if retag is not None: tid = retag(k, tid)
            out.append((k, tid)); n += 1; tid = feed(emb[tid][None])
    for k in range(K):
        if on_chunk is not None: on_chunk(k, pos + int(pending))                       # 오디오 k 는 앞 라운드 <NEXT_AUDIO> 뒤에 놓인다
        torch.cuda.synchronize(); t = time.perf_counter()
        emit_round(k, ce[k]); torch.cuda.synchronize(); ticks.append((time.perf_counter() - t) * 1000)
    rounds = 0; t = time.perf_counter()
    for r in range(max_flush_rounds):
        rounds += 1
        if emit_round(K + r, e_empty) == 0: break
    torch.cuda.synchronize()
    return out, forced, ticks, rounds, (time.perf_counter() - t) * 1000


@torch.inference_mode()
def realtime_block_decode(model, gt, feats, prefix_ids: List[int], D: int, act=None, max_flush_rounds: Optional[int] = None, max_per_chunk: int = 0,
                          max_total_per_chunk: float = 6.0, on_chunk=None, mix_off: bool = False, gate=None, trans: bool = False):
    """E9 슬롯 블록 디코드(2026-10-28 — cst_dialogue 모듈 설명 E9): batch_decode.decode_blocks_batch 와 같은 출력 규칙을 GraphThinker·LongThinker(run·run_all)로 돈다.
    청크마다 [오디오 k, 태그 0 … D−1] 를 한 forward 로 미리 넣고(run_all — 위치마다 argmax), 태그 s 뒤 예측이 멈춤이 아니면 태그 s 바로 뒤로 되돌아가 그 슬롯 출력을
    한 토큰씩 낸 뒤(뒤 태그 자리는 덮어쓴다 — 쿼리는 자기 위치까지만 본다) 남은 태그를 다시 한 forward 로 넣는다. 모든 슬롯이 멈추는 청크는 forward 한 번이다.
    feats (1,K,Din)·(S,K,Din) 또는 (K,Din), act (K, N) 태그 활동(없으면 model.block_act). on_chunk(k, 오디오 k 위치): LongThinker 문맥 창.
    E10(model.adapter_slotx): 태그마다 그 슬롯 소리 토큰(model.slot_audio_of)을 더한다. mix_off: 섞인 소리 토큰을 adapter_mixnull 로(decode_blocks_batch 와 같다).
    gate(batch_decode.ActGate, 2026-10-30 진단): 태그 s 뒤 첫 예측이 멈춤이 아니어도 게이트가 막으면 그 라운드 그 슬롯은 말을 내지 않는다(decode_blocks_batch 와 같은 규칙 —
    뒤 태그는 이미 그 꼴로 미리 넣었다).
    E13(2026-11-02): config.slot_gate 모델은 라운드마다 활동 창·열린 발화(<ONSET> 뒤 <EOT> 전) 슬롯 태그만 넣고(batch_decode.BlockState 와 같은 규칙), trans 면
    그 칸마다 [<TRANS>] 번역 칸이 뒤따른다 — 칸(태그·<TRANS>)들을 태그처럼 한 forward 로 미리 넣고 같은 방식으로 되돌아간다.
    → (emitted [(k, tid)] — 슬롯 출력 앞에 그 슬롯 태그(번역 칸이면 태그·<TRANS>), forced, step_ms [K], flush_rounds, flush_ms, forward 수)."""
    from .batch_decode import block_gate, block_stops
    cfg = model.config
    max_flush_rounds = cfg.max_flush_rounds if max_flush_rounds is None else max_flush_rounds
    if feats.dim() == 2: feats = feats[None]
    emb = gt.embed
    ce = model.chunk_embed(feats[None], mix_mask=torch.ones(1, dtype=torch.bool, device=emb.device) if mix_off else None)[0].to(emb.dtype)   # (K, D) — decode_blocks_batch 와 같다
    A = model.block_act(feats, act); X = model.slot_audio_of(feats, A) if getattr(model, "adapter_slotx", None) is not None else None
    K = ce.shape[0]; max_total = int(max_total_per_chunk * (2 if trans else 1) * K) + 64
    cap = min(max_per_chunk, cfg.runaway_cap) if max_per_chunk else cfg.runaway_cap
    tags, stop, e_empty = model.slot_ids(D), block_stops(model, D, trans), emb[model.empty_audio]
    gl, ga = block_gate(model, A); TR = int(model.sp_ids["<TRANS>"]) if trans else -1
    BG, EOT = int(model.sp_ids.get("<ONSET>", -1)), int(model.sp_ids["<EOT>"]); utt = [0] * D       # E13 열린 발화 수(<ONSET> − <EOT> — BlockState 와 같다)
    sync = torch.cuda.synchronize if emb.is_cuda else (lambda: None)
    gt.run(emb[torch.as_tensor(prefix_ids, device=emb.device)], 0); pos = len(prefix_ids)    # prefix: 마지막 logits 는 쓰지 않는다
    out, ticks, st = [], [], dict(forced=0, total=0, forwards=1)

    def block_round(k, head):
        """라운드 k(머리 = 오디오 k 또는 <EMPTY_AUDIO>) → 이 라운드에 낸 출력 수."""
        nonlocal pos
        if gl:
            lo, hi = max(0, k - gl + 1), min(ga.shape[0], k + 1)
            sl = [s for s in range(D) if utt[s] > 0 or (s < ga.shape[1] and hi > lo and bool(ga[lo:hi, s].any()))]
        else: sl = range(D)
        items = [(s, x) for s in sl for x in ((0, 1) if trans else (0,))]
        te = model.tag_embeds(k, A, D, X).to(emb.dtype)
        ie = torch.stack([te[s] if x == 0 else emb[TR] for s, x in items]) if items else te[:0]
        r = gt.run_all(torch.cat([head[None], ie]), pos); st["forwards"] += 1
        n_it = len(items); pr = dict(enumerate(r[1:])); at = {j: pos + 1 + j for j in range(n_it)}; pos += 1 + n_it; n_round = 0   # 칸 j 뒤 예측·칸 j 위치
        for j, (s, x) in enumerate(items):
            tid = pr[j]
            if tid in stop or (gate is not None and not gate.allows(k, s, tid)): continue          # 게이트가 막은 슬롯도 멈춤과 같다(뒤 칸 자리 그대로)
            pos = at[j] + 1; n = 0; tagged = False                                              # 칸 j 바로 뒤로 되돌아간다
            while True:
                if tid in stop or n >= cap or st["total"] >= max_total: st["forced"] += int(tid not in stop); break
                if not tagged: out.extend([(k, tags[s])] + ([(k, TR)] if x else [])); tagged = True
                out.append((k, int(tid))); n += 1; n_round += 1; st["total"] += 1
                if gl and tid in (BG, EOT): utt[s] = utt[s] + 1 if tid == BG else max(0, utt[s] - 1)
                tid = gt.run(emb[tid][None], pos); pos += 1; st["forwards"] += 1
            if j + 1 < n_it:                                                                   # 남은 칸을 다시 미리 넣는다
                r = gt.run_all(ie[j + 1:], pos); st["forwards"] += 1
                for jj in range(j + 1, n_it): pr[jj], at[jj] = r[jj - j - 1], pos + jj - j - 1
                pos += n_it - j - 1
        return n_round
    for k in range(K):
        if on_chunk is not None: on_chunk(k, pos)
        sync(); t = time.perf_counter()
        block_round(k, ce[k]); sync(); ticks.append((time.perf_counter() - t) * 1000)
    rounds = 0; t = time.perf_counter()
    for r_ in range(max_flush_rounds):
        rounds += 1
        if block_round(K + r_, e_empty) == 0: break
    sync()
    return out, st["forced"], ticks, rounds, (time.perf_counter() - t) * 1000, st["forwards"]


class StreamingEncoder:
    """Nemotron 인코더(NemotronOnline: pre=mel, enc=ConformerEncoder)를 캐시 재사용 스트리밍으로 돌린다."""

    def __init__(self, nemotron_online, cuda_graphs: bool = True):
        self.pre, self.enc = nemotron_online.pre, nemotron_online.enc
        if self.enc.streaming_cfg is None: self.enc.setup_streaming_params()
        self.cfg = self.enc.streaming_cfg
        if cuda_graphs: self.enc.set_streaming_cuda_graphs(True)
        pe = getattr(self.enc, "pre_encode", None)
        self.sampling_frames = pe.get_sampling_frames() if pe is not None and hasattr(pe, "get_sampling_frames") else None

    @staticmethod
    def _pick(v, first):
        return (v[0] if first else v[1]) if isinstance(v, list) else v

    @torch.inference_mode()
    def encode(self, wav: torch.Tensor, K: int) -> Tuple[torch.Tensor, List[float], float]:
        """wav (L,) float32 cuda → (특징 (1, K, D) — NemotronOnline.forward 와 같이 K 프레임으로 자르거나 0 패딩, 청크별 인코더 ms, mel ms)."""
        torch.cuda.synchronize(); t = time.perf_counter()
        f, fl = self.pre(input_signal=wav[None].float(), length=torch.tensor([wav.shape[0]], device=wav.device))
        torch.cuda.synchronize(); mel_ms = (time.perf_counter() - t) * 1000
        T = int(fl[0]); f = f[:, :, :T]
        cache = self.enc.get_initial_cache_state(batch_size=1, dtype=f.dtype, device=f.device)
        idx, step, outs, ms = 0, 0, [], []
        while idx < T:
            first = idx == 0
            size, shift = self._pick(self.cfg.chunk_size, first), self._pick(self.cfg.shift_size, first)
            pre_n = self._pick(self.cfg.pre_encode_cache_size, first)
            chunk = f[:, :, idx: idx + size]
            if self.sampling_frames is not None and chunk.shape[-1] < self._pick(self.sampling_frames, first):
                break                                                                       # NeMo 버퍼와 같이 서브샘플링 최소 길이보다 짧은 끝 조각은 버린다
            if pre_n > 0:
                c = f[:, :, max(0, idx - pre_n): idx]
                if c.shape[-1] < pre_n: c = torch.cat((c.new_zeros(1, f.shape[1], pre_n - c.shape[-1]), c), dim=-1)
                chunk = torch.cat((c, chunk), dim=-1); added = pre_n
            else:
                added = 0
            length = torch.tensor([min(T - idx, size) + added], device=f.device)
            drop = 0 if first else self.cfg.drop_extra_pre_encoded
            torch.cuda.synchronize(); t = time.perf_counter()
            h, hl, *cache = self.enc.cache_aware_stream_step(processed_signal=chunk, processed_signal_length=length, cache_last_channel=cache[0],
                                                              cache_last_time=cache[1], cache_last_channel_len=cache[2],
                                                              keep_all_outputs=idx + shift >= T, drop_extra_pre_encoded=drop)
            torch.cuda.synchronize(); ms.append((time.perf_counter() - t) * 1000)
            outs.append(h[:, :, : int(hl[0])]); idx += shift; step += 1
        h = torch.cat(outs, dim=-1).float().transpose(1, 2)                                 # (1, T', D)
        out = h.new_zeros(1, K, h.shape[2]); n = min(K, h.shape[1]); out[:, :n] = h[:, :n]
        return out, ms, mel_ms
