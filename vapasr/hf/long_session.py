"""회의 통째 디코드(2026-10-24, 사용자 "AMI 평가에 한번에 돌릴 수 있도록 창 길이를 늘리고 싶은데, 자르지 않고"): 수십 분 세션을 창으로 자르지 않고
한 스트림으로 디코드한다 — experiments/cst_stage3_eval.py run --window full.

- encode_long: 인코더를 겹치는 조각으로 돌려 이어 붙인다. 파일 전체 forward 는 rel_pos 어텐션이 O(T²) 라 50 분(T' 3.7 만 프레임)이면 층마다 수십 GB 다.
  Nemotron 은 chunked_limited [56, rc](청크 = rc + 1 프레임, 청크 번호는 입력 첫 프레임부터)·causal conv(커널 9)·causal 서브샘플링·mel 정규화 없음이라,
  조각 시작을 청크 경계에 두고 앞 ctx 프레임(기본 1760 = 140.8 s ≥ 수용 영역 24 층 × (59 + 8) = 1608)·뒤 tail 프레임을 겹쳐 버리면
  파일 전체 forward 와 같은 특징이다(부동소수 오차만 — cst_stage3_eval --verify 가 5 분 구간으로 잰다).
- LongThinker: thinker 언어모델(Qwen3 구조)을 eager 로 돈다. 층 연산·RoPE 계산은 realtime.GraphThinker 와 같고, 어텐션만 지금까지의 위치
  [0, start + q) 를 blk 위치 블록으로 나눈 행렬곱(쿼리 머리를 KV 머리 묶음으로 — GQA 를 확장하지 않는다)으로 한다. KV 캐시는 모자라면 늘린다.
  fp32 SDPA(math 커널 — GQA 확장)는 위치 10 만에서 층마다 2.4–5.2 ms, 블록 행렬곱은 0.43–0.48 ms 다(2026-10-24 H200 실측, 출력 차이 ≤ 4e-8).
- long_decode: realtime.realtime_decode 에 GraphThinker 대신 LongThinker 를 넣는다(run·embed 규약이 같다) — 청크 상한·flush·전체 상한 규칙은
  stream_decode·decode_batch 와 같다(2026-10-12 192 디코드 토큰 대조).
- AOSC(E7, 2026-10-27): 화자 분리 헤드(E6·E7) 모델이면 encode_long(aosc=…)이 조각마다 인코더 탭에서 가지 임베딩을 이어 붙여 회의 전체를 AOSC 스트리밍
  (diar.aosc_stream)으로 돌린다 — 예측 슬롯이 회의 내내 같은 화자다(조각마다 model.encode 로 예측하면 조각마다 새로 정해진다).
- 문맥 창(context_s, 2026-10-24): V2 final 을 14 분 회의에 문맥 전부로 디코드하면 앞 3.5 분(약 5,600 위치 — 학습 창 20–90 s 는 2,500 위치 안팎)만
  받아쓰고 나머지는 말을 내지 않았다. 오디오는 자르지 않고 한 스트림으로 계속 디코드하되 LLM 이 보는 문맥만 prefix + 최근 context_s 초로 둔다.
- 태그 이끌기(make_retag, 2026-10-28 — E7): E6 LLM 이 회의 통째에서 헤드의 회의 단위 슬롯을 따르지 않아(96172 귀속 46 pt) 화자 태그를 낼 때 그 뒤 말이 올 오디오의
  활동이 가장 많은 슬롯으로 바꾼다(학습 없이).
- E9 슬롯 블록(long_block_decode, 2026-10-28): 청크마다 [오디오, 태그 D 개]를 한 forward 로 미리 넣고 말이 있는 슬롯에서만 되돌아가 한 토큰씩 낸다
  (realtime.realtime_block_decode — LongThinker.run_all).
"""
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .realtime import _rotate_half

SR, FRAME_S = 16000, 0.08
HOP = int(round(SR * FRAME_S))                                                                # 인코더 프레임 하나 = 1280 샘플


def segment_plan(wav: np.ndarray, K: int, ch: int, core: int = 4800, ctx: int = 1760, tail: int = 32):
    """겹침 조각 인코딩 계획(encode_long·화자 분리 회의 평가 공용) → [(조각 오디오, 조각 프레임 수, 조각 첫 프레임 a, 쓸 몫 첫 프레임 c0, 쓸 몫 길이 n)].
    조각 = 프레임 [c0 − ctx, c0 + core + tail), 쓸 몫 = [c0, c0 + core). core·ctx 는 청크(ch 프레임) 배수, tail ≥ 청크."""
    assert core > 0 and core % ch == 0 and ctx % ch == 0 and tail >= ch, f"core {core}·ctx {ctx} 는 청크 {ch} 프레임 배수, tail {tail} ≥ 청크"
    out = []
    for c0 in range(0, K, core):
        a = max(0, c0 - ctx); b = min(K, c0 + core + tail)
        seg = wav[a * HOP:] if b == K else wav[a * HOP: b * HOP]                              # 끝 조각은 오디오 끝까지(전체 forward 와 같은 끝 처리)
        out.append((seg, b - a, a, c0, min(core, K - c0)))
    return out


@torch.inference_mode()
def encode_long(model, wav: np.ndarray, K: int, core: int = 4800, ctx: int = 1760, tail: int = 32, aosc=None) -> torch.Tensor:
    """wav (N,) float32 16 kHz 세션 전체 → 특징 (S, K, D) — model.encode(wav, N, K)[0] 과 같은 값. 조각 [c0 − ctx, c0 + core + tail) 프레임을 돌려
    [c0, c0 + core) 만 쓴다. core·ctx 는 청크(encoder_right_context + 1 프레임) 배수, tail ≥ 청크. K ≤ core 면 model.encode 한 번과 같은 호출이다.
    aosc(diar.AoscConfig, E7): 화자 분리 헤드 모델 — 스트림 0 은 같고, 스트림 1(예측 슬롯 활동)을 회의 전체 AOSC 스트리밍으로 만든다(model.encode 와 같은 꼴)."""
    ch = int(getattr(model.config, "encoder_right_context", 0) or 0) + 1
    dev = next(model.parameters()).device; out = None
    if aosc is not None:
        assert getattr(model, "diar_head", None) is not None, "aosc 는 화자 분리 헤드(E6·E7) 모델에서만"
        assert not int(getattr(model.config, "act_shift", 0) or 0), "E8 act_shift 모델은 아직 AOSC 와 함께 쓰지 않는다(캐시 점수가 p − R 시점과 어긋난다)"
        from .diar import aosc_stream
        embs, sembs = [], []; sx = getattr(model, "adapter_slotx", None)
        for seg, Ks, a, c0, n in segment_plan(wav, K, ch, core, ctx, tail):
            w = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).to(dev)
            allf = model.encoder(w[None], torch.tensor([len(seg)], device=dev), torch.tensor([Ks], device=dev), taps=tuple(model.config.diar_taps))   # (1, 1 + 탭, Ks, D)
            if out is None: out = allf.new_zeros(1, K, allf.shape[-1])
            out[:, c0: c0 + n] = allf[0, :1, c0 - a: c0 - a + n]; embs.append(model.diar_head.embed(allf)[0, c0 - a: c0 - a + n])
            if sx is not None: sembs.append(sx.embed(allf)[0, c0 - a: c0 - a + n])             # E10: 추출기의 화자 단서 임베딩(헤드를 바꿔 끼워도 그대로)
            del allf
        E_ = torch.cat(embs); act = model.predict_act(aosc_stream(model.diar_head, E_, aosc)).to(out.dtype)          # (K, N) — 임계 config.act_threshold
        streams = [out, F.pad(act, (0, out.shape[-1] - act.shape[-1]))[None]]
        if sx is not None: S_ = torch.cat(sembs); streams.append(F.pad(S_.to(out.dtype), (0, out.shape[-1] - S_.shape[-1]))[None])   # E10: 화자 단서 임베딩 스트림
        return torch.cat(streams, 0)
    for seg, Ks, a, c0, n in segment_plan(wav, K, ch, core, ctx, tail):
        w = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)).to(dev)
        f = model.encode(w[None], torch.tensor([len(seg)], device=dev), torch.tensor([Ks], device=dev))[0]   # (S, b − a, D)
        if out is None: out = f.new_zeros(f.shape[0], K, f.shape[2])
        out[:, c0: c0 + n] = f[:, c0 - a: c0 - a + n]; del f
    return out


class LongThinker:
    """thinker 언어모델 eager 디코드(긴 문맥). run(x (q, D), start) → 다음 토큰 id(차단 −∞ 뒤 argmax) — GraphThinker.run 과 같은 규약(max_len 은 늘어난다).
    used = 지금까지 넣은 위치 수. context(청크 수, 0 = 문맥 전부): 문맥 창 — prefix(시스템·선언·지연, 첫 run 의 입력 = 학습 창의 머리)와 최근 context 청크
    (오디오와 그 사이 말)만 본다. 창 키·쿼리는 실제 위치의 RoPE 그대로, prefix 키만 창 바로 앞 위치로 돌려 붙인다(상대 위치가 학습 창과 같은 꼴 —
    StreamingLLM(arXiv:2309.17453) 의 sink + 창). 청크 경계는 realtime_decode 의 on_chunk 가 알려 준다. context 0 이면 예전 연산 그대로다."""

    def __init__(self, thinker, blocked_ids, max_len: int = 8192, blk: int = 4096, context: int = 0):
        tm = thinker.model; cfg = tm.config
        self.layers, self.norm, self.lm_head, self.embed = tm.layers, tm.norm, thinker.lm_head, tm.embed_tokens.weight
        self.H, self.KV, self.HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        assert self.H % self.KV == 0; self.G, self.blk = self.H // self.KV, int(blk)
        rot = tm.rotary_emb
        assert float(getattr(rot, "attention_scaling", 1.0)) == 1.0
        self.inv_freq = rot.inv_freq.float()
        self.dtype, self.device = self.embed.dtype, self.embed.device
        self.block = torch.zeros(self.lm_head.weight.shape[0], dtype=torch.bool, device=self.device)
        self.block[torch.as_tensor(list(blocked_ids), dtype=torch.long, device=self.device)] = True
        self.C, self.P, self.lo, self.base, self.chunk_pos = int(context), None, None, None, {}   # 창: prefix 길이 · 창 첫 위치 · 버퍼 기준(위치 a ↔ 칸 P + a − base)
        self.cap, self.used, self.k, self.v = 0, 0, None, None
        self._grow(int(max_len))

    def _grow(self, n: int) -> None:
        k = torch.zeros(len(self.layers), self.KV, n, self.HD, device=self.device, dtype=self.dtype); v = torch.zeros_like(k)
        if self.cap: k[:, :, :self.cap] = self.k; v[:, :, :self.cap] = self.v
        self.k, self.v, self.cap = k, v, n

    def on_chunk(self, k: int, pos: int) -> None:
        """청크 k 의 오디오가 pos 에 놓이기 직전: 창이면 청크 k − C + 1 오디오 앞 위치를 내보낸다(창 = 그 오디오부터 — 학습 창처럼 오디오로 시작)."""
        self.chunk_pos[k] = pos
        if self.C and k - self.C + 1 > 0:
            j = k - self.C + 1; self.lo = max(self.lo, self.chunk_pos[j]); self.chunk_pos.pop(j - 1, None)

    def _rope(self, a: int, b: int):
        f = torch.arange(a, b, device=self.device).float()[:, None] * self.inv_freq[None, :]   # GraphThinker 표와 같은 곱
        e = torch.cat((f, f), dim=-1); return e.cos().to(self.dtype)[None], e.sin().to(self.dtype)[None]

    def _attend(self, qs, kw, vw, i0: int, scale: float, kp=None, vp=None):
        """qs (H, q, HD) → (H, q, HD). kw·vw (KV, W, HD) = 창 키(연속), 쿼리 i 는 창 키 [0, i0 + i] 를 본다. kp·vp (KV, P, HD) = 늘 보는 prefix 키(문맥 창).
        머리 h = j·G + g 는 KV 머리 j(HF repeat_kv 와 같은 짝) — 행 g·q + i."""
        q = qs.shape[1]; W = kw.shape[1]; KV, G, HD, B = self.KV, self.G, self.HD, self.blk
        qg = qs.reshape(KV, G * q, HD); P = 0 if kp is None else kp.shape[1]
        nb = W // B; S1 = nb * B; parts = []
        if kp is not None: parts.append(torch.matmul(qg, kp.transpose(-1, -2)))
        if nb: parts.append(torch.matmul(qg[:, None], kw[:, :S1].reshape(KV, nb, B, HD).transpose(-1, -2)).permute(0, 2, 1, 3).reshape(KV, G * q, S1))
        if S1 < W: parts.append(torch.matmul(qg, kw[:, S1:W].transpose(-1, -2)))
        s = (parts[0] if len(parts) == 1 else torch.cat(parts, -1)) * scale                   # (KV, G·q, P + W)
        if q > 1:
            sv = s.view(KV, G, q, P + W)
            for i in range(q - 1): sv[:, :, i, P + i0 + i + 1:] = float("-inf")
        p = torch.softmax(s, -1); o = None
        if kp is not None: o = torch.matmul(p[..., :P], vp)
        if nb:
            ob = torch.matmul(p[..., P:P + S1].reshape(KV, G * q, nb, B).permute(0, 2, 1, 3), vw[:, :S1].reshape(KV, nb, B, HD)).sum(1); o = ob if o is None else o + ob
        if S1 < W:
            r = torch.matmul(p[..., P + S1:], vw[:, S1:W]); o = r if o is None else o + r
        return o.reshape(self.H, q, HD)

    def _slots(self, start: int, S: int):
        """위치 [start, S) 를 쓸 버퍼 칸 → (첫 칸, 창 첫 칸). 문맥 전부면 칸 = 위치. 창이면 넘칠 때 살아 있는 창을 prefix 뒤로 당긴다(모자라면 늘린다)."""
        if not self.C:
            if S > self.cap: self._grow(max(S, self.cap + self.cap // 2))
            return start, 0
        if self.P is None: self.P = self.lo = self.base = S                                    # 첫 입력 = prefix
        if self.P + S - self.base > self.cap:
            a, n = self.P + self.lo - self.base, start - self.lo                                # 살아 있는 창 [lo, start)
            self.k[:, :, self.P:self.P + n] = self.k[:, :, a:a + n].clone(); self.v[:, :, self.P:self.P + n] = self.v[:, :, a:a + n].clone(); self.base = self.lo
            if self.P + S - self.base > self.cap: self._grow(max(self.P + S - self.base, self.cap + self.cap // 2))
        return self.P + start - self.base, self.P + self.lo - self.base

    @torch.inference_mode()
    def hidden(self, x, start: int):
        """x (q, D) 를 위치 start.. 에 넣고 마지막 층 hidden (1, q, D)(norm 전)을 돌려준다."""
        q = x.shape[0]; S = start + q; prefix = self.C and self.P is None
        c0, w0 = self._slots(start, S); c1 = c0 + q
        cos, sin = self._rope(start, S)
        dp = (self.lo - self.P) if self.C and not prefix else 0                                 # prefix 키를 창 바로 앞(lo − P ..)으로
        if dp: cd, sd = self._rope(dp, dp + 1)
        h = x[None]
        for li, layer in enumerate(self.layers):
            a = layer.self_attn
            r = h; hn = layer.input_layernorm(h)
            qs = a.q_norm(a.q_proj(hn).view(1, q, self.H, self.HD)).transpose(1, 2)[0]
            ks = a.k_norm(a.k_proj(hn).view(1, q, self.KV, self.HD)).transpose(1, 2)[0]
            vs = a.v_proj(hn).view(1, q, self.KV, self.HD).transpose(1, 2)[0]
            qs = qs * cos + _rotate_half(qs) * sin
            ks = ks * cos + _rotate_half(ks) * sin
            self.k[li, :, c0:c1] = ks; self.v[li, :, c0:c1] = vs
            if self.C and not prefix:
                kp = self.k[li, :, :self.P]
                if dp: kp = kp * cd + _rotate_half(kp) * sd
                o = self._attend(qs, self.k[li, :, w0:c1], self.v[li, :, w0:c1], c0 - w0, a.scaling, kp, self.v[li, :, :self.P])
            else:
                o = self._attend(qs, self.k[li, :, :c1], self.v[li, :, :c1], start, a.scaling)
            h = r + a.o_proj(o.transpose(0, 1).reshape(1, q, self.H * self.HD))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        self.used = max(self.used, S)
        return h

    def run(self, x, start: int) -> int:
        logits = self.lm_head(self.norm(self.hidden(x, start)[:, -1]))[0].float()
        return int(logits.masked_fill(self.block, float("-inf")).argmax())

    def run_all(self, x, start: int) -> list:
        """run 과 같고 위치마다 다음 토큰 id 목록(E9 추측 prefill — realtime.realtime_block_decode)."""
        logits = self.lm_head(self.norm(self.hidden(x, start)))[0].float()
        return logits.masked_fill(self.block, float("-inf")).argmax(-1).tolist()


def make_retag(act, slot_ids, delta: int, back: int = 2, stats: Optional[dict] = None):
    """E7 회의 통째(2026-10-28 — LLM 태그가 헤드의 회의 단위 슬롯을 따르지 않아 귀속이 무너졌다): 모델이 화자 태그를 내면, 청크 k 에서 그 뒤 말이 올 오디오
    [k − δ − back, k] 의 활동(act (K, N) 0/1 — encode_long(aosc) 스트림 1)이 가장 많은 슬롯 태그로 바꾼다. 활동이 없거나 모델이 고른 슬롯도 그 구간에서
    말하고 있으면(겹침) 그대로. stats(dict)에 tags·changed 를 센다."""
    ids = [int(x) for x in slot_ids]; A = act.float().cpu(); K, N = A.shape
    if stats is not None: stats.update(tags=0, changed=0)
    def retag(k, tid):
        if int(tid) not in ids: return tid
        if stats is not None: stats["tags"] += 1
        lo, hi = max(0, int(k) - int(delta) - back), min(K, int(k) + 1)
        if hi <= lo: return tid
        c = A[lo:hi, : min(N, len(ids))].sum(0)
        cur = ids.index(int(tid))
        if float(c.max()) <= 0 or (cur < len(c) and float(c[cur]) > 0): return tid
        if stats is not None: stats["changed"] += 1
        return ids[int(c.argmax())]
    return retag


def long_decode(model, feats, prefix_ids, max_flush_rounds=None, blk: int = 4096, context_s: float = 0.0, retag=None):
    """feats (S, K, D)(model.encode·encode_long 출력) → (emitted [(k, tid)], forced, flush_rounds, 넣은 위치 수). KV 는 prefix + 2.6·K + 1024 위치로 시작한다
    (청크마다 오디오·<NEXT_AUDIO> 두 위치 + 말 — AMI 회의는 2.3·K 안팎). context_s > 0 = 문맥 창(최근 그 초의 청크만, LongThinker) — KV 는 창 크기로.
    retag: make_retag(…)(E7 — 화자 태그를 활동으로 이끈다, 없으면 예전 그대로)."""
    from .realtime import realtime_decode
    K = int(feats.shape[-2]); C = int(round(context_s / FRAME_S)) if context_s else 0
    lt = LongThinker(model.thinker, model.config.blocked_ids, max_len=len(prefix_ids) + (4 * C + 2048 if C else int(2.6 * K) + 1024), blk=blk, context=C)
    out, forced, _, rounds, _ = realtime_decode(model, lt, feats, list(prefix_ids), max_flush_rounds=max_flush_rounds, on_chunk=lt.on_chunk if C else None, retag=retag)
    return out, forced, rounds, lt.used


def long_block_decode(model, feats, prefix_ids, D: int, act=None, max_flush_rounds=None, blk: int = 4096, context_s: float = 0.0, mix_off: bool = False):
    """E9 회의 통째 디코드(2026-10-28): realtime.realtime_block_decode 에 LongThinker(추측 prefill·되돌리기 — 출력 규칙은 batch_decode.decode_blocks_batch 와 같다).
    feats (S, K, D)(encode_long — aosc 면 스트림 1 = 회의 단위 예측 활동이 태그·오디오 활동이 된다), D = 선언 슬롯 수.
    KV 는 prefix + (D + 2.6)·K + 1024 위치로 시작한다(청크마다 오디오 + 태그 D 개 + 말). context_s > 0 = 문맥 창(LongThinker).
    → (emitted, forced, flush_rounds, 넣은 위치 수, forward 수, 청크 ms 목록)."""
    from .realtime import realtime_block_decode
    K = int(feats.shape[-2]); C = int(round(context_s / FRAME_S)) if context_s else 0
    lt = LongThinker(model.thinker, model.config.blocked_ids, max_len=len(prefix_ids) + ((D + 4) * C + 2048 if C else int((D + 2.6) * K) + 1024), blk=blk, context=C)
    out, forced, ticks, rounds, _, nf = realtime_block_decode(model, lt, feats, list(prefix_ids), D, act=act, max_flush_rounds=max_flush_rounds,
                                                              on_chunk=lt.on_chunk if C else None, mix_off=mix_off)
    return out, forced, rounds, lt.used, nf, ticks
