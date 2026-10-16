"""Adapter + MlxThinker (thinker-spec §1, §2, §4.4, §6).

- 고정 KV (층마다 (1, kv_heads, max_len, head_dim)), 쓰기 포인터는 호출자가 관리 — 되감기 = 다음 forward 의 start 를 작게.
- RoPE 는 HF 와 같은 식의 cos/sin 표를 numpy fp32 로 한 번 만든다 (nn.RoPE(base=) 는 위치 1024+ 에서 5e-4 오차).
- q/k RMSNorm 은 RoPE 전, GQA, scale 1/√head_dim, SwiGLU, tied lm_head (151 936 행 전부).
- logits 는 fp32 로 바꾼 뒤 차단 마스크(−inf)·argmax 를 그래프 안에서, 동기화는 forward 당 한 번.
- 저정밀(fp16/bf16/q8): 잔차 fp32 옵션, 상위 k 후보를 fp32 임베딩으로 다시 채점, 유한성 검사(보고만).
- Draft: 같은 thinker 의 q4 자기 초안 (KV 공유, 텍스트 토큰 투기 디코드용). TwoStageHead: fp32 lm_head 의 인증서 두 단계 헤드."""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

F32 = mx.float32
# 이름 → (가중치 dtype, 잔차 fp32, 재채점 k, 양자화 비트)
PRECISIONS = {
    "fp32": (mx.float32, True, 0, 0),
    "fp16": (mx.float16, True, 8, 0),          # 기본 후보: fp16 가중치 + fp32 잔차/노름 + top-8 fp32 재채점
    "fp16-plain": (mx.float16, False, 0, 0),   # 비교용: 전부 fp16
    "bf16": (mx.bfloat16, True, 8, 0),
    "q8": (mx.float16, True, 8, 8),            # 보고용(정확하지 않음): 선형층·lm_head 8비트, 입력 임베딩은 원본
}


class Adapter:
    """LayerNorm(eps 1e-5) → Linear → GELU(erf) → Linear, 항상 fp32 (thinker-spec §1)."""

    def __init__(self, w: dict):
        self.w = {k: v.astype(F32) for k, v in w.items()}
        mx.eval(self.w)

    def __call__(self, x: mx.array) -> mx.array:
        W = self.w
        h = mx.fast.layer_norm(x.astype(F32), W["net.0.weight"], W["net.0.bias"], 1e-5)
        h = nn.gelu(h @ W["net.1.weight"].T + W["net.1.bias"])
        return h @ W["net.3.weight"].T + W["net.3.bias"]


def rope_table(n: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """HF 식(inv = 1/θ^(2i/d), 각 = pos·inv, emb = cat(각, 각))을 numpy fp32 로 (torch 표와 ≤ 1 ulp)."""
    inv = (np.float32(1.0) / (np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
                                                    / np.float32(head_dim)))).astype(np.float32)
    f = np.arange(n, dtype=np.float32)[:, None] * inv[None]
    e = np.concatenate([f, f], axis=-1)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)


def _rope_apply(x, cos, sin):
    """x·cos + rotate_half(x)·sin (HF apply_rotary_pos_emb)."""
    h = x.shape[-1] // 2
    return x * cos + mx.concatenate([-x[..., h:], x[..., :h]], axis=-1) * sin


_rope_c = mx.compile(_rope_apply)   # 모양별로 따로 추적된다 (작은 q 에서만 사용)


class _Lin:
    """y = x Wᵀ (가중치 (out, in)), 선택적으로 group-64 양자화."""
    __slots__ = ("w", "q", "bits")

    def __init__(self, w: mx.array, dtype, bits: int = 0):
        self.bits = bits
        if bits:
            self.w, self.q = None, mx.quantize(w.astype(dtype), group_size=64, bits=bits)
        else:
            self.w, self.q = w.astype(dtype), None

    def __call__(self, x):
        if self.q is None:
            return x @ self.w.T
        wq, s, b = self.q
        return mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=self.bits)

    def arrays(self):
        return [self.w] if self.q is None else list(self.q)


class MlxThinker:
    """Qwen3-ASR thinker. w: 'thinker.' 를 뗀 키 (model.embed_tokens.weight, model.layers.N...). w 의 항목은 소비(pop)된다."""

    def __init__(self, w: dict, text_cfg: dict, blocked_ids: list[int], dtype=mx.float16, max_len: int = 2048,
                 resid_fp32: bool = True, rescore_k: int = 8, quant_bits: int = 0):
        c = text_cfg
        self.n_layers, self.D = c["num_hidden_layers"], c["hidden_size"]
        D = self.D
        self.H, self.KVH = c["num_attention_heads"], c["num_key_value_heads"]
        self.HD = c.get("head_dim") or D // self.H
        self.F, self.V, self.eps = c["intermediate_size"], c["vocab_size"], float(c["rms_norm_eps"])
        self.scale = self.HD ** -0.5
        self.dtype, self.max_len, self.quant_bits = dtype, int(max_len), quant_bits
        self.resid_fp32 = bool(resid_fp32) or dtype == F32
        self.hdt = F32 if self.resid_fp32 else dtype              # 잔차 dtype
        self.rescore_k = int(rescore_k) if (dtype != F32 or quant_bits) else 0
        ndt = F32 if self.resid_fp32 else dtype                    # 노름 가중치 dtype
        E = w.pop("model.embed_tokens.weight", None)
        lm = w.pop("lm_head.weight", None)
        if E is None:                                              # 다른 별칭만 저장된 경우 (thinker-spec §2.5)
            E = lm
        elif lm is not None and not mx.array_equal(E, lm).item():
            raise RuntimeError("lm_head.weight 가 embed_tokens 와 다릅니다 (tied 가정 위반)")
        if tuple(E.shape) != (self.V, D):
            raise RuntimeError(f"임베딩 모양 {E.shape} ≠ ({self.V}, {D})")
        self.E = E.astype(dtype)                                   # lm_head (와 저정밀 입력 임베딩)
        # fp32 임베딩: 재채점 후보 행 + (잔차 fp32 일 때) 입력 토큰 행을 정확히
        self.E32 = self.E if dtype == F32 else (E.astype(F32) if self.rescore_k else None)
        self.in_dtype = self.hdt
        self.Eq = mx.quantize(self.E, group_size=64, bits=quant_bits) if quant_bits else None
        mx.eval([a for a in (self.E, self.E32) if a is not None] + (list(self.Eq) if self.Eq else []))
        del E, lm
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            g = lambda n: w.pop(p + n)
            L = dict(q=_Lin(g("self_attn.q_proj.weight"), dtype, quant_bits), k=_Lin(g("self_attn.k_proj.weight"), dtype, quant_bits),
                     v=_Lin(g("self_attn.v_proj.weight"), dtype, quant_bits), o=_Lin(g("self_attn.o_proj.weight"), dtype, quant_bits),
                     gate=_Lin(g("mlp.gate_proj.weight"), dtype, quant_bits), up=_Lin(g("mlp.up_proj.weight"), dtype, quant_bits),
                     down=_Lin(g("mlp.down_proj.weight"), dtype, quant_bits),
                     n1=g("input_layernorm.weight").astype(ndt), n2=g("post_attention_layernorm.weight").astype(ndt),
                     qn=g("self_attn.q_norm.weight").astype(ndt), kn=g("self_attn.k_norm.weight").astype(ndt))
            mx.eval([a for k in ("q", "k", "v", "o", "gate", "up", "down") for a in L[k].arrays()] + [L["n1"], L["n2"], L["qn"], L["kn"]])
            self.layers.append(L)
        self.norm = w.pop("model.norm.weight").astype(ndt)
        ba = np.zeros(self.V, np.float32)
        ba[list(blocked_ids)] = -np.inf
        self.block = mx.array(ba)
        cos, sin = rope_table(self.max_len, self.HD, float(c["rope_theta"]))
        self.cos, self.sin = mx.array(cos), mx.array(sin)
        self.kv = self.new_kv(self.max_len)
        mx.eval(self.norm, self.block, self.cos, self.sin)
        self.nonfinite = 0                                         # 유한성 검사 실패 횟수 (보고용)
        self.ts: TwoStageHead | None = None                        # 두 단계 헤드 (enable_two_stage, fp32 전용)
        self.ts_stats = dict(calls=0, rows=0, fallback_rows=0)

    # ── 공개 API
    def new_kv(self, length: int) -> tuple[list, list]:
        """별도 KV 버퍼 (final 재디코드 등). 길이 ≤ max_len."""
        n = min(int(length), self.max_len)
        K = [mx.zeros((1, self.KVH, n, self.HD), self.dtype) for _ in range(self.n_layers)]
        V = [mx.zeros((1, self.KVH, n, self.HD), self.dtype) for _ in range(self.n_layers)]
        mx.eval(K, V)
        return K, V

    def embed(self, ids: list[int]) -> mx.array:
        """(n, D) 입력 행. 잔차 fp32 면 fp32 임베딩(정확), 아니면 가중치 dtype."""
        return self.embed_lazy(mx.array([int(i) for i in ids], dtype=mx.int32))

    def embed_lazy(self, ids: mx.array) -> mx.array:
        """embed 와 같은 행, id 가 기기 위 int32 배열일 때 (동기화 없이 초안 체인을 잇는다)."""
        E = self.E32 if (self.resid_fp32 and self.E32 is not None) else self.E
        return E[ids].astype(self.in_dtype)

    def kv_rows(self, n: int) -> tuple[list, list]:
        """스트리밍 KV 의 행 0..n-1 깊은 복사 (prefix 캐시용). mx.contiguous 는 새 버퍼 — KV 버퍼 뷰를 남기지 않는다
        (뷰가 남으면 버퍼를 붙잡고, 이후 KV 쓰기가 제자리 갱신 대신 버퍼 전체 복사가 된다)."""
        Kc, Vc = self.kv
        ks = [mx.contiguous(K[..., :n, :]) for K in Kc]
        vs = [mx.contiguous(V[..., :n, :]) for V in Vc]
        mx.eval(ks, vs)
        return ks, vs

    def set_kv_rows(self, rows: tuple[list, list]) -> int:
        """kv_rows 로 떠 둔 행을 스트리밍 KV 의 0..n-1 에 되돌린다 → n. 같은 입력의 같은 q=P forward 출력이라 비트 동일."""
        ks, vs = rows
        Kc, Vc = self.kv
        n = int(ks[0].shape[2])
        for i in range(len(Kc)):
            Kc[i][..., :n, :] = ks[i]
            Vc[i][..., :n, :] = vs[i]
        mx.eval(Kc, Vc)
        return n

    def make_draft(self, bits: int = 4) -> "Draft | None":
        """q4 자기 초안 (선형층·lm_head 양자화 사본, KV·노름·RoPE 공유). q8 처럼 원본 가중치가 없으면 None."""
        if self.quant_bits or any(L["q"].w is None for L in self.layers):
            return None
        return Draft(self, bits)

    def enable_two_stage(self, bits: int = 8, M: int = 64, tau: float = 1e-3) -> bool:
        """fp32 thinker 에만: lm_head 를 두 단계(q8 근사 → 상한 상위 M → fp32 재채점 + 인증서)로. → 켰는지."""
        if self.dtype != F32 or self.rescore_k or self.quant_bits:
            return False
        self.ts = TwoStageHead(self.E, bits=bits, M=M, tau=tau)
        return True

    def forward(self, x: mx.array, start: int, sel: list[int] | None = None, block: mx.array | None = None,
                kv: tuple | None = None) -> list[int]:
        """x (q, D) 를 위치 start..start+q-1 에 쓰고, sel 행(기본 [q-1])의 마스크 argmax id. sel=[] 면 KV 만 채운다."""
        ids, _ = self._run(x, start, sel, block, kv, want_logits=False)
        return ids

    def forward_logits(self, x: mx.array, start: int, sel: list[int] | None = None, block: mx.array | None = None,
                       kv: tuple | None = None) -> np.ndarray:
        """검증용: sel 행의 fp32 logits (차단 −inf 포함; 재채점 후보 자리는 fp32 재채점 값)."""
        return self.forward_ids_logits(x, start, sel, block, kv)[1]

    def forward_ids_logits(self, x, start, sel=None, block=None, kv=None) -> tuple[list[int], np.ndarray]:
        return self._run(x, start, sel, block, kv, want_logits=True)

    # ── 내부
    def _row_add(self, start: int, sel: list[int], block: mx.array) -> mx.array:
        """logits 에 더할 값: (V,) 또는 (len(sel), V). 테스트에서 덮어써 방출 일정을 강제한다."""
        return block

    def _run(self, x, start, sel, block, kv, want_logits):
        q = int(x.shape[0])
        sel = [q - 1] if sel is None else [int(s) for s in sel]
        Kc, Vc = kv if kv is not None else self.kv
        cap = int(Kc[0].shape[2])
        if start < 0 or start + q > cap:
            raise ValueError(f"KV 범위 밖: start {start} + q {q} > {cap}")
        h = self._layers(x, start, q, Kc, Vc)
        if not sel:
            mx.eval(Kc, Vc)
            return [], None
        hs = h[-1:] if sel == [q - 1] else h[mx.array(sel, dtype=mx.int32)]
        hn = mx.fast.rms_norm(hs, self.norm, self.eps)
        add = self._row_add(start, sel, self.block if block is None else block)
        fin = mx.all(mx.isfinite(hn)).astype(mx.int32).reshape(1)
        if self.ts is not None and not want_logits:
            return self._run_two_stage(hn, add, fin), None
        ids, logits, extra = self._head(hn, add)
        packed = mx.concatenate([ids.astype(mx.int32), fin])
        if want_logits:
            mx.eval([packed, logits] + extra)
        res = packed.tolist()                                      # 동기화 한 번
        if not res[-1]:
            self.nonfinite += 1
        if not want_logits:
            return res[:-1], None
        L = np.array(logits)
        if extra:                                                  # 재채점한 후보 자리를 fp32 값으로
            c, e = np.array(extra[0]), np.array(extra[1])
            np.put_along_axis(L, c, e, axis=-1)
        return res[:-1], L

    def _head(self, hn, add):
        """현재(전체) lm_head: logits = hn·Eᵀ (+ 차단) → argmax (저정밀이면 상위 k 를 fp32 로 재채점). → (ids, logits, extra)."""
        lo = hn.astype(self.dtype)
        if self.Eq is not None:
            wq, s, b = self.Eq
            logits = mx.quantized_matmul(lo, wq, s, b, transpose=True, group_size=64, bits=self.quant_bits)
        else:
            logits = lo @ self.E.T
        logits = logits.astype(F32) + add
        extra = []
        if self.rescore_k:
            k = self.rescore_k
            cand = mx.sort(mx.argpartition(-logits, kth=k - 1, axis=-1)[:, :k], axis=-1)   # 오름차순: 동점은 낮은 id
            a = add[cand] if add.ndim == 1 else mx.take_along_axis(add, cand, axis=-1)
            exact = (self.E32[cand] @ hn.astype(F32)[:, :, None])[..., 0] + a
            ids = mx.take_along_axis(cand, mx.argmax(exact, axis=-1)[:, None], axis=-1)[:, 0]
            extra = [cand, exact]
        else:
            ids = mx.argmax(logits, axis=-1)
        return ids, logits, extra

    def _run_two_stage(self, hn, add, fin) -> list[int]:
        """두 단계 헤드: id·인증 통과·유한성을 동기화 한 번에. 인증 실패 행이 있으면 현재 헤드를 같은 행 전부(같은 모양)로
        다시 계산해 그 행만 바꾼다 (행 수가 다르면 matmul 반올림이 달라질 수 있어 부분 행으로 하지 않는다) — 동기화 한 번 더."""
        r = int(hn.shape[0])
        bid, ok = self.ts.lazy(hn, add)
        res = mx.concatenate([bid.astype(mx.int32), ok.astype(mx.int32), fin]).tolist()
        ids, oks = res[:r], res[r:2 * r]
        if not res[-1]:
            self.nonfinite += 1
        st = self.ts_stats
        st["calls"] += 1
        st["rows"] += r
        if not all(oks):
            full = self._head(hn, add)[0].tolist()
            st["fallback_rows"] += sum(1 for o in oks if not o)
            ids = [ids[i] if oks[i] else full[i] for i in range(r)]
        return ids

    def _layers(self, x, start, q, Kc, Vc):
        dt, H, KVH, HD, eps = self.dtype, self.H, self.KVH, self.HD, self.eps
        end = start + q
        cos, sin = self.cos[start:end], self.sin[start:end]
        rope = _rope_c if q <= 16 else _rope_apply
        mask = None if q == 1 else "causal"                        # [:end] 슬라이스와 함께면 아래-오른쪽 정렬 causal
        h = x.astype(self.hdt)[None]
        for i, L in enumerate(self.layers):
            hn = mx.fast.rms_norm(h, L["n1"], eps).astype(dt)
            qs = mx.fast.rms_norm(L["q"](hn).reshape(1, q, H, HD), L["qn"], eps).transpose(0, 2, 1, 3)
            ks = mx.fast.rms_norm(L["k"](hn).reshape(1, q, KVH, HD), L["kn"], eps).transpose(0, 2, 1, 3)
            vs = L["v"](hn).reshape(1, q, KVH, HD).transpose(0, 2, 1, 3)
            qs = rope(qs, cos, sin).astype(dt)
            ks = rope(ks, cos, sin).astype(dt)
            Kc[i][..., start:end, :] = ks
            Vc[i][..., start:end, :] = vs.astype(dt)
            o = mx.fast.scaled_dot_product_attention(qs, Kc[i][..., :end, :], Vc[i][..., :end, :], scale=self.scale, mask=mask)
            h = h + L["o"](o.transpose(0, 2, 1, 3).reshape(1, q, H * HD))
            hn = mx.fast.rms_norm(h, L["n2"], eps).astype(dt)
            h = h + L["down"](nn.silu(L["gate"](hn)) * L["up"](hn))
        return h[0]


LINS = ("q", "k", "v", "o", "gate", "up", "down")


class Draft:
    """q4 자기 초안 (텍스트 토큰 투기 디코드, decoder.SegmentDecoder): 목표 thinker 의 선형층·lm_head 를 group-64 양자화한 사본.
    노름·RoPE 표·KV 버퍼는 목표 것을 그대로 쓴다. 초안이 KV 행 pos.. 에 쓴 근사값은 검증 forward 가 정확한 값으로 덮어쓰고,
    덮이지 않은 행은 아무도 읽지 않는다 (모든 forward 는 [start:end] 를 먼저 쓰고 인과 마스크로 [:end] 만 본다).
    초안 id 는 제안일 뿐 — 내는 토큰은 언제나 목표 forward 의 argmax 다."""
    _layers = MlxThinker._layers

    def __init__(self, th: MlxThinker, bits: int = 4):
        self.th, self.bits = th, int(bits)
        for a in ("dtype", "H", "KVH", "HD", "eps", "cos", "sin", "hdt", "scale"):
            setattr(self, a, getattr(th, a))
        self.layers = []
        for L in th.layers:
            d = {k: _Lin(L[k].w, th.dtype, self.bits) for k in LINS}
            d.update({k: L[k] for k in ("n1", "n2", "qn", "kn")})
            mx.eval([a for k in LINS for a in d[k].arrays()])
            self.layers.append(d)
        self.Eq = mx.quantize(th.E, group_size=64, bits=self.bits)
        mx.eval(list(self.Eq))
        self.nbytes = sum(a.nbytes for d in self.layers for k in LINS for a in d[k].arrays()) + sum(a.nbytes for a in self.Eq)

    def ids_lazy(self, x: mx.array, start: int) -> mx.array:
        """x (1, D) 를 스트리밍 KV 의 start 에 쓰고 (1,) int32 argmax (차단 마스크 포함) — 동기화 없음."""
        th = self.th
        Kc, Vc = th.kv
        h = self._layers(x, start, int(x.shape[0]), Kc, Vc)
        hn = mx.fast.rms_norm(h[-1:], th.norm, th.eps)
        wq, s, b = self.Eq
        lg = mx.quantized_matmul(hn.astype(self.dtype), wq, s, b, transpose=True, group_size=64, bits=self.bits)
        return mx.argmax(lg.astype(F32) + th.block, axis=-1).astype(mx.int32)


class TwoStageHead:
    """fp32 lm_head 의 두 단계 계산 (perf F20). 근사 s̃ = hn·Ê (q8 group-64, fp32 스케일·입력), 상한
    ub_v = s̃_v + ‖hn‖₂·r_v + τ + 1e-6·|s̃_v| + add_v  (r_v = ‖E_v − Ê_v‖₂ 를 적재 때 조금 부풀려 계산: |hn·(E_v − Ê_v)| ≤ ‖hn‖·r_v).
    후보 S = ub 상위 M, u_out = (M+1) 번째 ub. 정확 e_S = E[S]·hn + add (fp32). 인증: e_best − max(e_2nd, u_out) > τ 이면
    S 밖의 어떤 id 도 e_best 를 넘을 수 없으므로 전체 argmax = best (fp32 반올림 ≪ τ 가정, torch↔MLX Δm 실측 1.4e-4).
    실패하면 호출자가 현재 헤드로 다시 계산한다."""

    def __init__(self, E: mx.array, bits: int = 8, M: int = 64, tau: float = 1e-3, group: int = 64):
        self.E, self.bits, self.M, self.tau, self.group = E, int(bits), int(M), float(tau), int(group)
        self.q = mx.quantize(E, group_size=self.group, bits=self.bits)
        mx.eval(list(self.q))
        r = []
        for i in range(0, E.shape[0], 16384):                     # r_v = ‖E_v − 역양자화‖₂ (조각별로: 일시 메모리↓)
            dq = mx.dequantize(*[a[i:i + 16384] for a in self.q], group_size=self.group, bits=self.bits)
            r.append(mx.sqrt(mx.sum(mx.square(E[i:i + 16384] - dq), axis=-1)))
            mx.eval(r[-1])
        self.r = (mx.concatenate(r) * (1 + 1e-4) + 1e-7).astype(F32)
        mx.eval(self.r)
        self.nbytes = sum(a.nbytes for a in self.q) + self.r.nbytes

    def lazy(self, hn: mx.array, add: mx.array) -> tuple[mx.array, mx.array]:
        """hn (r, D), add (V,) 또는 (r, V) → (best id (r,), 인증 통과 (r,) bool) — 동기화 없음."""
        M, tau = self.M, self.tau
        h32 = hn.astype(F32)
        wq, s, b = self.q
        sa = mx.quantized_matmul(h32, wq, s, b, transpose=True, group_size=self.group, bits=self.bits)
        hnorm = mx.sqrt(mx.sum(mx.square(h32), axis=-1, keepdims=True))
        ub = sa + hnorm * self.r + (tau + 1e-6 * mx.abs(sa)) + add             # 차단 id 는 −inf
        part = mx.argpartition(-ub, kth=M, axis=-1)
        S = mx.sort(part[:, :M], axis=-1)                                        # 오름차순 (동점이면 인증 실패)
        u_out = mx.take_along_axis(ub, part[:, M:M + 1], axis=-1)[:, 0]
        aS = add[S] if add.ndim == 1 else mx.take_along_axis(add, S, axis=-1)
        e = (self.E[S] @ h32[:, :, None])[..., 0] + aS
        jb = mx.argmax(e, axis=-1)
        eb = mx.take_along_axis(e, jb[:, None], axis=-1)[:, 0]
        e2 = mx.max(mx.where(mx.arange(M)[None] == jb[:, None], -mx.inf, e), axis=-1)
        bid = mx.take_along_axis(S, jb[:, None], axis=-1)[:, 0]
        ok = (eb - mx.maximum(e2, u_out)) > tau
        return bid, ok
