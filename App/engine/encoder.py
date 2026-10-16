"""Nemotron FastConformer 캐시 인식 스트리밍 인코더 (MLX, encoder-spec §2–§8).

- 서브샘플링: causal dw_striding ×8 (항상 fp32).
- 24 × ConformerLayer: FF½ → rel-pos MHSA (투영된 K/V 를 C=56 프레임 캐시) → causal conv k9 (8 프레임 캐시) → FF½ → LN.
- 청크 [C,R] = [56,3]: 첫 스텝 mel 25 행 → 4 프레임, 이후 [9 캐시 | 32 새] = 41 행 → 6 행 중 앞 2 행 버림 → 4 프레임.
  일반 규칙 s = max(0, idx-9), drop = emitted - s/8 (첫 스텝 0, 정상 상태 2). [56,0] 도 같은 규칙(정확 규칙, NeMo 와 다름).
- 정밀도: dtype=float32 (기본, 기준과 1e-5 일치) 또는 float16/bfloat16 = 큰 Linear 가중치만 저정밀,
  잔차·LayerNorm·어텐션(점수/softmax)·GLU·depthwise conv·서브샘플링은 fp32.
- 정상 스텝(첫 스텝 / 정상 상태 모양)은 mx.compile, 발화 끝의 짧은 스텝은 eager.
"""
from __future__ import annotations

import math
import time
import weakref

import numpy as np
import mlx.core as mx

from .frontend import HOP, NMEL, LogMel, MelStream, check_preprocessor_cfg

LN_EPS = 1e-5
MASK_NEG = -1e9            # 빈 캐시 슬롯용 가산 마스크 (NeMo 는 -10000 후 확률 0 → 결과 동일)
FRAME_HZ = 12.5            # config.frame_hz (= 16000 / (160·8))


def k_frames(n_samples: int, frame_hz: float = FRAME_HZ) -> int:
    """eval 과 같은 K 식 (eval_single_turn_asr.py:431). round(n/1280) 과 반올림 경계에서 다르다."""
    return int(round(n_samples / 16000 * frame_hz))


def _sub_len(L: int) -> int:
    """causal stride-2 conv 한 층 (k3, 패딩 2+1) 의 출력 길이."""
    return L // 2 + 1


def _ln(x, wb):
    return mx.fast.layer_norm(x, wb[0], wb[1], LN_EPS)


def _silu(x):
    return x * mx.sigmoid(x)


def _pe(rels: np.ndarray, d: int) -> np.ndarray:
    """NeMo RelPositionalEncoding 의 사인 표 (상대 위치 rels) → (R, d) fp32."""
    pos = rels.astype(np.float32)[:, None]
    div = np.exp(np.arange(0, d, 2, dtype=np.float32) * np.float32(-(math.log(10000.0) / d)))
    pe = np.zeros((len(rels), d), np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def _check_encoder_cfg(ec: dict) -> None:
    want = dict(subsampling="dw_striding", subsampling_factor=8, causal_downsampling=True, self_attention_model="rel_pos",
                xscaling=False, untie_biases=True, conv_norm_type="layer_norm", conv_context_size="causal",
                att_context_style="chunked_limited", use_bias=False, feat_out=-1, reduction=None, feat_in=NMEL)
    bad = {k: (ec.get(k), v) for k, v in want.items() if k in ec and ec.get(k) != v}
    if bad:
        raise ValueError(f"지원하지 않는 encoder 설정: {bad}")


class _Layer:
    __slots__ = ("n_ff1", "n_att", "n_conv", "n_cbn", "n_ff2", "n_out", "ff1_1", "ff1_2", "ff2_1", "ff2_2",
                 "wqkv", "wo", "pbu", "pbv", "P_t", "pw1", "dw", "pw2")


class EncoderModel:
    """체크포인트 가중치(앞의 "encoder." 제거: "pre.featurizer.fb", "pre.featurizer.window", "enc.<NeMo 키>") → MLX 인코더.
    weights 는 mx.load 의 지연 배열을 그대로 넘기는 것이 좋다 (필요한 텐서만 읽고 변환본만 남김). 생성 뒤 dict 는 버려도 된다.
    consume=True: weights 를 비우고 텐서를 하나씩 꺼내 쓴다 — 레이어마다 실체화한 뒤 fp32 원본이 바로 풀려, 저정밀 생성 중
    fp32 원본 전체(≈ 2.4 GB)가 한꺼번에 상주하지 않는다 (Engine 적재 최대 메모리↓)."""

    def __init__(self, weights: dict, cfg: dict, right_context: int = 3, dtype=mx.float32, consume: bool = False):
        ec = cfg["encoder"]
        check_preprocessor_cfg(cfg.get("preprocessor", {}))
        _check_encoder_cfg(ec)
        ctxs = ec.get("att_context_size", [[56, right_context]])
        ctxs = [list(ctxs)] if ctxs and not isinstance(ctxs[0], (list, tuple)) else [list(c) for c in ctxs]
        lefts = [c[0] for c in ctxs if c[1] == right_context]
        if not lefts:
            raise ValueError(f"att_context_size {ctxs} 에 right={right_context} 가 없다")
        self.C, self.R = int(lefts[0]), int(right_context)
        self.Tm = self.R + 1                                  # 스텝당 출력 프레임 (valid_out_len)
        assert self.C % self.Tm == 0, "청크 경계가 캐시 크기와 맞아야 한다"
        self.D, self.H, self.NL = int(ec["d_model"]), int(ec["n_heads"]), int(ec["n_layers"])
        self.DK = self.D // self.H
        self.KC = int(ec.get("conv_kernel_size", 9))
        self.SUB_CH = int(ec.get("subsampling_conv_channels", 256))
        self.SUB_F = _sub_len(_sub_len(_sub_len(NMEL)))       # 128 → 65 → 33 → 17
        self.sub = 8
        self.first_size = 1 + self.sub * self.R               # 첫 스텝 새 mel 프레임 (25)
        self.size = self.sub * self.Tm                        # 이후 새 mel 프레임 (32)
        self.pre_cache = self.sub + 1                         # pre-encode 캐시 9 프레임
        self.dtype = dtype
        self.right_context = self.R
        self.compile = True                                   # False 면 모든 스텝 eager (측정용)
        self.probe = None                                     # 검증용: probe(name, layer, array) — 설정 시 모든 스텝 eager

        w = {(k[8:] if k.startswith("encoder.") else k): v for k, v in weights.items()}
        if consume:
            weights.clear()
        f32 = mx.float32
        get = w.pop if consume else w.__getitem__                                             # 키마다 한 번만 쓴다

        def A(k):
            v = get(k)
            return (mx.array(v) if isinstance(v, np.ndarray) else v).astype(f32)
        C = lambda a: a.astype(dtype)                                                         # noqa: E731
        self.window = np.array(A("pre.featurizer.window")).reshape(-1)
        self.fb = np.array(A("pre.featurizer.fb")).reshape(NMEL, -1)
        self.logmel = LogMel(self.window, self.fb)

        p = "enc.pre_encode.conv."
        self.c0w, self.c0b = A(p + "0.weight").transpose(0, 2, 3, 1), A(p + "0.bias")        # (256,3,3,1) OHWI
        self.subw = []
        for dw_i, pw_i in ((2, 3), (5, 6)):
            self.subw.append((A(f"{p}{dw_i}.weight").transpose(0, 2, 3, 1), A(f"{p}{dw_i}.bias"),
                              A(f"{p}{pw_i}.weight")[:, :, 0, 0], A(f"{p}{pw_i}.bias")))
        ow = A("enc.pre_encode.out.weight")                                                    # 열 = c*17+f
        self.out_w = ow.reshape(self.D, self.SUB_CH, self.SUB_F).transpose(0, 2, 1).reshape(self.D, -1)   # 열 = f*256+c
        self.out_b = A("enc.pre_encode.out.bias")

        # 상대 위치 표: 행 m ↔ rel = C+Tm-1-m (59 … -3)
        rels = (self.C + self.Tm - 1) - np.arange(self.C + 2 * self.Tm - 1)
        pe = mx.array(_pe(rels, self.D))
        self.layers = []
        for i in range(self.NL):
            q = f"enc.layers.{i}."
            L = _Layer()
            ln = lambda n: (A(q + n + ".weight"), A(q + n + ".bias"))   # noqa: E731
            L.n_ff1, L.n_att, L.n_conv = ln("norm_feed_forward1"), ln("norm_self_att"), ln("norm_conv")
            L.n_cbn, L.n_ff2, L.n_out = ln("conv.batch_norm"), ln("norm_feed_forward2"), ln("norm_out")
            L.ff1_1, L.ff1_2 = C(A(q + "feed_forward1.linear1.weight")), C(A(q + "feed_forward1.linear2.weight"))
            L.ff2_1, L.ff2_2 = C(A(q + "feed_forward2.linear1.weight")), C(A(q + "feed_forward2.linear2.weight"))
            L.wqkv = C(mx.concatenate([A(q + f"self_attn.linear_{n}.weight") for n in ("q", "k", "v")], axis=0))   # (3D,D)
            L.wo = C(A(q + "self_attn.linear_out.weight"))
            L.pbu = A(q + "self_attn.pos_bias_u")[:, None, :]                                 # (H,1,DK)
            L.pbv = A(q + "self_attn.pos_bias_v")[:, None, :]
            P = pe @ A(q + "self_attn.linear_pos.weight").T                                     # (R,D) 상수
            L.P_t = P.reshape(-1, self.H, self.DK).transpose(1, 2, 0)                           # (H,DK,R)
            L.pw1 = C(A(q + "conv.pointwise_conv1.weight")[:, :, 0])                            # (2D,D)
            L.dw = A(q + "conv.depthwise_conv.weight")[:, 0, :].T                               # (KC,D)
            L.pw2 = C(A(q + "conv.pointwise_conv2.weight")[:, :, 0])                            # (D,D)
            self.layers.append(L)
            mx.eval([getattr(L, s) for s in _Layer.__slots__])                                  # 레이어별 실체화 (최대 메모리 억제)
        mx.eval(self.c0w, self.c0b, self.out_w, self.out_b, self.subw)

        self._zk = mx.zeros((self.H, self.C, self.DK), f32)                                     # 공유 0 캐시 (불변 배열)
        self._zc = mx.zeros((self.KC - 1, self.D), f32)
        self._fns: dict = {}                                                                    # (rows, drop, T, compiled) → fn
        self._masks: dict = {}                                                                  # (valid, T) → mx 가산 마스크
        self._std = self._standard_shapes()

    # ------------------------------------------------------------------ 계산
    def _mm(self, x, w):
        """x(fp32) @ wᵀ — 가중치가 저정밀이면 입력만 내려 곱하고 결과는 fp32."""
        if w.dtype == mx.float32:
            return x @ w.T
        return (x.astype(w.dtype) @ w.T).astype(mx.float32)

    def _ff(self, x, w1, w2, li, tag):
        h = self._mm(x, w1)
        self._p(tag + "_h", li, h)
        return self._mm(_silu(h), w2)

    def _p(self, name, li, x):
        if self.probe is not None:
            self.probe(name, li, x)

    def _pre_encode(self, mel):
        """(rows,128) fp32 → (rows', D), rows' = f(f(f(rows)))."""
        h = mel[None, :, :, None]                                             # NHWC (1,T,F,1)
        h = mx.pad(h, [(0, 0), (2, 1), (2, 1), (0, 0)])                       # CausalConv2D: 왼쪽 2, 오른쪽 1 (T, F 모두)
        h = mx.maximum(mx.conv2d(h, self.c0w, stride=2) + self.c0b, 0)
        for dw_w, dw_b, pw_w, pw_b in self.subw:
            h = mx.pad(h, [(0, 0), (2, 1), (2, 1), (0, 0)])
            h = mx.conv2d(h, dw_w, stride=2, groups=self.SUB_CH) + dw_b
            h = mx.maximum(h @ pw_w.T + pw_b, 0)
        return h.reshape(h.shape[1], -1) @ self.out_w.T + self.out_b

    def _layer(self, li, L, x, kc, vc, cc, maskadd, idx):
        T, H, DK, D = x.shape[0], self.H, self.DK, self.D
        r = x + 0.5 * self._ff(_ln(x, L.n_ff1), L.ff1_1, L.ff1_2, li, "ff1")
        a = _ln(r, L.n_att)
        qkv = self._mm(a, L.wqkv)                                             # (T,3D) fp32
        self._p("qkv", li, qkv)
        q, k, v = (qkv[:, j * D:(j + 1) * D].reshape(T, H, DK).transpose(1, 0, 2) for j in range(3))
        K = mx.concatenate([kc, k], axis=1)                                   # (H,C+T,DK)
        V = mx.concatenate([vc, v], axis=1)
        bd = mx.take_along_axis((q + L.pbv) @ L.P_t, idx, axis=2)            # (H,T,C+T): rel = C+i-j
        sc = 1.0 / math.sqrt(DK)
        o = mx.fast.scaled_dot_product_attention((q + L.pbu)[None], K[None], V[None], scale=sc,
                                                 mask=(bd * sc + maskadd)[None])
        att = self._mm(o[0].transpose(1, 0, 2).reshape(T, D), L.wo)
        self._p("att", li, att)
        r = r + att
        g = self._mm(_ln(r, L.n_conv), L.pw1)                                 # (T,2D)
        self._p("pw1", li, g)
        g = g[:, :D] * mx.sigmoid(g[:, D:])                                   # GLU
        hc = mx.concatenate([cc, g], axis=0)                                  # (KC-1+T, D)
        y = hc[0:T] * L.dw[0]
        for j in range(1, self.KC):
            y = y + hc[j:j + T] * L.dw[j]                                     # causal depthwise (cross-correlation)
        cv = self._mm(_silu(_ln(y, L.n_cbn)), L.pw2)
        self._p("pw2", li, cv)
        r = r + cv
        r = r + 0.5 * self._ff(_ln(r, L.n_ff2), L.ff2_1, L.ff2_2, li, "ff2")
        self._p("resid", li, r)
        C = self.C
        return _ln(r, L.n_out), K[:, -C:], V[:, -C:], hc[-(self.KC - 1):]

    def _idx(self, T):
        i = np.arange(T)[:, None]
        j = np.arange(self.C + T)[None]
        idx = ((self.Tm - 1) - i + j).astype(np.int32)                        # P 표의 행 (rel = C+i-j)
        return mx.array(np.broadcast_to(idx[None], (self.H, T, self.C + T)).copy())

    def _step_fn(self, rows, drop, T, compiled):
        key = (rows, drop, T, compiled)
        fn = self._fns.get(key)
        if fn is None:
            idx = self._idx(T)
            net = weakref.proxy(self)            # 순환 참조 방지: del model 즉시 가중치·컴파일 캐시 해제

            def step(mel, kcs, vcs, ccs, maskadd):
                x = net._pre_encode(mel)[drop:]
                nk, nv, nc = [], [], []
                for li, (L, kc, vc, cc) in enumerate(zip(net.layers, kcs, vcs, ccs)):
                    x, kc, vc, cc = net._layer(li, L, x, kc, vc, cc, maskadd, idx)
                    nk.append(kc); nv.append(vc); nc.append(cc)
                return x, nk, nv, nc

            fn = mx.compile(step) if compiled else step
            self._fns[key] = fn
        return fn

    def _mask(self, valid, T):
        m = self._masks.get((valid, T))
        if m is None:
            a = np.zeros(self.C + T, np.float32)
            a[: self.C - valid] = MASK_NEG                                    # 아직 안 찬 캐시 슬롯
            m = self._masks[(valid, T)] = mx.array(a)
        return m

    def _standard_shapes(self):
        """완전한 청크 스텝의 (rows, drop) 모양들 — 이 모양만 컴파일한다."""
        out, idx, emitted = set(), 0, 0
        for c in range(4):
            size = self.first_size if c == 0 else self.size
            s = max(0, idx - self.pre_cache)
            rows = idx + size - s
            drop = emitted - s // self.sub
            out.add((rows, drop))
            emitted += self.Tm
            idx += size
        return out

    # ------------------------------------------------------------------ API
    def new_stream(self) -> "EncoderStream":
        return EncoderStream(self)

    def warmup(self) -> None:
        """첫 스텝·정상 스텝(컴파일)과 끝의 짧은 스텝 모양을 한 번씩 실행."""
        rng = np.random.default_rng(0)
        st = self.new_stream()
        st.feed((rng.standard_normal(4096 + 5120 * 2) * 0.01).astype(np.float32))
        mx.eval(st.finish(16000)[0])


class EncoderStream:
    """세그먼트 하나의 상태 (mel + K/V·conv 캐시). 새 세그먼트마다 model.new_stream()."""

    def __init__(self, model: EncoderModel):
        m = self._m = model
        self._mel = MelStream(logmel=m.logmel)
        self._kc = [m._zk] * m.NL
        self._vc = [m._zk] * m.NL
        self._cc = [m._zc] * m.NL
        self._valid = 0                      # NeMo cache_last_channel_len
        self._idx = 0                        # 다음 새 mel 프레임
        self._steps = 0
        self._emitted = 0                    # 계산된 인코더 프레임 수
        self._prev = np.zeros((0, NMEL), np.float32)   # mel[max(0,idx-9) : idx]
        self._done = False
        self.samples_fed = 0                 # 실제 + 0 꼬리 샘플 수
        self.frames_out = 0                  # 호출자에게 돌려준 프레임 수
        self.last_step_ms = 0.0              # 직전 feed/finish 의 스텝 시간 합 (mx.eval 후)
        self.step_ms: list[float] = []       # 직전 호출의 스텝별 시간
        self.K: int | None = None

    @property
    def k_received(self) -> int:
        """지금까지 받은 샘플 기준 K(n) — 프레임 k 는 k < K(n) 일 때만 디코드해도 eval 과 같다 (encoder-spec §6)."""
        return k_frames(self.samples_fed)

    def chunk_samples(self, c: int) -> int:
        """청크 c (프레임 Tm·c …) 계산에 필요한 세그먼트 샘플 수 ([56,3]: 4096 + 5120c)."""
        m = self._m
        last = m.first_size + m.size * c - 1                  # 청크 c 의 마지막 mel 프레임
        return HOP * last + 256

    def _step(self, new: np.ndarray, compiled_ok: bool) -> mx.array:
        m = self._m
        s = max(0, self._idx - m.pre_cache)
        assert len(self._prev) == self._idx - s
        rows = np.concatenate([self._prev, new]) if len(self._prev) else new   # mel[s : end], 0 으로 채우지 않는다
        drop = self._emitted - s // m.sub
        T = (rows.shape[0] + 14) // 8 - drop                                   # f³(rows) - drop
        compiled = compiled_ok and m.compile and m.probe is None and T == m.Tm and (rows.shape[0], drop) in m._std
        fn = m._step_fn(rows.shape[0], drop, T, compiled)
        t0 = time.perf_counter()
        x, self._kc, self._vc, self._cc = fn(mx.array(rows), self._kc, self._vc, self._cc, m._mask(self._valid, T))
        mx.eval(x, self._kc, self._vc, self._cc)
        self.step_ms.append((time.perf_counter() - t0) * 1e3)
        self._valid = min(m.C, self._valid + T)
        self._emitted += T
        self._steps += 1
        self._prev = rows[-m.pre_cache:]                                       # = mel[max(0, idx'-9) : idx']
        return x

    def _run(self, final: bool) -> list:
        m, outs = self._m, []
        while True:
            size = m.first_size if self._steps == 0 else m.size
            end, avail = self._idx + size, self._mel.ready()
            if avail < end:
                if not final:
                    break
                end = avail
                if end - self._idx < (1 if self._steps == 0 else m.sub):     # sampling_frames [1,8] 미만 꼬리는 버린다
                    break
            outs.append(self._step(self._mel.take(end), compiled_ok=not final or end == self._idx + size))
            self._idx += size
        return outs

    def _cat(self, outs):
        if not outs:
            return mx.zeros((0, self._m.D), mx.float32)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)

    def feed(self, pcm) -> mx.array:
        """16 kHz float32 PCM (임의 길이). 완성된 청크를 모두 계산 → (n, D) fp32 새 프레임, n ∈ {0, Tm, 2Tm, …}."""
        assert not self._done, "finish() 뒤에는 feed 할 수 없다"
        x = np.asarray(pcm, np.float32).reshape(-1)
        self.step_ms = []
        self._mel.push(x)
        self.samples_fed += x.size
        out = self._cat(self._run(final=False))
        self.last_step_ms = float(sum(self.step_ms))
        self.frames_out += out.shape[0]
        return out

    def finish(self, zero_tail_samples: int) -> tuple:
        """0 꼬리를 붙이고 K = int(round(n/16000·12.5)) 계산, 남은 청크 + 끝의 짧은 스텝(keep_all_outputs) 실행.
        → (남은 프레임 (frames_out 합계 ≤ K 로 자름), K). 합계가 K 보다 작으면 호출자가 0 으로 채운다."""
        assert not self._done, "finish() 는 한 번만"
        self.step_ms = []
        z = int(zero_tail_samples)
        if z > 0:
            self._mel.push(np.zeros(z, np.float32))
            self.samples_fed += z
        outs = self._run(final=False)
        self._mel.finish()
        self._done = True
        outs += self._run(final=True)
        self.K = k_frames(self.samples_fed)
        out = self._cat(outs)
        out = out[: max(0, self.K - self.frames_out)]
        self.last_step_ms = float(sum(self.step_ms))
        self.frames_out += out.shape[0]
        return out, self.K


def offline_features(model: EncoderModel, wav, tail_samples: int = 16000) -> mx.array:
    """eval 프로토콜 (wav + 0 꼬리) 의 특징 (K, D) — 스트리밍 경로 그대로."""
    st = model.new_stream()
    a = st.feed(np.asarray(wav, np.float32))
    b, K = st.finish(tail_samples)
    x = mx.concatenate([a, b], axis=0)[:K]
    if x.shape[0] < K:
        x = mx.concatenate([x, mx.zeros((K - x.shape[0], model.D), mx.float32)], axis=0)
    return x
