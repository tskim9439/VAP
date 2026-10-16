"""VAPKT 실시간 엔진 (MLX 전용): Engine, SegmentDecoder, Round, find_model, ModelNotFound.

Engine 은 워커 스레드 하나에서만 만들고 쓰고 버린다 (MLX 배열은 스레드를 넘을 수 없다)."""
from __future__ import annotations

import gc
import json
import math
import os
import time

from .checkpoint import ModelNotFound, find_model, load_config, load_tokenizer, load_weights, split_weights
from .decoder import MIN_TAIL, FinalJob, PrefixCache, Round, SegmentDecoder, k_of
from .tokens import DEFAULTS, DELAYS, LANGS, PAIRS, TokenTable, check_mode

__all__ = ["Engine", "SegmentDecoder", "Round", "FinalJob", "find_model", "ModelNotFound", "PRECISION_CHOICES", "DRAFT_CHOICES",
           "auto_note"]

ASSET_ENCODER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "nemotron_encoder.json")
PRECISION_CHOICES = ("auto", "fp32", "fp16", "fp16-plain", "bf16", "q8")
DRAFT_CHOICES = ("auto", "q4", "off")

# ── 자동 정밀도 (thinker-spec §6.4 [critic] + 골든 청크 통계, 통합 단계 W6 결정) ───────────────────────────────
# 시작 때 구성요소(thinker fp32/fp16, 인코더 fp32/fp16)를 하나씩 따로 재고(메모리에 한 번에 하나만 추가), 정확한 순서의
# 조합 중 실시간 여유 규칙을 처음 통과하는 것을 고른다. 하나도 통과 못 하면 예측 p95 가 가장 작은(가장 빠른) 조합.
# 청크(320 ms 오디오 4 프레임) 비용 모델 — enc: 인코더 정상 스텝, t1: q=1 토큰 forward, t8: 투기 배치 q=8 forward (문맥 320):
#   p95   = enc + 2·t8 + 13·t1       골든 번역 청크: 토큰 p95 13, 방출 프레임 p95 2 (방출 프레임마다 남은 프레임 재배치)
#   mean  = enc + 1.73·t8 + 2.85·t1  골든 번역 평균: 토큰 2.85/청크, 방출 프레임 0.73/청크
#   burst = enc + t8 + 23·t1         thinker-spec §6.4 의 p99 묶음 (골든 최대 20)
# 기준: p95 ≤ 0.8·320 ms, mean ≤ 0.5·320 ms (RTF 0.5), burst ≤ 450 ms (한 번 밀려도 다음 청크에서 회복).
# q4 초안(draft)을 쓰면 토큰 비용 = 검증 라운드 수 × c,  c = L·t_d + t_v  (t_d: 초안 q=1 한 스텝, t_v: 검증 q=L+1 forward):
#   p95 = enc + 2·t8 + ⌈13/(L+1)⌉·c,  burst = enc + t8 + ⌈23/(L+1)⌉·c,  mean = enc + 1.73·t8 + 2.85·t_d + 0.73·t_v
#   (방출 라운드 0.73/청크마다 검증 한 번, 토큰마다 초안 한 스텝; 골든 fp16 일치율 95 % — 라운드 끝 투기로 아끼는 t8 은 빼지 않음)
# 초안 'auto' 는 c ≤ 3.5·t1 (일치율 95 % 면 검증 한 번에 ≈ 4.5 토큰)일 때만 켠다 — Pro/Max 처럼 q4 스텝이 덜 싼 GPU 대비.
AUTO_RULE = dict(p95_tok=13, p95_t8=2.0, mean_tok=2.85, mean_t8=1.73, burst_tok=23, burst_t8=1.0,
                 p95_ms=256.0, mean_ms=160.0, burst_ms=450.0, draft_L=4, draft_gain=3.5)
# 정확한 것부터 (골든 마진 실측, W6 보고서 §2.2): fp32 thinker ≫ fp16 thinker. fp16 thinker 에서는 인코더 정밀도가 정확도에
# 차이가 없고(fp16/fp32 가 오히려 가장 가까운 near-flip 4.96e-5) fp32 인코더는 청크당 ≈ 17 ms 비싸므로 fp16/fp16 을 먼저 —
# 초안 예측으로 fp16 조합이 기준을 통과하게 되면서 (fp16, fp32) 가 먼저 뽑히던 것을 막는다.
AUTO_ORDER = (("fp32", "fp32"), ("fp32", "fp16"), ("fp16", "fp16"), ("fp16", "fp32"))


def predict_chunk_ms(enc_ms: float, t1: float, t8: float, rule: dict = AUTO_RULE, td: float | None = None,
                     tv: float | None = None) -> dict:
    """청크 비용 예측. td/tv 를 주면 q4 초안 검증 디코드 기준 (위 식)."""
    if td is None or tv is None:
        return dict(p95=enc_ms + rule["p95_t8"] * t8 + rule["p95_tok"] * t1, mean=enc_ms + rule["mean_t8"] * t8 + rule["mean_tok"] * t1,
                    burst=enc_ms + rule["burst_t8"] * t8 + rule["burst_tok"] * t1)
    L = int(rule.get("draft_L", 4))
    c = L * td + tv
    return dict(p95=enc_ms + rule["p95_t8"] * t8 + math.ceil(rule["p95_tok"] / (L + 1)) * c,
                mean=enc_ms + rule["mean_t8"] * t8 + rule["mean_tok"] * td + (rule["mean_t8"] - 1.0) * tv,
                burst=enc_ms + rule["burst_t8"] * t8 + math.ceil(rule["burst_tok"] / (L + 1)) * c)


def draft_pays(t1: float, td: float, tv: float, rule: dict = AUTO_RULE) -> bool:
    """초안 검증 라운드 한 번(L·t_d + t_v)이 q=1 forward draft_gain 번보다 싸면 켤 가치가 있다."""
    return int(rule.get("draft_L", 4)) * td + tv <= rule.get("draft_gain", 3.5) * t1


def choose_precision(cands, measure, rule: dict = AUTO_RULE, default: tuple | None = None) -> tuple[tuple, list[dict], str]:
    """cands: 정확한 순서의 (thinker, encoder) 후보. measure(th, en) → (enc_ms, t1_ms, t8_ms[, td_ms, tv_ms]) (필요할 때만
    불린다; td/tv 가 None 이 아니면 초안 기준 예측). 측정이 예외를 내면 그 후보는 실패로 적고 다음 후보로.
    → (고른 후보, 측정 표, 이유 'pass' | 'fallback_fastest' | 'fallback_default')."""
    table = []
    for th, en in cands:
        try:
            m = tuple(measure(th, en))
        except Exception as e:                                   # 측정 실패(메모리 등): 이 후보만 버린다
            table.append(dict(thinker=th, encoder=en, ok=False, error=f"{type(e).__name__}: {e}"[:300]))
            continue
        e, t1, t8 = m[:3]
        td, tv = (m[3], m[4]) if len(m) >= 5 else (None, None)
        p = predict_chunk_ms(e, t1, t8, rule, td, tv)
        ok = p["p95"] <= rule["p95_ms"] and p["mean"] <= rule["mean_ms"] and p["burst"] <= rule["burst_ms"]
        row = dict(thinker=th, encoder=en, enc_ms=round(e, 2), t1_ms=round(t1, 2), t8_ms=round(t8, 2),
                   **{k: round(v, 1) for k, v in p.items()}, ok=ok)
        if td is not None and tv is not None:
            row.update(draft=True, td_ms=round(td, 2), tv_ms=round(tv, 2))
        table.append(row)
        if ok:
            return (th, en), table, "pass"
    done = [r for r in table if "p95" in r]
    if not done:                                                 # 하나도 못 쟀다: 고정 기본값
        return (default or tuple(cands[-1])), table, "fallback_default"
    best = min(done, key=lambda r: (r["p95"], r["mean"]))
    return (best["thinker"], best["encoder"]), table, "fallback_fastest"


def auto_note(auto: dict | None) -> str:
    """info['auto'] → 한 줄 한국어 설명 (로그·UI)."""
    if not auto:
        return ""
    c = auto["choice"]
    row = next((r for r in auto["table"] if r["thinker"] == c["thinker"] and r["encoder"] == c["encoder"]), None)
    pred = (f"예측 청크 p95 {row['p95']:.0f} ms·평균 {row['mean']:.0f} ms / 320 ms" + (" (q4 초안)" if row.get("draft") else "")
            if row and "p95" in row else "")
    why = {"pass": "실시간 여유 기준 통과", "fallback_fastest": "어느 조합도 여유 기준(p95 ≤ 256 ms) 미달 → 가장 빠른 조합",
           "fallback_default": "측정 실패 → 고정 기본값"}.get(auto["reason"], auto["reason"])
    fails = [f"{r['thinker']}/{r['encoder']}: {r['error']}" for r in auto["table"] if r.get("error")]
    return (f"자동 정밀도: thinker {c['thinker']} · 인코더 {c['encoder'] or '없음'} ({why}; {pred}; 측정 {auto['ms']:.0f} ms)"
            + (f" — 측정 실패: {'; '.join(fails)}" if fails else ""))


def _median_ms(f, iters: int, warm: int = 2) -> float:
    for _ in range(warm):
        f()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        f()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _rnd_rows(th, seed: int = 0):
    import mlx.core as mx
    import numpy as np
    rng = np.random.default_rng(seed)
    return lambda n: mx.array((rng.standard_normal((n, th.D)) * 0.05).astype(np.float32)).astype(th.in_dtype)  # noqa: E731


def bench_thinker(th, ctx: int = 320, iters: int = 9) -> tuple[float, float]:
    """(t1, t8) ms: KV 문맥 ctx 에서 q=1 forward 와 투기 q=8 (sel 4 행) forward 의 중앙값 (argmax·동기화 포함).
    KV 가 작으면 문맥을 줄인다 (ctx + 8 ≤ max_len)."""
    import mlx.core as mx
    ctx = max(1, min(int(ctx), th.max_len - 8))
    rnd = _rnd_rows(th)
    th.forward(rnd(ctx), 0, sel=[])
    x1, x8 = rnd(1), rnd(8)
    mx.eval(x1, x8)
    return _median_ms(lambda: th.forward(x1, ctx), iters), _median_ms(lambda: th.forward(x8, ctx, [1, 3, 5, 7]), iters)


def bench_draft(th, draft, L: int = 4, ctx: int = 320, iters: int = 9) -> tuple[float, float]:
    """(t_d, t_v) ms: 초안 q=1 한 스텝(동기화 포함, 파이프라인보다 보수적)과 검증 forward q=L+1 (sel 전부), 문맥 ctx."""
    import mlx.core as mx
    ctx = max(1, min(int(ctx), th.max_len - L - 2))
    rnd = _rnd_rows(th, 1)
    th.forward(rnd(ctx), 0, sel=[])
    x1, xv = rnd(1), rnd(L + 1)
    mx.eval(x1, xv)
    td = _median_ms(lambda: draft.ids_lazy(x1, ctx).item(), iters)
    tv = _median_ms(lambda: th.forward(xv, ctx, list(range(L + 1))), iters)
    return td, tv


def bench_encoder(enc, steps: int = 8) -> float:
    """인코더 정상 스텝(320 ms 청크) ms 중앙값 (warmup 뒤, 컴파일 경로)."""
    import numpy as np
    enc.warmup()
    st = enc.new_stream()
    st.feed((np.random.default_rng(1).standard_normal(4096 + 5120 * steps) * 0.01).astype(np.float32))
    s = sorted(st.step_ms[1:])
    return s[len(s) // 2]


def _model_name(d: str) -> str:
    d = d.rstrip("/")
    if os.path.basename(d) == "final":
        d = os.path.dirname(d)
    n = os.path.basename(d)
    return n[: -len(".nosync")] if n.endswith(".nosync") else n


class Engine:
    """체크포인트 적재 + 세그먼트 생성. precision: thinker 정밀도, encoder_precision: 인코더 정밀도.

    'auto' = 시작 때 마이크로 벤치(choose_precision): (fp32,fp32) → (fp32,fp16) → (fp16,fp16) → (fp16,fp32) 순으로
    실시간 여유 규칙(AUTO_RULE)을 처음 통과하는 조합, 없으면 가장 빠른 조합. 결정·측정값은 info['auto'] 에 남는다.
    한쪽만 auto 면 다른 쪽은 고정한 채 후보를 고른다. 측정이 하나도 안 되면 고정 fp16/fp16.
    draft: 'auto' (q4 초안이 이 기기에서 이득일 때만, draft_pays) | 'q4' | 'off'. two_stage_head: fp32 thinker 의 lm_head 를
    인증서 두 단계로. prefix_cache: prefix KV 캐시 항목 수 (0 = 끔). final_lookahead: final 재디코드 한 걸음의 초안 수 (0 = q=1,
    None = fp32 thinker 면 11, 아니면 0 — fp16 final 모드에는 마진 0.0101 의 near-tie 가 있어(epst-de2-de2en-d10) q>1 검증 forward
    수치로 fp16/fp32 골든 final 이 38/39 가 됐다. fp32 thinker 는 4 조합 모두 39/39)."""

    def __init__(self, model_dir: str | None = None, kv_capacity: int = 2048, precision: str = "auto",
                 encoder_precision: str = "auto", speculative: bool = True, *, encoder: bool = True,
                 hard_max_s: float = 30.0, tail_s: float = 1.0, strict: bool = True, auto_rule: dict | None = None,
                 cache_limit_gb: float | None = 1.0, draft: str = "auto", two_stage_head: bool = True,
                 prefix_cache: int = 16, final_lookahead: int | None = None):
        import mlx.core as mx
        from .thinker import PRECISIONS, Adapter, MlxThinker
        t0 = time.perf_counter()
        if precision not in PRECISION_CHOICES:
            raise ValueError(f"precision 은 {PRECISION_CHOICES} 중 하나여야 합니다: {precision!r}")
        if encoder_precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError(f"encoder_precision 은 auto/fp32/fp16/bf16 중 하나: {encoder_precision!r}")
        if draft not in DRAFT_CHOICES:
            raise ValueError(f"draft 는 {DRAFT_CHOICES} 중 하나: {draft!r}")
        rule = auto_rule or AUTO_RULE
        self.dir = find_model(model_dir)
        self.cfg = load_config(self.dir, strict=strict)
        tc = self.cfg["thinker"]["text_config"]
        self.tokens = TokenTable(load_tokenizer(self.dir), self.cfg, tc["vocab_size"])
        sp = self.sp = self.tokens.sp
        p_max = max(len(self.tokens.prefix_ids(s, t, d)) for s, t in [(x, None) for x in LANGS] + list(PAIRS) for d in DELAYS)
        if int(kv_capacity) < p_max + 2:            # 무거운 적재 전에: 너무 작으면 측정·warmup 이 알 수 없는 오류로 죽는다
            raise ValueError(f"--kv-capacity {kv_capacity} 가 너무 작습니다: prefix 최대 {p_max} 토큰 + 2 = {p_max + 2} 이상이어야 "
                             f"합니다 (기본 2048, 세그먼트 30 s 번역에 ≈ 1100)")
        w = load_weights(self.dir)
        th_w, ad_w, en_w = split_weights(w, self.dir)
        del w
        self.adapter = Adapter(ad_w)
        del ad_w
        ecfg = None
        if encoder:
            n_enc = sum(k.startswith("enc.") for k in en_w)
            n_pre = sum(k.startswith("pre.") for k in en_w)
            if n_enc != 636 or sorted(k for k in en_w if k.startswith("pre.")) != ["pre.featurizer.fb", "pre.featurizer.window"]:
                raise RuntimeError(f"인코더 키 수가 다릅니다: enc {n_enc} (기대 636), pre {n_pre} (기대 2)")
            with open(ASSET_ENCODER) as f:
                ecfg = json.load(f)
        del th_w, en_w                                      # 지연 배열 (아직 안 읽음) — 구성요소마다 새로 연다 (fresh)
        rc = int(self.cfg["encoder_right_context"])

        def fresh(kind):
            """구성요소의 fp32 원본을 새 지연 배열로 (mx.load 는 지연 적재). 만든 모델이 dict 를 소비하고 버리므로 측정·생성 사이에
            fp32 원본이 메모리에 남지 않는다 (예전: 두 원본 4.8 GB 가 측정 내내 상주, 적재 최대 6.65 GB)."""
            t, _, e = split_weights(load_weights(self.dir), self.dir)
            return t if kind == "th" else e

        def build(kind, p):
            if kind == "th":
                dt, resid, rk, qb = PRECISIONS[p]
                th = MlxThinker(fresh("th"), tc, self.cfg["blocked_ids"], dtype=dt, max_len=kv_capacity,
                                resid_fp32=resid, rescore_k=rk, quant_bits=qb)
                if two_stage_head:
                    th.enable_two_stage()                   # fp32 thinker 일 때만 켜진다
                return th
            from .encoder import EncoderModel
            edt = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[p]
            return EncoderModel(fresh("en"), ecfg, right_context=rc, dtype=edt, consume=True)

        def draft_for(th, t1: float | None = None) -> tuple:
            """→ (Draft | None, 측정 dict | None). 'auto' 는 이 기기에서 이득일 때만 켠다 (t1: 이미 잰 q=1 ms).
            'auto'·'q4' 모두 t_d·t_v 를 재서 남긴다 (자동 정밀도 예측에 쓴다). 실패하면 초안 없이 (q=1 디코드)."""
            if draft == "off":
                return None, None
            try:
                d = th.make_draft(4)
                if d is None:
                    return None, dict(on=False, reason="원본 가중치 없음 (q8)")
                if t1 is None:
                    t1, _ = bench_thinker(th)
                td, tv = bench_draft(th, d, int(rule.get("draft_L", 4)))
            except Exception as e:                          # 초안은 가속일 뿐: 만들거나 재다 실패해도 엔진은 뜬다
                gc.collect(); mx.clear_cache()
                return None, dict(on=False, reason=f"오류 {type(e).__name__}: {e}"[:200])
            on = draft == "q4" or draft_pays(t1, td, tv, rule)
            m = dict(on=on, t1_ms=round(t1, 2), td_ms=round(td, 2), tv_ms=round(tv, 2),
                     reason="forced" if draft == "q4" else ("pass" if on else "no_gain"))
            return (d if on else None), m

        slot: dict = {}                                     # 측정용 한 칸: kind → (정밀도, 모델, 초안, 초안 측정)

        def measured(kind, p):
            slot.clear(); gc.collect(); mx.clear_cache()
            m = build(kind, p)
            slot[kind] = (p, m, None, None)
            return m

        auto = None
        if precision == "auto" or (encoder and encoder_precision == "auto"):
            tcands = ("fp32", "fp16") if precision == "auto" else (precision,)
            ecands = (("fp32", "fp16") if encoder_precision == "auto" else (encoder_precision,)) if encoder else (None,)
            cands = [c for c in AUTO_ORDER if c[0] in tcands and c[1] in ecands]
            cands += [(t, e) for t in tcands for e in ecands if (t, e) not in cands]
            meas: dict = {}

            def measure(tp, ep):
                if ("th", tp) not in meas:
                    try:
                        th = measured("th", tp)
                        t1, t8 = bench_thinker(th)
                        d, dm = draft_for(th, t1)
                        slot["th"] = (tp, th, d, dm)
                        meas[("th", tp)] = (t1, t8) + ((dm["td_ms"], dm["tv_ms"]) if d is not None else (None, None))
                    except Exception:
                        slot.clear(); gc.collect(); mx.clear_cache()
                        raise
                if ep is not None and ("en", ep) not in meas:
                    try:
                        meas[("en", ep)] = bench_encoder(measured("en", ep))
                    except Exception:
                        slot.clear(); gc.collect(); mx.clear_cache()
                        raise
                return (meas.get(("en", ep), 0.0),) + meas[("th", tp)]

            ta = time.perf_counter()
            dflt = ("fp16" if "fp16" in tcands else tcands[-1], ("fp16" if "fp16" in ecands else ecands[-1]) if encoder else None)
            (tprec, eprec), table, why = choose_precision(cands, measure, rule, default=dflt)
            auto = dict(choice=dict(thinker=tprec, encoder=eprec), reason=why, table=table, rule=rule,
                        ms=round((time.perf_counter() - ta) * 1000, 1),
                        measured={f"{k}:{p}": [None if v is None else round(v, 2) for v in (x if isinstance(x, tuple) else (x,))]
                                  for (k, p), x in meas.items()})
        else:
            tprec, eprec = precision, (encoder_precision if encoder else None)
        reuse = {k: v for k, v in slot.items() if v[0] == {"th": tprec, "en": eprec}[k]}   # 마지막 측정 모델이 맞으면 그대로
        slot.clear(); gc.collect(); mx.clear_cache()
        need_draft = "th" not in reuse
        if need_draft:
            self.thinker = build("th", tprec)
            self.draft, dmeas = None, None
        else:
            _, self.thinker, self.draft, dmeas = reuse["th"]
        gc.collect()
        self.encoder = (reuse["en"][1] if "en" in reuse else build("en", eprec)) if encoder else None
        reuse.clear()
        if need_draft:                                      # 인코더 생성(일시 fp32 원본) 뒤에: 적재 최대 메모리에 초안을 겹치지 않는다
            gc.collect()
            self.draft, dmeas = draft_for(self.thinker)
        gc.collect()
        mx.clear_cache()
        if cache_limit_gb is not None:      # MLX 빈 버퍼 캐시 상한: final KV·prefill 처럼 크기가 매번 다른 버퍼가 쌓이지 않게 (기본은 메모리 한도까지)
            mx.set_cache_limit(int(cache_limit_gb * 2 ** 30))
        import numpy as np
        ob = np.zeros(tc["vocab_size"], np.float32)
        ob[self.tokens.offline_blocked] = -np.inf
        self._offline_block = mx.array(ob)
        self.kv_capacity = int(kv_capacity)
        self.speculative = bool(speculative)
        self.draft_len = int(rule.get("draft_L", 4))
        self.draft_on = self.draft is not None              # 실행 중 끄고 켤 수 있다 (검증·벤치)
        self._prefix_cache = PrefixCache(self.thinker, prefix_cache) if prefix_cache else None
        self.prefix_cache_on = self._prefix_cache is not None
        if final_lookahead is None:
            final_lookahead = 11 if self.thinker.dtype == mx.float32 else 0
        self.final_lookahead = max(0, min(15, int(final_lookahead)))
        self.k_cap = k_of(int(round((hard_max_s + tail_s) * 16000)), self.cfg["frame_hz"])
        self.langs, self.pairs, self.delays, self.defaults = list(LANGS), list(PAIRS), list(DELAYS), DEFAULTS
        self.lang_labels = dict(LANGS)
        self.spk_ids, self.trans_id = (sp.spk_a, sp.spk_b), sp.trans
        dev = mx.device_info() if hasattr(mx, "device_info") else {}
        dinfo = dict(dmeas or {})
        dinfo.update(mode=draft, on=self.draft_on, bits=4, L=self.draft_len,
                     gb=round(self.draft.nbytes / 1e9, 3) if self.draft is not None else 0.0)
        self.info = dict(name=_model_name(self.dir), dir=self.dir, precision=tprec, precision_requested=precision,
                         encoder_precision=eprec if encoder else None, kv_capacity=self.kv_capacity,
                         load_ms=round((time.perf_counter() - t0) * 1000, 1), speculative=self.speculative,
                         mlx=mx.__version__, device=dev.get("device_name", str(mx.default_device())),
                         runaway_cap=self.cfg["runaway_cap"], max_flush_rounds=self.cfg["max_flush_rounds"], k_cap=self.k_cap,
                         encoder_precision_requested=encoder_precision if encoder else None, auto=auto, cache_limit_gb=cache_limit_gb,
                         draft=dinfo, two_stage=self.thinker.ts is not None, prefix_cache=int(prefix_cache or 0),
                         final_lookahead=self.final_lookahead,
                         precision_note=(auto_note(auto) or f"정밀도 고정: thinker {tprec} · 인코더 {eprec or '없음'}")
                         + (" · q4 초안 켬" if self.draft_on else ""))

    # ── API
    def new_segment(self, src_lang: str, tgt_lang: str | None, delay: int, *, known_K: int | None = None,
                    on_token=None) -> SegmentDecoder:
        """새 세그먼트 (새 인코더 스트림·KV·prefix). tgt_lang=None 이면 전사만. on_token(k, id): 토큰이 확정되는 즉시
        (워커 스레드에서; 같은 토큰이 그 feed/finish 가 돌려주는 Round.ids 에도 있다)."""
        check_mode(src_lang, tgt_lang, delay)
        return SegmentDecoder(self.thinker, self.adapter, self.tokens.prefix_ids(src_lang, tgt_lang, delay), self.sp,
                              cap=self.cfg["runaway_cap"], max_flush=self.cfg["max_flush_rounds"], frame_hz=self.cfg["frame_hz"],
                              kv_capacity=self.kv_capacity, stream=self.encoder.new_stream() if self.encoder else None,
                              speculative=self.speculative, k_cap=self.k_cap, known_K=known_K, tokens=self.tokens,
                              mode=(src_lang, tgt_lang), offline_block=self._offline_block,
                              draft=self.draft if self.draft_on else None, draft_len=self.draft_len, on_token=on_token,
                              prefix_cache=self._prefix_cache if self.prefix_cache_on else None,
                              final_lookahead=self.final_lookahead)

    def token_bytes(self, tid: int) -> bytes:
        return self.tokens.token_bytes(tid)

    def warmup(self) -> dict:
        """모든 모양을 한 번씩: 기본 δ 의 prefix (캐시도 채움), q=1..12 (sel 패턴·검증 행 전부), 초안 q=1,
        두 단계 헤드의 대체 경로(현재 헤드), 인코더 첫/정상/끝 스텝, 합성 세그먼트."""
        import mlx.core as mx
        import numpy as np
        t0 = time.perf_counter()
        th = self.thinker
        if self.encoder is not None:
            self.encoder.warmup()
        t_enc = time.perf_counter()
        modes = [(s, None) for s in LANGS] + list(PAIRS)
        for s, t in modes:                                     # prefix 길이 14/15/19/20, 기본 δ (캐시 항목)
            self.new_segment(s, t, DEFAULTS["asr_delay"] if t is None else DEFAULTS["translate_delay"][f"{s}>{t}"])
        P = len(self.tokens.prefix_ids("English", None, 6))
        rng = np.random.default_rng(0)
        x = mx.array(rng.standard_normal((12, th.D)).astype(np.float32) * 0.05).astype(th.in_dtype)
        ts = th.ts
        for use_ts in ((False, True) if ts is not None else (True,)):
            th.ts = ts if use_ts else None                     # 두 단계 헤드면 대체 경로(현재 헤드)도 컴파일해 둔다
            for q in range(1, 9):                              # 첫 그룹 q=1/3/5/7 (sel 짝수), 이후 q=2/4/6/8 (sel 홀수), 토큰 q=1
                sel = list(range(q % 2 == 0, q, 2)) if q > 1 else [0]
                th.forward(x[:q], P, sel)
            th.forward(x[:2], P, [1])
            for q in range(2, 13):                             # 검증 forward (초안·라운드 끝 투기·final lookahead): sel 전부
                th.forward(x[:q], P, list(range(q)))
        th.ts = ts
        if self.draft is not None:
            for _ in range(2):
                self.draft.ids_lazy(x[:1], P).item()
        segs = 0
        if self.encoder is not None:                           # 합성 세그먼트: −50 dBFS 잡음 2 s + 꼬리 1 s
            pcm = (rng.standard_normal(32000) * 10 ** (-50 / 20) * np.sqrt(2)).astype(np.float32)
            seg = self.new_segment("English", None, DEFAULTS["asr_delay"])
            for i in range(0, len(pcm), 2560):
                seg.feed(pcm[i:i + 2560])
            seg.finish(16000)
            segs = 1
        return dict(warmup_ms=round((time.perf_counter() - t0) * 1000, 1), encoder_ms=round((t_enc - t0) * 1000, 1),
                    prefixes=len(modes), segments=segs)

    def close(self) -> None:
        """MLX 메모리 해제 (워커 종료 시)."""
        import mlx.core as mx
        if self._prefix_cache is not None:
            self._prefix_cache.clear()
        self.thinker = self.adapter = self.encoder = self.draft = self._prefix_cache = None
        self.draft_on = self.prefix_cache_on = False
        gc.collect()
        mx.clear_cache()
