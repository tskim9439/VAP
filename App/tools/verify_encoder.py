#!/usr/bin/env python3
"""MLX 인코더(engine/frontend.py + engine/encoder.py) 검증·측정 도구.

섹션 (여러 개 가능, 무거운 실행은 반드시 heavy lock 아래에서):
  nemo       (a) NeMo 랜덤 가중치 골든과 비교 — <randw>/model.safetensors(체크포인트 형식) + golden/*.npz (+ 예전 probe npz)
  golden     (b) 실제 s4 가중치 골든 <GOLD>/<case>/{meta.json,wav.npy,mel.npy,feats.npy,feats_stream.npy} 과 비교 (fp32/fp16):
                 mel 차이, 특징 차이(torch 오프라인·torch 스트리밍), 패킷 방식 6 가지의 비트 일치, K·K 게이팅 통계
  edge       (b2) 경계 길이(K=4c+3, 반올림 경계, 아주 짧은 세그먼트, 끝 mel 나머지; 꼬리 0/385/3200/16000) torch 기준과 비교
                 — 기준은 scratch 의 make_edge_ref.py 가 <EDGE>/cases.json + <name>.npz 로 만든다 (torch 는 App 밖)
  timing     (c) 스텝 시간 p50/p95 (fp32/fp16 × compiled/eager), 첫 스텝·끝 스텝·mel·warmup·적재 시간
  precision  실제 가중치로 fp16/bf16 vs fp32 (테스트 wav + 1 s 꼬리), 레이어별 최대 |활성값| (fp16 여유)

예) /usr/bin/lockf -k -t 14400 <scratchpad>/heavy.lock python tools/verify_encoder.py nemo timing --json out.json
torch/NeMo 를 import 하지 않는다 (골든은 미리 만들어 둔 파일).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
import wave

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import mlx.core as mx  # noqa: E402

from engine.encoder import EncoderModel, k_frames  # noqa: E402
from engine.frontend import MelStream, final_frames, logmel_full  # noqa: E402

CKPT = os.path.expanduser("~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final")
RANDW = "/Volumes/Samsung_T5/vapkt-models/app-work/encoder-randw"
GOLD = "/Volumes/Samsung_T5/vapkt-models/app-work/golden-s4"
EDGE = "/Volumes/Samsung_T5/vapkt-models/app-work/encoder-edge-s4"       # 경계 길이 torch 기준 (scratch make_edge_ref.py)
PROBE_NPZ = ("/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad/"
             "scratch-encoder/out/nemo-6s-56-3.npz")
WAVS = ("/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad/"
        "testaudio/*.wav")
DT = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ 공통
def load_cfg(path):
    with open(path) as f:
        return json.load(f)


def load_encoder_weights(path: str) -> dict:
    """safetensors (파일 또는 <dir>[/final]/model.safetensors) → 'encoder.' 를 뗀 지연 배열 dict."""
    if os.path.isdir(path):
        for c in (os.path.join(path, "model.safetensors"), os.path.join(path, "final", "model.safetensors")):
            if os.path.exists(c):
                path = c
                break
    w = {k[len("encoder."):]: v for k, v in mx.load(path).items() if k.startswith("encoder.")}
    n_enc = sum(k.startswith("enc.") for k in w)
    assert n_enc == 636 and "pre.featurizer.fb" in w and "pre.featurizer.window" in w, (n_enc, len(w))
    return w


def free():
    gc.collect()
    mx.clear_cache()


def build(wpath, cfg, R, dt):
    t0 = time.perf_counter()
    m = EncoderModel(load_encoder_weights(wpath), cfg, right_context=R, dtype=DT[dt])
    return m, (time.perf_counter() - t0) * 1e3


def cmp(a, b) -> dict:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.abs(a - b)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    return dict(n=n, max=float(d.max()) if n else 0.0, mean=float(d.mean()) if n else 0.0,
                rel_l2=float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)), min_cos=float(cos.min()) if n else 1.0,
                argmax_frame=int(d.max(1).argmax()) if n else -1, finite=bool(np.isfinite(a).all()))


class FixedMel:
    """미리 계산된 mel (T,128) 을 MelStream 처럼 내준다 (인코더만 따로 비교할 때)."""

    def __init__(self, mel):
        self.mel, self.n, self.t, self.done = np.asarray(mel, np.float32), 0, 0, False

    def push(self, x):
        self.n += len(x)

    def finish(self):
        self.done = True

    def ready(self):
        return len(self.mel) if self.done else min(final_frames(self.n), len(self.mel))

    def take(self, t1):
        out = self.mel[self.t:t1]
        self.t = max(self.t, t1)
        return out


def run_stream(model, wav, n_real, tail, mel=None, seed=5, sizes=(64, 2048), live_tail=False, stats=None):
    """wav[:n_real] 을 무작위 패킷(sizes=(lo,hi) 균등, 또는 정수 = 고정 크기, None = 한 번에)으로 feed + finish(tail)
    → (프레임 np, K). live_tail=True 면 0 꼬리도 feed 로 넣고 finish(0) (실제 쉼 오디오를 라이브로 받은 경우).
    stats(dict) 를 주면 K 게이팅 통계를 채운다: feed 가 낸 프레임 중 최종 K 이상인 수, 나올 때 k_received 이상이던 수."""
    st = model.new_stream()
    if mel is not None:
        st._mel = FixedMel(mel)
    x = np.asarray(wav, np.float32)
    n_feed = n_real + (tail if live_tail else 0)
    if len(x) < n_feed:
        x = np.concatenate([x, np.zeros(n_feed - len(x), np.float32)])
    rng = np.random.default_rng(seed)
    outs, pos, ge_recv, feed_idx = [], 0, 0, []
    while pos < n_feed:
        b = n_feed if sizes is None else (sizes if isinstance(sizes, int) else int(rng.integers(sizes[0], sizes[1] + 1)))
        o = np.array(st.feed(x[pos:min(n_feed, pos + b)]))
        k0 = sum(len(t) for t in outs)
        feed_idx += range(k0, k0 + len(o))
        ge_recv += sum(k >= st.k_received for k in range(k0, k0 + len(o)))
        outs.append(o)
        pos += b
    o, K = st.finish(0 if live_tail else tail)
    outs.append(np.array(o))
    y = np.concatenate(outs)
    if stats is not None:
        stats.update(feed_frames=len(feed_idx), feed_ge_K=sum(k >= K for k in feed_idx), feed_ge_k_received=int(ge_recv),
                     finish_frames=len(o), frames_out=st.frames_out, K=K)
    return y, K


def read_wav(p):
    with wave.open(p, "rb") as w:
        assert w.getsampwidth() == 2 and w.getframerate() == 16000 and w.getnchannels() == 1, p
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


# ------------------------------------------------------------------ (a) NeMo 랜덤 가중치
def sec_nemo(a, cfg, R):
    meta = json.load(open(os.path.join(a.randw, "meta.json")))
    res = dict(meta_checksum=meta.get("checksum"), probe6s_regen_vs_stored=meta.get("probe6s_vs_stored_off"), cases={})
    for dt in a.dtypes:
        models = {}
        for name, cm in meta["cases"].items():
            ctx = tuple(cm["ctx"])
            if ctx[1] not in models:
                models[ctx[1]], ms = build(a.randw, cfg, ctx[1], dt)
                res.setdefault("build_ms", {})[f"{dt}|R{ctx[1]}"] = round(ms)
            m = models[ctx[1]]
            z = np.load(os.path.join(a.randw, "golden", name + ".npz"))
            wav, mel, off, K = z["wav"], z["mel"], z["off"], cm["K"]
            tail = cm["n_total"] - cm["n_real"]
            y_live, K1 = run_stream(m, wav, cm["n_real"], tail)
            y_mel, K2 = run_stream(m, wav, cm["n_real"], tail, mel=mel)
            assert K1 == K2 == K, (K1, K2, K)
            row = dict(K=K, stream_frames=len(y_live), offline_frames=len(off),
                       live_vs_offline_K=cmp(y_live[:K], off[:K]), nemo_mel_vs_offline_K=cmp(y_mel[:K], off[:K]),
                       live_vs_offline_all=cmp(y_live, off))
            if dt == "fp32":
                mm = logmel_full(wav, m.window, m.fb)
                row["mel_vs_nemo"] = dict(max=float(np.abs(mm - mel).max()), mean=float(np.abs(mm - mel).mean()))
            if name == "probe6s" and os.path.exists(a.probe_npz):
                pz = np.load(a.probe_npz)
                row["live_vs_stored_probe_off_K"] = cmp(y_live[:K], pz["off"][:K])
                row["live_vs_stored_probe_nemo_stream"] = cmp(y_live, pz["st"])
            res["cases"][f"{dt}|{name}"] = row
            log(f"[nemo] {dt} {name:11s} K={K} frames={len(y_live)}/{len(off)} live max {row['live_vs_offline_K']['max']:.3g} "
                f"mean {row['live_vs_offline_K']['mean']:.3g} cos {row['live_vs_offline_K']['min_cos']:.7f} | "
                f"nemo-mel max {row['nemo_mel_vs_offline_K']['max']:.3g}"
                + (f" | mel {row['mel_vs_nemo']['max']:.3g}" if "mel_vs_nemo" in row else ""))
        del models, m
        free()
    return res


# ------------------------------------------------------------------ (b) 실제 s4 골든
# 패킷 방식: 이름 → (sizes, seed, live_tail). 첫 항목이 기준, 나머지는 기준과 비트 단위로 같아야 한다.
REGIMES = {
    "rand64-2048": ((64, 2048), 5, False),
    "rand1-400": ((1, 400), 6, False),
    "rand2k-24k": ((2048, 24000), 7, False),
    "fixed320": (320, 0, False),
    "oneshot": (None, 0, False),
    "livetail": ((64, 2048), 8, True),     # 0 꼬리를 feed 로 (finish(0)) — 결과는 [:K] 만 비교
}


def mel_diff(wav, gm, window, fb):
    """mel 차이: 전체 최대와 log-mel > -5 (무음 근처 제외) 최대."""
    mm = logmel_full(wav, window, fb)
    n = min(len(mm), len(gm))
    d = np.abs(mm[:n] - gm[:n])
    hi = gm[:n] > -5
    return dict(T=[len(mm), len(gm)], max=float(d.max()), mean=float(d.mean()), max_logmel_gt_m5=float(d[hi].max()) if hi.any() else 0.0)


def regimes_run(m, wav, n_real, tail, K_ref, names):
    """여러 패킷 방식으로 실행 → (기준 프레임, {방식: 결과})."""
    out, ref = {}, None
    for nm in names:
        sizes, seed, live = REGIMES[nm]
        stt = {}
        y, K1 = run_stream(m, wav, n_real, tail, seed=seed, sizes=sizes, live_tail=live, stats=stt)
        if ref is None:
            ref = y
        out[nm] = dict(K_ok=K1 == K_ref, frames=len(y), covers_K=len(y) >= K_ref,
                       bit_identical=bool(np.array_equal(y[:K_ref], ref[:K_ref])), **stt)
    return ref, out


def summarize(rows: dict) -> dict:
    """케이스별 결과 → 최악값 요약."""
    s = {}
    for k in ("feats", "feats_stream", "from_gold_mel"):
        v = [r[k] for r in rows.values() if k in r]
        if v:
            s[k] = dict(max=max(x["max"] for x in v), mean_max=max(x["mean"] for x in v), rel_l2_max=max(x["rel_l2"] for x in v),
                        min_cos=min(x["min_cos"] for x in v), all_finite=all(x["finite"] for x in v))
    reg = [rr for r in rows.values() for rr in r.get("regimes", {}).values()]
    if reg:
        s["all_K_ok"] = all(r["K_ok"] for r in reg)
        s["all_cover_K"] = all(r["covers_K"] for r in reg)
        s["all_bit_identical"] = all(r["bit_identical"] for r in reg)
        s["feed_ge_K_finish_tail"] = sum(r["feed_ge_K"] for r in reg if r.get("live") is False)
    return s


def sec_golden(a, cfg, R):
    if not os.path.exists(os.path.join(a.gold, "DONE")):
        log(f"[golden] {a.gold}/DONE 없음 — 건너뜀")
        return dict(skipped="no DONE")
    idx = os.path.join(a.gold, "index.json")
    cases = json.load(open(idx)) if os.path.exists(idx) else None
    if isinstance(cases, dict):
        cases = cases.get("cases", list(cases))
    names = [c if isinstance(c, str) else c.get("case", c.get("name")) for c in cases] if cases else \
        sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(a.gold, "*", "feats.npy")))
    res = {}
    for dt in a.dtypes:
        m, ms = build(a.ckpt, cfg, R, dt)
        res[f"build_ms|{dt}"] = round(ms)
        done, rows = {}, {}
        for name in names:
            d = os.path.join(a.gold, name)
            if not os.path.exists(os.path.join(d, "feats.npy")):
                continue
            meta = json.load(open(os.path.join(d, "meta.json")))
            wav = np.load(os.path.join(d, "wav.npy")).astype(np.float32)
            n_total = int(meta.get("n_total", len(wav)))
            n_real = int(meta.get("n_real", n_total - 16000))
            K = int(meta["K"])
            key = (n_real, n_total, hash(wav.tobytes()))
            if key not in done:
                feats = np.load(os.path.join(d, "feats.npy"))
                fs_p = os.path.join(d, "feats_stream.npy")
                gm = np.load(os.path.join(d, "mel.npy"))
                assert gm.shape[-1] == 128 and len(feats) == K, (gm.shape, feats.shape, K)
                ref, reg = regimes_run(m, wav, n_real, n_total - n_real, K, list(REGIMES))
                for nm, r in reg.items():
                    r["live"] = REGIMES[nm][2]
                ym, _ = run_stream(m, wav, n_real, n_total - n_real, mel=gm)
                row = dict(K=K, K_eval=k_frames(n_total), feats=cmp(ref[:K], feats), from_gold_mel=cmp(ym[:K], feats), regimes=reg)
                if os.path.exists(fs_p):
                    row["feats_stream"] = cmp(ref[:K], np.load(fs_p))
                if dt == "fp32":
                    row["mel"] = mel_diff(wav[:n_total], gm, m.window, m.fb)
                done[key] = row
            row = dict(done[key], case=name, K=K)
            rows[name] = row
            rg = row["regimes"]
            log(f"[golden] {dt} {name:22s} K={K} max {row['feats']['max']:.3g} mean {row['feats']['mean']:.3g} "
                f"rel {row['feats']['rel_l2']:.2g} cos {row['feats']['min_cos']:.7f} | vs torch-stream {row.get('feats_stream', {}).get('max', float('nan')):.3g}"
                f" | gold-mel {row['from_gold_mel']['max']:.3g}"
                + (f" | mel {row['mel']['max']:.3g} (>-5: {row['mel']['max_logmel_gt_m5']:.2g})" if "mel" in row else "")
                + f" | regimes bit-identical {all(r['bit_identical'] for r in rg.values())} K ok {all(r['K_ok'] and r['covers_K'] for r in rg.values())}"
                f" feed≥K {[r['feed_ge_K'] for r in rg.values()]} wait {[r['feed_ge_k_received'] for r in rg.values()]}")
        res[dt] = dict(cases=rows, summary=summarize(rows))
        log(f"[golden] {dt} 요약: {json.dumps(res[dt]['summary'])}")
        del m
        free()
    return res


# ------------------------------------------------------------------ (b2) 경계 길이 (실제 가중치 torch 기준, make_edge_ref)
def sec_edge(a, cfg, R):
    """<edge>/cases.json + <name>.npz(wav, feats) — K=4c+3 경계·반올림 경계·아주 짧은 세그먼트·끝 mel 나머지, 꼬리 0/385/3200/16000."""
    cj = os.path.join(a.edge, "cases.json")
    if not os.path.exists(cj):
        log(f"[edge] {cj} 없음 — 건너뜀")
        return dict(skipped="no cases.json")
    cases = json.load(open(cj))["cases"]
    res = {}
    for dt in a.dtypes:
        m, ms = build(a.ckpt, cfg, R, dt)
        rows = {}
        for name, c in cases.items():
            if "error" in c:
                rows[name] = dict(torch_error=c["error"])
                log(f"[edge] {dt} {name}: torch 기준 없음 ({c['error']})")
                continue
            z = np.load(os.path.join(a.edge, name + ".npz"))
            wav, feats, K = z["wav"], z["feats"], int(c["K"])
            ref, reg = regimes_run(m, wav, c["n_real"], c["tail"], K, ["rand64-2048", "rand1-400", "oneshot", "livetail"])
            for nm, r in reg.items():
                r["live"] = REGIMES[nm][2]
            row = dict(K=K, K_eval=k_frames(c["n_total"]), n_total=c["n_total"], tail=c["tail"], T_enc_torch=c.get("T_enc"),
                       torch_zero_padded=c.get("zero_padded"), feats=cmp(ref[:K], feats), regimes=reg)
            rows[name] = row
            r0 = reg["rand64-2048"]
            log(f"[edge] {dt} {name:22s} n={c['n_total']} tail={c['tail']} K={K} frames {r0['frames']} (feed {r0['feed_frames']}"
                f" finish {r0['finish_frames']}) max {row['feats']['max']:.3g} cos {row['feats']['min_cos']:.7f}"
                f" | bit-identical {all(r['bit_identical'] for r in reg.values())} K ok {all(r['K_ok'] and r['covers_K'] for r in reg.values())}"
                f" | feed≥K (finish tail) {r0['feed_ge_K']} (live tail) {reg['livetail']['feed_ge_K']}"
                f" | wait {[r['feed_ge_k_received'] for r in reg.values()]}")
        res[dt] = dict(cases=rows, summary=summarize({k: v for k, v in rows.items() if "feats" in v}))
        log(f"[edge] {dt} 요약: {json.dumps(res[dt]['summary'])}")
        del m
        free()
    return res


# ------------------------------------------------------------------ (c) 시간
def pct(ts):
    return dict(p50=round(float(np.percentile(ts, 50)), 2), p95=round(float(np.percentile(ts, 95)), 2),
                max=round(float(np.max(ts)), 2), n=len(ts))


def speech(seconds):
    clips = [read_wav(p) for p in sorted(glob.glob(a_wavs))] if glob.glob(a_wavs) else []
    x = np.concatenate(clips) if clips else (np.random.default_rng(0).standard_normal(16000) * 0.05).astype(np.float32)
    reps = int(np.ceil(seconds * 16000 / len(x)))
    return np.tile(x, reps)[: int(seconds * 16000)]


def sec_timing(a, cfg, R):
    res = {}
    x = speech(a.steps * 0.32 + 2)
    # mel 비용 (numpy, 320 ms 당)
    m0 = MelStream(np.hanning(400).astype(np.float32), np.abs(np.random.default_rng(0).standard_normal((128, 257))).astype(np.float32))
    tm, pos = [], 0
    m0.push(x[:4096]); m0.take(m0.ready())
    for pos in range(4096, len(x) - 5120, 5120):
        t0 = time.perf_counter(); m0.push(x[pos:pos + 5120]); m0.take(m0.ready()); tm.append((time.perf_counter() - t0) * 1e3)
    res["mel_ms_per_320ms"] = pct(tm)
    log("[timing] mel per 320 ms", res["mel_ms_per_320ms"])
    for dt in a.dtypes:
        free()
        mx.reset_peak_memory()
        m0 = mx.get_active_memory()
        m, ms = build(a.ckpt, cfg, R, dt)
        r = dict(build_ms=round(ms), model_gb=round((mx.get_active_memory() - m0) / 1e9, 2),
                 build_peak_gb=round((mx.get_peak_memory() - m0) / 1e9, 2))
        t0 = time.perf_counter(); m.warmup(); r["warmup_ms"] = round((time.perf_counter() - t0) * 1e3)
        for comp in (True, False):
            m.compile = comp
            st = m.new_stream()
            first = st.feed(x[:4096]); first_ms = st.last_step_ms
            steps, feeds = [], []
            for c in range(1, a.steps + 1):
                lo, hi = st.chunk_samples(c - 1), st.chunk_samples(c)
                t0 = time.perf_counter(); o = st.feed(x[lo:hi]); feeds.append((time.perf_counter() - t0) * 1e3)
                assert o.shape[0] == m.Tm
                steps.append(st.last_step_ms)
            fin = []
            for r_ in (8, 16, 24, 31):                       # 끝의 짧은 스텝 (eager), 남은 mel r 프레임
                s2 = m.new_stream()
                n = 160 * (25 + 32 * 3 + r_) + 100
                s2.feed(x[:n - 1600]); s2.finish(1600)
                fin.append(s2.step_ms[-1])
            key = "compiled" if comp else "eager"
            r[key] = dict(step=pct(steps[3:]), feed_wall=pct(feeds[3:]), first_step_ms=round(first_ms, 2),
                          first_steady_ms=round(steps[0], 2), final_short_steps_ms=[round(v, 2) for v in fin])
            log(f"[timing] {dt} {key}: step p50 {r[key]['step']['p50']} p95 {r[key]['step']['p95']} ms | feed wall p50 "
                f"{r[key]['feed_wall']['p50']} | first {first_ms:.1f} | final-short {r[key]['final_short_steps_ms']}")
        m.compile = True
        r["run_peak_gb"] = round((mx.get_peak_memory() - m0) / 1e9, 2)
        res[dt] = r
        log(f"[timing] {dt} build {r['build_ms']} ms, warmup {r['warmup_ms']} ms, model {r['model_gb']} GB, "
            f"build peak {r['build_peak_gb']} GB, run peak {r['run_peak_gb']} GB")
        del m, st, s2
        free()
    return res


# ------------------------------------------------------------------ 정밀도 (실제 가중치, 골든 없이)
def sec_precision(a, cfg, R):
    wavs = sorted(glob.glob(a_wavs))
    res = dict(stats={}, cases={})
    m32, _ = build(a.ckpt, cfg, R, "fp32")
    ref = {}
    stats = {}

    def probe(name, li, arr):
        v = float(mx.max(mx.abs(arr)).item())
        k = f"{name}"
        stats.setdefault(k, np.zeros(m32.NL))[li] = max(stats.get(k, np.zeros(m32.NL))[li], v)

    for p in wavs:
        x = read_wav(p)
        ref[p] = run_stream(m32, x, len(x), 16000)[0]
    m32.probe = probe
    run_stream(m32, read_wav(wavs[0]), len(read_wav(wavs[0])), 16000)
    m32.probe = None
    res["stats"] = {k: dict(max=round(float(v.max()), 2), argmax_layer=int(v.argmax()), per_layer=[round(float(t), 1) for t in v])
                    for k, v in stats.items()}
    log("[precision] max |activation| (fp32, " + os.path.basename(wavs[0]) + "): "
        + ", ".join(f"{k} {v['max']} (L{v['argmax_layer']})" for k, v in res["stats"].items()))
    del m32
    free()
    for dt in [d for d in a.dtypes if d != "fp32"] or ["fp16"]:
        m, _ = build(a.ckpt, cfg, R, dt)
        for p in wavs:
            x = read_wav(p)
            y = run_stream(m, x, len(x), 16000)[0]
            c = cmp(y, ref[p])
            res["cases"][f"{dt}|{os.path.basename(p)}"] = c
            log(f"[precision] {dt} {os.path.basename(p)} K={len(y)} max {c['max']:.3g} mean {c['mean']:.3g} rel {c['rel_l2']:.3g} "
                f"cos {c['min_cos']:.6f} finite {c['finite']}")
        del m
        free()
    return res


def main():
    global a_wavs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections", nargs="+", choices=["nemo", "golden", "edge", "timing", "precision"])
    ap.add_argument("--cfg", default=os.path.join(APP, "assets", "nemotron_encoder.json"))
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--randw", default=RANDW)
    ap.add_argument("--probe-npz", default=PROBE_NPZ)
    ap.add_argument("--gold", default=GOLD)
    ap.add_argument("--edge", default=EDGE)
    ap.add_argument("--wavs", default=WAVS)
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--right-context", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--json", default=None, help="결과 JSON 경로")
    a = ap.parse_args()
    a.dtypes = a.dtypes.split(",")
    a_wavs = a.wavs
    cfg = load_cfg(a.cfg)
    out = dict(mlx=mx.__version__, device=str(mx.default_device()), dtypes=a.dtypes)
    fns = dict(nemo=sec_nemo, golden=sec_golden, edge=sec_edge, timing=sec_timing, precision=sec_precision)
    t0 = time.time()
    for s in a.sections:
        out[s] = fns[s](a, cfg, a.right_context)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(out, f, indent=1)
    out["total_s"] = round(time.time() - t0, 1)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    log("done", out["total_s"], "s")


a_wavs = WAVS
if __name__ == "__main__":
    main()
