"""엔진 타이밍 측정.

파일 모드: wav → (실시간 속도 또는 최대 속도로) 20 ms 패킷 feed → 라운드별 dec_ms, 청크별 enc+dec ms p50/p95/max
  (라운드 귀속 합 chunk_ms 와 청크를 끝낸 feed() 호출의 벽시계 chunk_wall_ms), RTF, 지연, (--final) final 재디코드 시간.
  python tools/bench.py --wav x.wav --src English [--tgt German] [--delay 6] [--precision fp16] [--pace 1|0] [--json out.json]
  가속 경로 끄기 (A/B): --draft off|q4|auto, --no-two-stage, --no-prefix-cache, --final-lookahead 0.
  chunk_ms·RTF 는 Round 귀속 합 — 세그먼트 시작 prefix 시간(캐시 적중이면 ≈ 1 ms)은 첫 Round 의 dec_ms 에 들어 있다.
마이크로 모드: thinker forward 한 번의 시간 (q=1, q=2, 투기 q=8 sel 4행, 4×q=2, 검증 q=5, q4 초안 한 스텝) × 문맥 길이 × 정밀도.
  python tools/bench.py --micro [--precisions fp32,fp16] [--contexts 64,512,1024] [--iters 30]
무거운 작업이므로 lockf 로 감싸 실행 (app-design §5)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, np.float64), p)) if len(xs) else None


def stat(xs) -> dict:
    return dict(n=len(xs), p50=round(pct(xs, 50), 2) if xs else None, p95=round(pct(xs, 95), 2) if xs else None,
                max=round(max(xs), 2) if xs else None, mean=round(float(np.mean(xs)), 2) if xs else None)


def read_wav(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except ImportError:
        import wave
        with wave.open(path) as w:
            sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            assert sw == 2, "16-bit PCM wav 만 지원 (soundfile 없음)"
            x = np.frombuffer(w.readframes(n), np.int16).reshape(-1, ch).mean(axis=1).astype(np.float32) / 32768
    if sr != 16000:
        raise SystemExit(f"16 kHz wav 가 필요합니다 ({sr} Hz)")
    return x.astype(np.float32)


def run_file(args) -> dict:
    from engine import Engine
    t0 = time.perf_counter()
    eng = Engine(args.model, kv_capacity=args.kv, precision=args.precision, encoder_precision=args.encoder_precision,
                 speculative=not args.no_speculative, **engine_kw(args))
    load_ms = (time.perf_counter() - t0) * 1000
    wu = eng.warmup() if not args.no_warmup else {}
    wav = read_wav(args.wav)
    pk = int(16 * args.packet_ms)
    res_all = []
    for rep in range(args.repeat):
        seg = eng.new_segment(args.src, args.tgt, args.delay or (eng.defaults["asr_delay"] if not args.tgt else
                                                                 eng.defaults["translate_delay"][f"{args.src}>{args.tgt}"]))
        rounds, arrive, walls = [], [], []                        # arrive[i] = 샘플 i·pk 까지 도착한 벽시계
        t_start = time.perf_counter()
        for i in range(0, len(wav), pk):
            if args.pace > 0:                                     # 실시간 속도: 패킷 끝 시각까지 기다린다
                due = t_start + (i + pk) / 16000 / args.pace
                while time.perf_counter() < due:
                    time.sleep(min(0.002, max(0.0, due - time.perf_counter())))
            arrive.append((i + min(pk, len(wav) - i), time.perf_counter()))
            tc = time.perf_counter()
            rs = seg.feed(wav[i:i + pk])
            if rs:                                                # 이 호출이 인코더 청크를 끝냈다: 청크 enc+dec 벽시계
                walls.append((time.perf_counter() - tc) * 1000)
            for r in rs:
                rounds.append((r, time.perf_counter()))
        t_end = time.perf_counter()
        for r in seg.finish(args.tail):
            rounds.append((r, time.perf_counter()))
        t_fin = time.perf_counter()
        fin = None
        if args.final:                                            # final 재디코드 (블로킹으로 잼)
            ids_f, ms_f = seg.final_redecode()
            fin = dict(ms=round(ms_f, 1), tokens=len(ids_f), lookahead=eng.final_lookahead if hasattr(eng, "final_lookahead") else 0)
        n_total = seg.samples_fed
        arr_n = np.array([a for a, _ in arrive])
        arr_t = np.array([t for _, t in arrive])
        lags = []
        for r, tw in rounds:
            if r.is_tail or r.is_flush:
                ref = t_end                                       # 꼬리·flush 는 끝 결정 시각 기준
            else:
                j = int(np.searchsorted(arr_n, r.need_samples))
                ref = arr_t[min(j, len(arr_t) - 1)]
            lags.append((tw - ref) * 1000)
        chunks = {}
        for r, _ in rounds:
            if not r.is_flush:
                c = chunks.setdefault(r.k // 4, [0.0, 0.0])
                c[0] += r.enc_ms
                c[1] += r.dec_ms
        chunk_ms = [a + b for a, b in chunks.values()]
        comp = sum(r.enc_ms + r.dec_ms for r, _ in rounds)
        runs, cur = ([], []), 0                                   # 전사/번역 run 분리 (split_runs 규칙)
        for _, t in seg.out:
            if t in eng.spk_ids:
                cur = 0
            elif t == eng.trans_id:
                cur = 1
            else:
                runs[cur].append(t)
        res_all.append(dict(
            rep=rep, audio_s=round(len(wav) / 16000, 3), total_s=round(n_total / 16000, 3), K=seg.K, rounds=len(rounds),
            tokens=len(seg.out), forced=seg.forced, flush_rounds=seg.flush_rounds, forwards=seg.stats["forwards"],
            prefix_ms=round(seg.stats["prefix_ms"], 2), prefix_cached=seg.stats.get("prefix_cached"),
            drafted=seg.stats.get("drafted"), accepted=seg.stats.get("accepted"), res_used=seg.stats.get("res_used"),
            dec_ms=stat([r.dec_ms for r, _ in rounds]), dec_ms_emitting=stat([r.dec_ms for r, _ in rounds if r.ids]),
            enc_ms_chunk=stat([r.enc_ms for r, _ in rounds if r.enc_ms > 0]), chunk_ms=stat(chunk_ms),
            rtf=round(comp / 1000 / (n_total / 16000), 4), wall_s=round(t_fin - t_start, 3), finalize_ms=round((t_fin - t_end) * 1000, 1),
            lag_ms=stat(lags), over_budget_chunks=sum(c > 320 for c in chunk_ms),
            chunk_wall_ms=stat(walls), over_budget_wall=sum(c > 320 for c in walls), final=fin,
            src_text=eng.tokens.decode(runs[0]), tr_text=eng.tokens.decode(runs[1])))
        r0 = res_all[-1]
        print(f"[{rep}] {args.wav} {eng.info['precision']}/{eng.info['encoder_precision']} (요청 {args.precision}/{args.encoder_precision}) "
              f"spec={not args.no_speculative} 초안={'on' if getattr(eng, 'draft_on', False) else 'off'} "
              f"두단계헤드={eng.info.get('two_stage')} pace={args.pace} | "
              f"K {r0['K']} tok {r0['tokens']} fwd {r0['forwards']} | dec p50 {r0['dec_ms']['p50']} p95 {r0['dec_ms']['p95']} max {r0['dec_ms']['max']} | "
              f"enc/청크 p50 {r0['enc_ms_chunk']['p50']} | 청크 p50 {r0['chunk_ms']['p50']} p95 {r0['chunk_ms']['p95']} max {r0['chunk_ms']['max']} | "
              f"청크 벽시계 p50 {r0['chunk_wall_ms']['p50']} p95 {r0['chunk_wall_ms']['p95']} max {r0['chunk_wall_ms']['max']} | "
              f"RTF {r0['rtf']} | lag p50 {r0['lag_ms']['p50']} p95 {r0['lag_ms']['p95']} | finalize {r0['finalize_ms']} ms"
              + (f" | final {fin['ms']} ms ({fin['tokens']} tok)" if fin else ""), flush=True)
    return dict(mode="file", info=eng.info, load_ms=round(load_ms, 1), warmup=wu, args=vars(args), runs=res_all)


def engine_kw(args) -> dict:
    return dict(draft=args.draft, two_stage_head=not args.no_two_stage, prefix_cache=0 if args.no_prefix_cache else 16,
                final_lookahead=args.final_lookahead)


def run_micro(args) -> dict:
    import mlx.core as mx
    from engine import Engine
    out = dict(mode="micro", args=vars(args), results={})
    for prec in args.precisions.split(","):
        t0 = time.perf_counter()
        eng = Engine(args.model, kv_capacity=2048, precision=prec, encoder=False, **dict(engine_kw(args), draft="q4"))
        load_s = time.perf_counter() - t0
        th = eng.thinker
        rng = np.random.default_rng(0)
        rnd = lambda n: mx.array((rng.standard_normal((n, th.D)) * 0.05).astype(np.float32)).astype(th.in_dtype)
        ids = list(rng.integers(0, 151643, 4))

        def timeit(f, warm=5):
            for _ in range(warm):
                f()
            ts = []
            for _ in range(args.iters):
                t = time.perf_counter()
                f()
                ts.append((time.perf_counter() - t) * 1000)
            return round(statistics.median(ts), 2), round(pct(ts, 90), 2)

        r = dict(load_s=round(load_s, 1), info=eng.info)
        for C in [int(c) for c in args.contexts.split(",")]:
            th.forward(rnd(C), 0, sel=[])
            x1, x2, x7, x8, x20 = rnd(1), rnd(2), rnd(7), rnd(8), rnd(20)
            r[f"C{C}"] = dict(
                q1=timeit(lambda: th.forward(x1, C)),
                q1_embed=timeit(lambda: th.forward(th.embed(ids[:1]), C)),
                q2=timeit(lambda: th.forward(x2, C, [1])),
                spec_q7=timeit(lambda: th.forward(x7, C, [0, 2, 4, 6])),
                spec_q8=timeit(lambda: th.forward(x8, C, [1, 3, 5, 7])),
                seq_4xq2=timeit(lambda: [th.forward(x2, C + 2 * j, [1]) for j in range(4)]),
                verify_q5=timeit(lambda: th.forward(x8[:5], C, list(range(5)))),
                prefix20=timeit(lambda: th.forward(x20, 0, sel=[]), warm=2))
            if eng.draft is not None:
                r[f"C{C}"]["draft_q4_step"] = timeit(lambda: eng.draft.ids_lazy(x1, C).item())
            print(f"[{prec}] C={C}: " + ", ".join(f"{k} {v[0]} ms" for k, v in r[f"C{C}"].items()), flush=True)
        # final 모드 prefill (K≈200 → 약 225 위치)
        kv = th.new_kv(512)
        xp = rnd(225)
        r["final_prefill_225"] = timeit(lambda: th.forward(xp, 0, kv=kv), warm=2)
        a = eng.adapter
        f4 = mx.array(rng.standard_normal((4, 1024)).astype(np.float32))
        r["adapter_4"] = timeit(lambda: mx.eval(a(f4)))
        print(f"[{prec}] final prefill 225: {r['final_prefill_225'][0]} ms, adapter 4 프레임 {r['adapter_4'][0]} ms, 적재 {load_s:.1f}s",
              flush=True)
        out["results"][prec] = r
        eng.close()
        del eng, th
        gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--precisions", default="fp32,fp16")
    ap.add_argument("--contexts", default="64,512,1024")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--wav")
    ap.add_argument("--src", default="English")
    ap.add_argument("--tgt", default=None)
    ap.add_argument("--delay", type=int, default=None)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--encoder-precision", default="auto")
    ap.add_argument("--no-speculative", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--pace", type=float, default=1.0, help="1 = 실시간, 0 = 최대 속도")
    ap.add_argument("--packet-ms", type=int, default=20)
    ap.add_argument("--tail", type=int, default=16000)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--final", action="store_true", help="세그먼트 끝에 final 재디코드 시간도 잰다")
    ap.add_argument("--kv", type=int, default=2048)
    ap.add_argument("--draft", default="auto", choices=("auto", "q4", "off"), help="q4 자기 초안 (텍스트 토큰 검증 디코드)")
    ap.add_argument("--no-two-stage", action="store_true", help="fp32 lm_head 두 단계(인증서) 끄기")
    ap.add_argument("--no-prefix-cache", action="store_true", help="prefix KV 캐시 끄기")
    ap.add_argument("--final-lookahead", type=int, default=None, help="final 재디코드 한 걸음의 초안 수 (0 = q=1, 기본: fp32 thinker 면 11)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.micro and not args.wav:
        ap.error("--wav 또는 --micro")
    res = run_micro(args) if args.micro else run_file(args)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=str)
        print("JSON:", args.json)


if __name__ == "__main__":
    main()
