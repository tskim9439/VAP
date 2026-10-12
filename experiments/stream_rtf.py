#!/usr/bin/env python3
"""우리 스트리밍 모델의 실시간성(RTF·청크별 계산 시간)과 계산 시간을 넣은 지연을 한 스트림(batch 1) 기준으로 잰다 (2026-10-12).

평가(eval_single_turn_asr.py run)와 같은 입력(원 발화 + 꼬리 무음 1 s)·같은 prefix(화자·언어·번역 줄 + <DELAY_δ>)·같은 직렬 디코드
(VapAsrForStreamingASR.stream_decode, 80 ms step 마다 torch.cuda.synchronize 후 계산 시간을 ms 로 돌려준다)를 쓴다.

실시간 조건: 인코더(Nemotron, att_context [56,3], chunked_limited)는 4 프레임 = 320 ms 를 한 청크로 처리하므로, 청크 c 의 오디오가 다 들어온
뒤(min(4(c+1), K)·80 ms) 인코더 한 청크 + 디코더 4 step 을 계산한다. 청크 하나의 계산이 320 ms 안에 끝나면 밀리지 않는다.
- 인코더 비용: 평가처럼 파일 전체를 한 번에 인코딩한 시간을 청크 수로 나눈 값(캐시 재사용 스트리밍에 가까운 하한)과,
  캐시 없이 청크마다 왼쪽 문맥 56 + 청크 4 프레임(4.8 s)을 다시 인코딩하는 시간(상한)을 둘 다 잰다.
- RTF = (인코더 + 디코더 계산 시간) / 처리한 오디오 길이(꼬리 포함).
- 지연(번역 단어, LAAL s, |X| = 원 발화):
  eval  = 평가와 같은 (k+1)·80 ms
  enc   = 인코더 청크가 다 들어온 시각(simulst_common.encoder_chunk_times) — 계산 시간 제외
  ca_lo = 계산 시간 포함(청크가 들어온 뒤 대기열로 처리, 인코더 하한 비용) / ca_hi = 같은데 인코더 상한 비용
  입력이 끝난 뒤 flush 라운드의 시간은 (디코드 전체 − step 합)을 라운드 수로 나눠 붙인다(prefix step 이 섞여 조금 크게 잡힌다).

usage(vapasr env, GPU 한 장):
  python experiments/stream_rtf.py --checkpoint CK --encoder NEMOTRON --manifest M --out DIR [--per-set 24] [--deltas 4 10] [--dtype float32]
산출: DIR/rows-<dtype>.jsonl(문장마다) + DIR/summary-<dtype>.json(세트·δ 마다)
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulst_common as C  # noqa: E402

ENC_CHUNK = 4                       # [56,3] chunked_limited → 4 프레임 청크
WINDOW_FRAMES = 56 + ENC_CHUNK      # 캐시 없이 다시 인코딩할 창(왼쪽 문맥 + 청크)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))] if xs else None


def pick_rows(manifest, per_set):
    """번역 행만, 세트마다 길이 순으로 고르게 per_set 개(eval_single_turn_asr --limit 과 같은 방식)."""
    import numpy as np
    rows = C.load_rows([manifest]); out = []
    for name in sorted({r["dataset"] for r in rows}):
        rr = sorted((r for r in rows if r["dataset"] == name), key=lambda r: r["audio_s"])
        out += [rr[i] for i in np.linspace(0, len(rr) - 1, min(len(rr), per_set), dtype=int)]
    return out


def queue_times(K, ticks, enc_chunk_ms, rounds, flush_ms):
    """청크가 들어온 뒤 대기열로 처리할 때 step k 출력이 나오는 시각(s): k < K 는 그 step 까지의 계산이 끝난 때, k ≥ K(flush)는 마지막 청크 뒤
    라운드마다 flush_ms/rounds 씩. 반환 (step 시각 dict, 청크별 계산 ms 목록, 마지막 청크가 끝난 때 − 입력 끝)."""
    t_step, chunk_ms, finish = {}, [], 0.0
    for c in range((K + ENC_CHUNK - 1) // ENC_CHUNK):
        arrive = min(ENC_CHUNK * (c + 1), K) * C.FRAME_S
        start = max(arrive, finish); acc = enc_chunk_ms
        for k in range(ENC_CHUNK * c, min(ENC_CHUNK * (c + 1), K)):
            acc += ticks[k]; t_step[k] = start + acc / 1000
        chunk_ms.append(acc); finish = start + acc / 1000
    per = flush_ms / max(1, rounds)
    for r in range(rounds):
        t_step[K + r] = finish + per * (r + 1) / 1000
    return t_step, chunk_ms, finish - K * C.FRAME_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-set", type=int, default=24)
    ap.add_argument("--deltas", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--tail-s", type=float, default=1.0)
    ap.add_argument("--max-flush", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    import torch
    from eval_single_turn_asr import read_audio
    from vapasr.hf.infer import load_model
    from vapasr.data.ast_dataset import ast_head_text
    from vapasr.data.ast_eval import split_runs, word_emission_times, laal
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, tok = load_model(a.checkpoint, encoder_path=a.encoder, dtype=getattr(torch, a.dtype))
    ctx = list(map(int, model.encoder.enc.att_context_size)); style = getattr(model.encoder.enc, "att_context_style", None)
    assert ctx == [56, ENC_CHUNK - 1] and style == "chunked_limited", (ctx, style)
    dev = next(model.parameters()).device
    spk = [model.config.sp_ids["<SPK_A>"], model.config.sp_ids["<SPK_B>"]]; trans_id = model.config.sp_ids["<TRANS>"]
    rows = pick_rows(a.manifest, a.per_set)
    os.makedirs(a.out, exist_ok=True)
    print(f"START {len(rows)} 행 × δ{a.deltas} · {a.dtype} · {torch.cuda.get_device_name()} · att_context {ctx} {style}", flush=True)

    def sync_ms(fn):
        torch.cuda.synchronize(); t = time.perf_counter(); out = fn(); torch.cuda.synchronize(); return out, (time.perf_counter() - t) * 1000

    def encode(wav):
        w = torch.from_numpy(wav).to(dev); K = int(round(len(wav) / C.SR * model.config.frame_hz))
        return model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0]

    def prefix(r, d):
        return (list(model.config.prefix_ids) + tok(ast_head_text(r["lang"], "<SPK_A>", r["tgt_lang"]), add_special_tokens=False)["input_ids"]
                + [model.config.sp_ids[f"<DELAY_{d}>"]])

    with torch.inference_mode():
        w0 = read_audio(rows[0], a.tail_s)
        for _ in range(a.warmup):                                                       # cuDNN·할당기·커널 준비
            model.stream_decode(encode(w0), prefix(rows[0], a.deltas[0]), max_flush_rounds=a.max_flush)
        win = w0[: int(WINDOW_FRAMES * C.FRAME_S * C.SR)]
        if len(win) < WINDOW_FRAMES * C.FRAME_S * C.SR:
            import numpy as np
            win = np.pad(win, (0, int(WINDOW_FRAMES * C.FRAME_S * C.SR) - len(win)))
        enc_win = sorted(sync_ms(lambda: encode(win))[1] for _ in range(30))
        enc_win_ms = statistics.median(enc_win)
        print(f"인코더 창 {WINDOW_FRAMES * C.FRAME_S:.2f} s 재인코딩: 중앙 {enc_win_ms:.1f} ms · p95 {pct(enc_win, 95):.1f} ms", flush=True)
        dest = open(os.path.join(a.out, f"rows-{a.dtype}.jsonl"), "w", encoding="utf-8"); recs = []
        for i, r in enumerate(rows):
            wav = read_audio(r, a.tail_s)
            feats, enc_ms = sync_ms(lambda: encode(wav))
            K = int(feats.shape[1]); n_chunks = (K + ENC_CHUNK - 1) // ENC_CHUNK
            for d in a.deltas:
                (out, forced, ticks, rounds), dec_ms = sync_ms(lambda: model.stream_decode(feats, prefix(r, d), max_flush_rounds=a.max_flush))
                flush_ms = max(0.0, dec_ms - sum(ticks))
                _, tr = split_runs(out, spk, trans_id)
                n_ref = len(r["translation_reference"].split())
                lat = {}
                for name, enc_chunk_ms in (("ca_lo", enc_ms / n_chunks), ("ca_hi", enc_win_ms)):
                    t_step, chunk_ms, backlog = queue_times(K, ticks, enc_chunk_ms, rounds, flush_ms)
                    tr_t = [(t_step[k] / C.FRAME_S - 1, tid) for k, tid in tr]
                    lat[name] = laal(word_emission_times(tok, tr_t)[1], r["audio_s"], n_ref) if tr else None
                    lat[f"{name}_chunk_ms"] = chunk_ms; lat[f"{name}_backlog_s"] = backlog
                text, times = word_emission_times(tok, tr) if tr else ("", [])
                rec = dict(id=r["id"], dataset=r["dataset"], delta=d, dtype=a.dtype, audio_s=r["audio_s"], input_s=len(wav) / C.SR, K=K,
                           tokens=len(out), trans_words=len(times), enc_full_ms=enc_ms, enc_window_ms=enc_win_ms, dec_ms=dec_ms, ticks_ms=ticks,
                           flush_rounds=rounds, flush_ms=flush_ms,
                           rtf_lo=(enc_ms + dec_ms) / 1000 / (len(wav) / C.SR), rtf_hi=(enc_win_ms * n_chunks + dec_ms) / 1000 / (len(wav) / C.SR),
                           laal_eval=laal(times, r["audio_s"], n_ref) if times else None,
                           laal_enc=laal(C.encoder_chunk_times(times, K, ENC_CHUNK), r["audio_s"], n_ref) if times else None,
                           laal_ca_lo=lat["ca_lo"], laal_ca_hi=lat["ca_hi"], chunk_ms_lo=lat["ca_lo_chunk_ms"], chunk_ms_hi=lat["ca_hi_chunk_ms"],
                           backlog_lo_s=lat["ca_lo_backlog_s"], backlog_hi_s=lat["ca_hi_backlog_s"], translation=text)
                dest.write(json.dumps(rec, ensure_ascii=False) + "\n"); dest.flush(); recs.append(rec)
            if i % 10 == 0:
                print(f"PROGRESS {i + 1}/{len(rows)} {r['dataset']} {r['audio_s']:.1f}s rtf_lo {recs[-1]['rtf_lo']:.3f} "
                      f"chunk p95 {pct(recs[-1]['chunk_ms_lo'], 95):.0f} ms", flush=True)
    summ = {}
    for key in sorted({(x["dataset"], x["delta"]) for x in recs}):
        xs = [x for x in recs if (x["dataset"], x["delta"]) == key]
        ticks = [t for x in xs for t in x["ticks_ms"]]; lo = [t for x in xs for t in x["chunk_ms_lo"]]; hi = [t for x in xs for t in x["chunk_ms_hi"]]
        mean = lambda f: statistics.mean(v for x in xs if (v := x[f]) is not None)
        summ[f"{key[0]}/d{key[1]}"] = dict(
            n=len(xs), audio_s=sum(x["audio_s"] for x in xs),
            rtf_lo=sum(x["enc_full_ms"] + x["dec_ms"] for x in xs) / 1000 / sum(x["input_s"] for x in xs),
            rtf_hi=sum(x["enc_window_ms"] * len(x["chunk_ms_hi"]) + x["dec_ms"] for x in xs) / 1000 / sum(x["input_s"] for x in xs),
            step_ms=dict(p50=pct(ticks, 50), p95=pct(ticks, 95), p99=pct(ticks, 99), max=max(ticks)),
            chunk_ms_lo=dict(p50=pct(lo, 50), p95=pct(lo, 95), p99=pct(lo, 99), max=max(lo), over_320=sum(t > 320 for t in lo) / len(lo)),
            chunk_ms_hi=dict(p50=pct(hi, 50), p95=pct(hi, 95), p99=pct(hi, 99), max=max(hi), over_320=sum(t > 320 for t in hi) / len(hi)),
            backlog_lo_s_max=max(x["backlog_lo_s"] for x in xs), backlog_hi_s_max=max(x["backlog_hi_s"] for x in xs),
            flush_ms_mean=statistics.mean(x["flush_ms"] for x in xs),
            laal_eval=mean("laal_eval"), laal_enc=mean("laal_enc"), laal_ca_lo=mean("laal_ca_lo"), laal_ca_hi=mean("laal_ca_hi"))
    meta = dict(checkpoint=a.checkpoint, manifest=a.manifest, dtype=a.dtype, gpu=torch.cuda.get_device_name(), deltas=a.deltas, per_set=a.per_set,
                tail_s=a.tail_s, enc_window_ms=dict(p50=enc_win_ms, p95=pct(enc_win, 95)), att_context=ctx, att_context_style=style,
                created=time.strftime("%F %T"))
    json.dump(dict(meta=meta, groups=summ), open(os.path.join(a.out, f"summary-{a.dtype}.json"), "w"), ensure_ascii=False, indent=1)
    for k, v in summ.items():
        print(f"{k:22s} RTF {v['rtf_lo']:.3f}–{v['rtf_hi']:.3f} · step p50/p95/max {v['step_ms']['p50']:.0f}/{v['step_ms']['p95']:.0f}/{v['step_ms']['max']:.0f} ms · "
              f"320ms 청크 p95/max {v['chunk_ms_lo']['p95']:.0f}/{v['chunk_ms_lo']['max']:.0f} (상한 {v['chunk_ms_hi']['p95']:.0f}/{v['chunk_ms_hi']['max']:.0f}) ms · "
              f"초과 {v['chunk_ms_lo']['over_320']:.1%}/{v['chunk_ms_hi']['over_320']:.1%} · LAAL eval {v['laal_eval']:.2f} enc {v['laal_enc']:.2f} "
              f"ca {v['laal_ca_lo']:.2f}–{v['laal_ca_hi']:.2f}", flush=True)


if __name__ == "__main__":
    main()
