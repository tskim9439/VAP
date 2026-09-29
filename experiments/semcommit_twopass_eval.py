#!/usr/bin/env python3
"""<SEM_END> 트리거 2-pass 최종 재디코드 평가(최소판) — 계획 wiki/outputs/output-semcommit-two-pass-plan.md, 세그먼트 규칙은 vapasr/hf/twopass.py.

1차 = semcommit_eval 이 이미 디코드한 스트림 jsonl(설정 하나, 기본 bias=0)을 그대로 쓴다(재디코드 없음). 오디오는 1차와 같은 조립
(semcommit_eval.load_stream_audio = streams.assemble_stream, 같은 --path-remap). 스트림마다 SEM(soft)·EOS(hard) 트리거로 세그먼트를 자르고
각 세그먼트를 Qwen3-ASR(qwen_asr transformers 백엔드, bf16, 언어 강제)로 다시 디코드해 최종 전사를 만든다.
  모드 iso  : 세그먼트만(context 없음)
       text : 직전까지의 최종 텍스트(끝 --context-chars 글자)를 Qwen3-ASR system context 로
       s5   : 스트림 전체를 한 번에(오프라인 상한 대조; 트리거 없음, 모든 단어가 EOS 에 확정)
  0.5 s 보다 짧은 입력은 꼬리에 0 을 붙인다(qwen_asr MIN_ASR_INPUT_SECONDS — transcribe() 는 1200 s 이하 입력에서 split 전에 돌아와 이 패딩을 건너뛴다).
채점: commit_metrics.asr_counts(= semcommit_eval 보고서 asr 블록·single_turn_eval.score_pair 와 같은 변형·카운트; 참조 = words 행 text).
  1차(스트림 행 hyp.text) · 2-pass · s5 를 같은 스트림에서 나란히, 세트·모드별 micro 오류율(EN wer, KO cer_nospace 주 지표).
지연(algorithmic): 참조 단어별 최종 확정 지연(twopass.final_latencies) vs 1차 표시 지연(twopass.display_latencies). 분당 세그먼트 수, 2차 계산 시간(벽시계, 배치 단위를
  오디오 길이 비례로 나눔)·RTF. 1차 스트림 행에 hyp.word_k 가 없으면(88220f7 디코드) semcommit_eval.backfill_word_k 로 메모리에서 채운다(파일은 안 바꾼다).
출력: <out-dir>/<set>-d<δ>-<decoder tag>.jsonl(스트림 × 모드 행, 이어하기) + .config.json(지문 — 다르면 멈춘다) + summary-d<δ>-<decoder tag>.json.

  python experiments/semcommit_twopass_eval.py --eval-dir <runs/semcommit/eval/v035-snap0929-d8> --words-dir <gold/v1> --sets ks-eval ks-long ls-test gs-test \\
      --delay 4 --decoder /soundai/Model/Qwen3-ASR-1.7B --modes iso text s5 --gpu 0 --out-dir <eval-dir>/twopass \\
      --path-remap /data5/LibriSpeech/=<...>/LibriSpeech/ ...
  --score-only: GPU 없이 기존 jsonl 로 요약만 다시 낸다.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __name__ == "__main__" and "--gpu" in sys.argv[:-1]:                                     # torch import 전에 GPU 고정(semcommit_eval 과 같은 방식)
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vapasr.hf.commit_metrics import SEM_END_ID, asr_counts, merge
from vapasr.hf.twopass import build_segments, compose, display_latencies, final_latencies, latency_summary, segment_stats

SR = 16000
MIN_IN_S = 0.5                                                                                 # qwen_asr MIN_ASR_INPUT_SECONDS
PROTOCOL = "semcommit-twopass-v1"
CODE_FILES = ("experiments/semcommit_twopass_eval.py", "vapasr/hf/twopass.py", "vapasr/hf/commit_metrics.py", "experiments/semcommit_eval.py", "vapasr/data/streams.py")
LANG = {"English": "English", "Korean": "Korean"}


def _semcommit_eval():
    spec = importlib.util.spec_from_file_location("semcommit_eval_mod", ROOT / "experiments" / "semcommit_eval.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def primary(lang): return "cer_nospace" if lang == "Korean" else "wer"


def pad_min(x):
    import numpy as np
    n = int(MIN_IN_S * SR)
    return x if len(x) >= n else np.pad(x, (0, n - len(x))).astype(np.float32)


class Decoder:
    """Qwen3-ASR transformers 백엔드(qwen_asr) — 배치 transcribe + 벽시계 기록."""
    def __init__(self, path, batch_size, max_new_tokens):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.m = Qwen3ASRModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=batch_size, max_new_tokens=max_new_tokens)
        self.bs = batch_size

    def run(self, wavs, langs, ctxs):
        """→ (texts, 입력별 계산 초(배치 벽시계를 오디오 길이 비례로 나눔)). 길이순으로 배치를 묶고 원래 순서로 돌려준다."""
        order = sorted(range(len(wavs)), key=lambda i: -len(wavs[i])); texts, cost = [None] * len(wavs), [0.0] * len(wavs)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]; t0 = time.monotonic()
            res = self.m.transcribe(audio=[(pad_min(wavs[i]), SR) for i in idx], context=[ctxs[i] for i in idx], language=[langs[i] for i in idx])
            dt = time.monotonic() - t0; tot = sum(len(wavs[i]) for i in idx) or 1
            for i, r in zip(idx, res): texts[i] = r.text; cost[i] = dt * len(wavs[i]) / tot
        return texts, cost


def load_set(a, se, s):
    """세트 s: words 행, 1차 스트림 행(설정 하나, words 지문 확인, word_k backfill), 디코드 설정."""
    words = {r["id"]: r for r in se.read_jsonl(Path(a.words_dir) / f"words-{s}.jsonl")}
    sp = Path(a.eval_dir) / f"{s}-d{a.delay}.streams.jsonl"; dc = json.loads(Path(str(sp) + ".config.json").read_text())
    recs = sorted((r for r in se.read_jsonl(sp) if r["config"]["name"] == a.config), key=lambda r: r["id"])
    assert recs, f"{sp}: 설정 {a.config} 행 없음"
    stale = [r["id"] for r in recs if r.get("words_digest") != se.row_digest(words[r["id"]])]
    assert not stale, f"{s}: words 행이 1차 디코드 때와 다름 {stale[:3]}"
    st = se.backfill_word_k(recs, words, dc, a.tokenizer or dc.get("model"))
    assert not st["missing"], f"{s}: word_k backfill 실패 {st}"
    return words, recs, dc, st


def fingerprint(a, s, dc):
    return dict(protocol=PROTOCOL, set=s, delay=a.delay, config=a.config, decoder=os.path.abspath(a.decoder), min_seg_s=a.min_seg, min_words=a.min_words,
                context_chars=a.context_chars, max_new_tokens=a.max_new_tokens, path_remap=list(a.path_remap or []), first_pass_model=dc.get("model"),
                streams_sha256=sha256_file(Path(a.eval_dir) / f"{s}-d{a.delay}.streams.jsonl"),
                code={f: sha256_file(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()})


def run_set(a, se, dec, s):
    words, recs, dc, bst = load_set(a, se, s)
    tag = Path(a.decoder).name; out = Path(a.out_dir) / f"{s}-d{a.delay}-{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(str(out) + ".config.json"); fp = fingerprint(a, s, dc)
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text()); diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
        assert not diff, f"이어하기 지문 불일치 {diff} — 새 --out-dir 을 쓰세요: {cfg_path}"
    else: se.write_json(cfg_path, fp)
    done = {(r["id"], r["mode"]) for r in se.read_jsonl(out)} if out.exists() else set()
    todo = {m: [r for r in recs if (r["id"], m) not in done] for m in a.modes}
    need = sorted({r["id"] for m in a.modes for r in todo[m]})
    print(f"[{s}] streams {len(recs)} · 할 일 {({m: len(v) for m, v in todo.items()})} · word_k backfill {bst}", flush=True)
    if not need: return out
    remaps = se.parse_remaps(a.path_remap); t0 = time.monotonic()
    with ThreadPoolExecutor(a.io_workers) as pool:
        wav = dict(zip(need, pool.map(lambda sid: se.load_stream_audio(words[sid], next(r["duration_s"] for r in recs if r["id"] == sid), remaps), need)))
    print(f"[{s}] 오디오 {len(wav)} 스트림 ({time.monotonic() - t0:.0f}s)", flush=True)
    rows = {}
    for r in recs:
        if r["id"] not in wav: continue
        w = words[r["id"]]; t_eos = float(w["duration_s"]); ws = sorted(w["words"], key=lambda x: int(x["i"]))
        segs = build_segments(r["hyp"], int(r["delta"]), int(r["K"]), t_eos, sem_id=int((r.get("event_ids") or [SEM_END_ID])[0]), min_seg_s=a.min_seg, min_words=a.min_words)
        rows[r["id"]] = dict(r=r, w=w, ws=ws, t_eos=t_eos, segs=segs)
    with out.open("a", encoding="utf-8") as f:
        for mode in a.modes:
            ids = [r["id"] for r in todo[mode] if r["id"] in rows]
            if not ids: continue
            t1 = time.monotonic()
            if mode == "s5":
                xs = [wav[i][:int(round(rows[i]["t_eos"] * SR))] for i in ids]
                texts, cost = dec.run(xs, [LANG[rows[i]["w"]["lang"]] for i in ids], [""] * len(ids))
                res = {i: dict(seg_texts=[t], cost=c, segs=[dict(start=0.0, end=rows[i]["t_eos"], t_trig=rows[i]["t_eos"], kind="eos", w0=0, w1=len(rows[i]["r"]["hyp"]["words"]),
                                                                    k_sem=None, held=0)]) for i, t, c in zip(ids, texts, cost)}
            else:
                res = {i: dict(seg_texts=[None] * len(rows[i]["segs"]), cost=0.0, segs=rows[i]["segs"]) for i in ids}
                if mode == "iso":
                    jobs = [(i, j) for i in ids for j in range(len(rows[i]["segs"]))]
                    xs = [wav[i][int(round(rows[i]["segs"][j]["start"] * SR)):int(round(rows[i]["segs"][j]["end"] * SR))] for i, j in jobs]
                    texts, cost = dec.run(xs, [LANG[rows[i]["w"]["lang"]] for i, _ in jobs], [""] * len(jobs))
                    for (i, j), t, c in zip(jobs, texts, cost): res[i]["seg_texts"][j] = t; res[i]["cost"] += c
                else:                                                                          # text: 세그먼트 순번 단위 wavefront(앞 세그먼트 최종 텍스트가 context)
                    for j in range(max(len(rows[i]["segs"]) for i in ids)):
                        jobs = [i for i in ids if j < len(rows[i]["segs"])]
                        ctx = [compose(res[i]["seg_texts"][:j])[-a.context_chars:] if a.context_chars > 0 else "" for i in jobs]
                        xs = [wav[i][int(round(rows[i]["segs"][j]["start"] * SR)):int(round(rows[i]["segs"][j]["end"] * SR))] for i in jobs]
                        texts, cost = dec.run(xs, [LANG[rows[i]["w"]["lang"]] for i in jobs], ctx)
                        for i, t, c in zip(jobs, texts, cost): res[i]["seg_texts"][j] = t; res[i]["cost"] += c
            for i in ids:
                x = rows[i]; r, w, ws = x["r"], x["w"], x["ws"]; lang = w["lang"]; ref = w.get("text") or " ".join(q["text"] for q in ws)
                ends = [float(q["end_time"]) for q in ws]; final = compose(res[i]["seg_texts"]); segs = res[i]["segs"]
                rec = dict(id=i, set=s, lang=lang, delta=int(r["delta"]), config=a.config, mode=mode, decoder=tag, t_eos=x["t_eos"], K=int(r["K"]),
                           segs=segs, seg_texts=res[i]["seg_texts"], final_text=final, first_text=r["hyp"]["text"],
                           asr_first=asr_counts(ref, r["hyp"]["text"], lang), asr_final=asr_counts(ref, final, lang),
                           lat_final=final_latencies(ends, segs), lat_display=display_latencies([q["text"] for q in ws], ends, r["hyp"]["words"], r["hyp"]["word_k"], int(r["K"])),
                           stats=segment_stats(r["hyp"], segs, int(r["K"]), x["t_eos"]) if mode != "s5" else None,
                           compute_s=round(res[i]["cost"], 4), seg_audio_s=round(sum(float(g["end"]) - float(g["start"]) for g in segs), 3))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{s}] {mode}: {len(ids)} 스트림 {time.monotonic() - t1:.0f}s", flush=True)
    return out


def summarize(a, se, paths):
    """세트 × 모드: 1차·최종 주 지표(micro), 지연 분위수(최종 확정 vs 1차 표시), 분당 세그먼트, 계산 시간·RTF, 트리거 통계."""
    summ = {}
    for p in paths:
        if not Path(p).exists(): continue
        for r in se.read_jsonl(p):
            g = summ.setdefault((r["set"], r["mode"]), dict(lang=r["lang"], streams=0, first=None, final=None, lat_final=[], lat_display=[], n_seg=0, audio_s=0.0,
                                                           compute_s=0.0, seg_audio_s=0.0, stats=None))
            g["streams"] += 1; g["first"] = merge(g["first"], r["asr_first"]); g["final"] = merge(g["final"], r["asr_final"])
            g["lat_final"] += r["lat_final"]; g["lat_display"] += r["lat_display"]; g["n_seg"] += len(r["segs"]); g["audio_s"] += float(r["t_eos"])
            g["compute_s"] += float(r["compute_s"]); g["seg_audio_s"] += float(r["seg_audio_s"])
            if r.get("stats"): g["stats"] = merge(g["stats"], r["stats"])
    out = {}
    for (s, m), g in sorted(summ.items()):
        k = primary(g["lang"]); rate = lambda c: round(c[k]["errors"] / c[k]["n_ref"], 5) if c and c[k]["n_ref"] else None
        out[f"{s}|{m}"] = dict(set=s, mode=m, lang=g["lang"], streams=g["streams"], metric=k, first=rate(g["first"]), final=rate(g["final"]),
                               first_counts=g["first"][k], final_counts=g["final"][k], lat_final=latency_summary(g["lat_final"]), lat_display=latency_summary(g["lat_display"]),
                               seg_per_min=round(g["n_seg"] / (g["audio_s"] / 60), 3) if g["audio_s"] else None, n_seg=g["n_seg"], audio_s=round(g["audio_s"], 2),
                               compute_s=round(g["compute_s"], 2), rtf=round(g["compute_s"] / g["seg_audio_s"], 4) if g["seg_audio_s"] else None, stats=g["stats"])
    return out


def print_summary(summ):
    f = lambda x, n=2: "   -  " if x is None else f"{x:.{n}f}"
    print(f"{'set':8s} {'mode':5s} {'n':>4s} {'metric':>11s} {'1st%':>6s} {'final%':>6s} {'Δrel':>6s} | {'fin p50':>7s} {'p90':>5s} | {'disp p50':>8s} {'p90':>5s} | {'seg/min':>7s} {'RTF':>6s} {'held':>4s}")
    for v in summ.values():
        d = None if v["first"] in (None, 0) or v["final"] is None else (v["final"] / v["first"] - 1) * 100
        st = v.get("stats") or {}
        print(f"{v['set']:8s} {v['mode']:5s} {v['streams']:4d} {v['metric']:>11s} {f(v['first'] * 100 if v['first'] is not None else None):>6s} "
              f"{f(v['final'] * 100 if v['final'] is not None else None):>6s} {f(d, 1):>6s} | {f(v['lat_final']['p50']):>7s} {f(v['lat_final']['p90']):>5s} | "
              f"{f(v['lat_display']['p50']):>8s} {f(v['lat_display']['p90']):>5s} | {f(v['seg_per_min']):>7s} {f(v['rtf'], 3):>6s} {st.get('held', 0):>4}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="semcommit_eval 출력 폴더(<set>-d<δ>.streams.jsonl + .config.json)")
    p.add_argument("--words-dir", required=True, help="words-<set>.jsonl 폴더(1차 디코드와 같은 words)")
    p.add_argument("--sets", nargs="+", required=True); p.add_argument("--delay", type=int, default=4); p.add_argument("--config", default="bias=0")
    p.add_argument("--decoder", required=True, help="Qwen3-ASR 디렉토리(2차 디코더)"); p.add_argument("--modes", nargs="+", default=["iso", "text", "s5"], choices=["iso", "text", "s5"])
    p.add_argument("--out-dir", required=True); p.add_argument("--gpu", default=None); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512); p.add_argument("--min-seg", type=float, default=0.8); p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--context-chars", type=int, default=600, help="text 모드 context 로 넘길 직전 최종 텍스트 끝 글자 수(0 = 끔)")
    p.add_argument("--path-remap", nargs="*", default=[], help="OLD=NEW (1차 디코드와 같은 값)"); p.add_argument("--tokenizer", default=None, help="word_k backfill tokenizer(기본: 1차 디코드 모델)")
    p.add_argument("--io-workers", type=int, default=8); p.add_argument("--score-only", action="store_true")
    a = p.parse_args(argv)
    se = _semcommit_eval(); tag = Path(a.decoder).name
    paths = [Path(a.out_dir) / f"{s}-d{a.delay}-{tag}.jsonl" for s in a.sets]
    if not a.score_only:
        dec = Decoder(a.decoder, a.batch_size, a.max_new_tokens)
        for s in a.sets: run_set(a, se, dec, s)
    summ = summarize(a, se, paths); print_summary(summ)
    se.write_json(Path(a.out_dir) / f"summary-d{a.delay}-{tag}.json", dict(protocol=PROTOCOL, delay=a.delay, config=a.config, decoder=os.path.abspath(a.decoder),
                                                                            min_seg_s=a.min_seg, min_words=a.min_words, context_chars=a.context_chars, summary=summ))
    return summ


if __name__ == "__main__":
    main()
