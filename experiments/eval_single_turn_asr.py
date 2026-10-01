#!/usr/bin/env python3
"""Prepare / evaluate / aggregate a reproducible single-turn ASR benchmark.

See experiments/single-turn-asr-evaluation.md. Raw audio is used exactly once per
utterance, followed by one second of digital silence. No leading silence, VAD,
reference-based stopping, speaker labels or concatenation is applied.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.single_turn_eval import (PROTOCOL, aggregate, canonical_reference,
    digest, fingerprint, read_jsonl, score_pair, write_json)


VOXPOPULI_AA_REVISION = "07adf4a242e2e5dd955a230909c177573c978ee3"   # HF ArtificialAnalysis/VoxPopuli-Cleaned-AA, 2026-02-17


def prepare_voxpopuli_aa(root, revision, workers):
    """VoxPopuli-Cleaned-AA(ESB VoxPopuli 영어 test 의 AA 교정 부분집합, 628 발화) → (rows, missing, coverage).
    root 는 HF 스냅숏과 같은 배치(voxpopuli_cleaned_aa_v1.jsonl + audio/<uuid>.wav, 16 kHz mono float32)."""
    import soundfile as sf
    root = Path(root)
    src = root / "voxpopuli_cleaned_aa_v1.jsonl"
    labels = read_jsonl(src)
    if not labels:
        raise ValueError(f"No labels: {src}")

    def inspect(item):
        p = root / item["url"]
        r = dict(id=f"voxpopuli-aa/{item['id']}", dataset="voxpopuli-aa-test", lang="English",
                 raw_reference=item["transcript"], source_id=item["id"], session=item["id"].split("-en_")[0],
                 gender=item.get("gender"))
        if not p.is_file():
            return r, False, None
        info = sf.info(str(p))
        if info.channels != 1 or info.samplerate != 16000:
            raise ValueError(f"Unexpected VoxPopuli-AA format: {p} {info}")
        if abs(info.duration - float(item["duration"])) > 0.05:
            raise ValueError(f"Duration differs from the dataset card: {p} {info.duration} vs {item['duration']}")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(r, path=str(p), sample_rate=info.samplerate, num_samples=info.frames, audio_s=info.frames / info.samplerate,
                    audio_format="wav", audio_subtype=info.subtype, reference=canonical_reference(item["transcript"], "English")), True, (item["url"], sha)

    rows, missing, hashes = [], [], []
    with ThreadPoolExecutor(workers) as pool:
        for r, ok, h in pool.map(inspect, labels):
            (rows if ok else missing).append(r)
            if h:
                hashes.append(h)
    coverage = dict(source_labels=len(labels), expected_standard_count=628, revision=revision,
                    jsonl_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), audio_sha256=digest(sorted(hashes)))
    return rows, missing, coverage


def prepare(a):
    import soundfile as sf
    from vapasr.data.kspon import read_trn, resolve_path
    if not (a.voxpopuli_aa_root or a.libri_root or a.kspon_root):
        raise SystemExit("prepare: --voxpopuli-aa-root / --kspon-root / --libri-root 중 하나 이상 필요")
    rows, missing, coverage = [], [], {}
    if a.voxpopuli_aa_root:                                      # 영어 기본 평가 세트(2026-09-29 부터)
        r, m, c = prepare_voxpopuli_aa(a.voxpopuli_aa_root, a.voxpopuli_aa_revision, a.workers)
        rows += r; missing += m; coverage["voxpopuli-aa-test"] = c
    for subset in (("test-clean", "test-other") if a.libri_root else ()):   # LibriSpeech: 이전 결과와의 비교용(legacy)
        candidates = []
        for p in sorted((Path(a.libri_root) / subset).glob("*/*/*.trans.txt")):
            for line in p.read_text().splitlines():
                uid, raw = line.split(" ", 1)
                candidates.append(dict(id=f"librispeech/{subset}/{uid}", dataset=f"librispeech-{subset}",
                    lang="English", path=str(p.parent / (uid + ".flac")), raw_reference=raw))
        if not candidates:
            raise ValueError(f"No source transcripts: {a.libri_root}/{subset}")
        def inspect(r):
            if not Path(r["path"]).is_file():
                return r, False
            info = sf.info(r["path"])
            if info.channels != 1 or info.samplerate != 16000:
                raise ValueError(f"Unexpected LibriSpeech format: {r['path']} {info}")
            return dict(r, sample_rate=info.samplerate, num_samples=info.frames, audio_s=info.duration,
                        audio_format="flac", reference=r["raw_reference"]), True
        with ThreadPoolExecutor(a.workers) as pool:
            for r, ok in pool.map(inspect, candidates):
                (rows if ok else missing).append(r)
        coverage[f"librispeech-{subset}"] = dict(source_labels=len(candidates), expected_standard_count={"test-clean": 2620, "test-other": 2939}[subset])
    for subset in (("eval_clean", "eval_other") if a.kspon_root else ()):
        source = Path(a.kspon_root) / (subset + ".trn")
        labels = read_trn(str(source))
        if not labels:
            raise ValueError(f"No labels: {source}")
        def inspect_ko(item):
            rel, raw = item
            p = resolve_path(a.kspon_root, rel)
            r = dict(id=f"kspon/{subset}/{Path(rel).stem}", dataset=f"kspon-{subset}",
                     lang="Korean", raw_reference=raw, source_path=rel)
            if p is None:
                return r, False
            size = Path(p).stat().st_size
            with open(p, "rb") as f:
                header = f.read(4)
            if header == b"RIFF":
                info = sf.info(p)
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError(f"Unexpected Kspon format: {p}")
                ns, fmt, odd = info.frames, "wav", False
            else:
                ns, fmt, odd = size // 2, "pcm_s16le", bool(size % 2)
            return dict(r, path=p, sample_rate=16000, num_samples=ns, audio_s=ns / 16000,
                        audio_format=fmt, odd_trailing_byte=odd,
                        reference=canonical_reference(raw, "Korean")), True
        with ThreadPoolExecutor(a.workers) as pool:
            for r, ok in pool.map(inspect_ko, labels):
                (rows if ok else missing).append(r)
        coverage[f"kspon-{subset}"] = dict(source_labels=len(labels), expected_standard_count=3000)
    rows.sort(key=lambda r: r["id"])
    assert len({r["id"] for r in rows}) == len(rows), "duplicate utterance IDs"
    for name, c in coverage.items():
        selected = [r for r in rows if r["dataset"] == name]
        c.update(available=len(selected), missing=sum(r["dataset"] == name for r in missing),
                 audio_hours=sum(r["audio_s"] for r in selected) / 3600,
                 full_standard_set=len(selected) == c["expected_standard_count"] == c["source_labels"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    if manifest.exists():
        assert digest(read_jsonl(manifest)) == digest(rows), "Manifest changed: use a new output directory"
    else:
        with manifest.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "coverage.json", dict(protocol=PROTOCOL, coverage=coverage, missing=missing,
                                          manifest_digest=digest(rows), textnorm=fingerprint()))
    print(json.dumps(coverage, ensure_ascii=False, indent=2), flush=True)


def read_audio(row, tail_s):
    import numpy as np
    import soundfile as sf
    if row["audio_format"] == "pcm_s16le":
        with open(row["path"], "rb") as f:
            raw = f.read()
        wav = np.frombuffer(raw[:len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.
    else:
        wav, sr = sf.read(row["path"], dtype="float32")
        assert sr == 16000 and wav.ndim == 1
    assert len(wav) == row["num_samples"], f"Audio changed: {row['id']}"
    return np.pad(wav, (0, round(tail_s * 16000)))


def run(a):
    import numpy as np
    import torch
    from vapasr.hf.batch_decode import decode_batch
    from vapasr.hf.infer import load_model, prefix_ids
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = read_jsonl(a.manifest)
    if a.limit:
        # A deterministic spread of durations, independently for every test set.
        picked = []
        for name in sorted({r["dataset"] for r in rows}):
            rr = sorted((r for r in rows if r["dataset"] == name), key=lambda r: r["audio_s"])
            idx = np.linspace(0, len(rr) - 1, min(len(rr), a.limit), dtype=int)
            picked += [rr[i] for i in idx]
        rows = picked
    ck = Path(a.checkpoint)
    weights = {p.name: dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
               for p in sorted(ck.glob("*.safetensors"))}
    code_root = Path(__file__).resolve().parents[1]
    code = {p: hashlib.sha256((code_root / p).read_bytes()).hexdigest() for p in
            ["experiments/eval_single_turn_asr.py", "vapasr/hf/batch_decode.py", "vapasr/data/single_turn_eval.py",
             "vapasr/hf/modeling_vapasr.py", "vapasr/features/online.py"]}
    config = dict(protocol=PROTOCOL, checkpoint=str(ck), checkpoint_weights=weights,
        checkpoint_config_sha256=hashlib.sha256((ck / "config.json").read_bytes()).hexdigest(),
        manifest_digest=digest(rows), utterances=len(rows), deltas=a.deltas, tail_s=a.tail_s,
        leading_silence_s=0, next_bias=0, dtype=a.dtype, tf32=False, max_flush_rounds=a.max_flush,
        textnorm=fingerprint(), code=code, world_size=a.world_size, limit_per_set=a.limit,
        batch_size=a.batch_size, encoder_batch_size=1, cpu_threads=a.cpu_threads, **({"final": True} if a.final else {}),
        **({"encoder_right_context": a.encoder_right_context} if a.encoder_right_context is not None else {}))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cp = out / f"config-rank{a.rank}.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == config, "Resume protocol mismatch; use a fresh output directory"
    else:
        write_json(cp, config)
    dest = out / f"predictions-rank{a.rank}.jsonl"
    previous = read_jsonl(dest) if dest.exists() else []
    done = {(r["id"], r["delta"]) for r in previous}
    assert len(done) == len(previous), "Duplicate predictions in resume file"
    assigned = [r for i, r in enumerate(rows) if i % a.world_size == a.rank]
    conds = list(a.deltas) + (["final"] if a.final else [])                     # final = 같은 모델의 오프라인 모드(vapasr/hf/offline_decode)
    assert conds, "--deltas 나 --final 중 하나는 필요"
    todo = [r for r in assigned if not all((r["id"], d) in done for d in conds)]
    # Sort by corpus and duration to minimize dense KV-cache padding/wasted work.
    todo.sort(key=lambda r: (r["dataset"], r["audio_s"], r["id"]))
    print(f"START rank={a.rank} total={len(assigned)} remaining={len(todo)} deltas={a.deltas}", flush=True)
    model, tok = load_model(str(ck), encoder_path=a.encoder, dtype=getattr(torch, a.dtype))
    assert model.config.lanes == 0, "This adapter is for the mono model"
    if a.encoder_right_context is not None:                                      # 평가 시 인코더 lookahead 만 바꾼다(config 는 메모리에서만)
        ctx = [int(model.config.encoder_left_context), int(a.encoder_right_context)]; allowed = [list(map(int, c)) for c in (getattr(model.encoder.enc, "att_context_size_all", None) or [])]
        assert not allowed or ctx in allowed, f"att_context {ctx} not in {allowed}"
        model.encoder.enc.set_default_att_context_size(ctx); model.config.encoder_right_context = ctx[1]
    print(f"encoder att_context {model.encoder.enc.att_context_size}", flush=True)
    for d in a.deltas:
        assert f"<DELAY_{d}>" in model.config.sp_ids
    dev = next(model.parameters()).device
    checked, completed, start = set(), len(previous), time.monotonic()

    def safe_decode(feats, prefixes):
        try:
            return decode_batch(model, feats, prefixes, max_flush_rounds=a.max_flush)
        except torch.cuda.OutOfMemoryError:
            if len(feats) == 1:
                raise
        # Leave the except scope first so the failed decoder traceback releases
        # its KV cache before retrying smaller batches.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        m = len(feats) // 2
        print(f"OOM splitting decode batch {len(feats)} into {m}+{len(feats)-m}", flush=True)
        return safe_decode(feats[:m], prefixes[:m]) + safe_decode(feats[m:], prefixes[m:])

    with ThreadPoolExecutor(a.io_workers) as pool, dest.open("a", buffering=1) as output, torch.inference_mode():
        # Submit in the background; waveform arrays stay in CPU RAM for reuse.
        futures = [pool.submit(read_audio, r, a.tail_s) for r in todo]
        pos = 0; ofmt = oblk = None
        while pos < len(todo):
            end = min(len(todo), pos + a.batch_size)
            while todo[end - 1]["dataset"] != todo[pos]["dataset"]:
                end -= 1
            batch = todo[pos:end]
            feats = []
            bt = time.monotonic()
            for i in range(pos, end):
                wav = futures[i].result()
                w = torch.from_numpy(wav).to(dev)
                K = int(round(len(wav) / 16000 * model.config.frame_hz))
                # Singleton encoder preserves per-file normalization and padding
                # exactly. Decoder batching is the main throughput improvement.
                f = model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0]
                feats.append(f)
                futures[i] = None
            encoded_s = time.monotonic() - bt
            for delta in a.deltas:
                prefixes = [prefix_ids(model, tok, r["lang"], delta) for r in batch]
                dt = time.monotonic()
                states = safe_decode(feats, prefixes)
                elapsed = time.monotonic() - dt
                check_key = (batch[0]["lang"], delta)
                if a.verify and check_key not in checked:
                    checks = []
                    for j in sorted({0, len(batch) - 1}):
                        serial, forced, _, rounds = model.stream_decode(feats[j], prefixes[j], max_flush_rounds=a.max_flush)
                        match = serial == states[j].emitted and forced == states[j].forced and rounds == states[j].flush_rounds
                        checks.append(dict(id=batch[j]["id"], delta=delta, match=match,
                            serial=serial, batched=states[j].emitted, forced_serial=forced, forced_batch=states[j].forced))
                    vp = out / f"parity-rank{a.rank}-{check_key[0]}-d{delta}.json"
                    write_json(vp, checks)
                    assert all(c["match"] for c in checks), f"Serial/batch parity failed: {vp}"
                    checked.add(check_key)
                    print(f"PARITY rank={a.rank} lang={check_key[0]} delta={delta} samples={len(checks)} PASS", flush=True)
                for r, s in zip(batch, states):
                    if (r["id"], delta) in done:
                        continue
                    ids = [tid for _, tid in s.emitted]
                    hyp = tok.decode(ids, skip_special_tokens=True).strip()
                    primary = "wer" if r["lang"] == "English" else "cer_nospace"
                    from vapasr.data.textnorm import score_en, score_ko
                    norm = score_en if r["lang"] == "English" else lambda x: score_ko(x, False)
                    rec = dict(id=r["id"], dataset=r["dataset"], lang=r["lang"], path=r["path"],
                        delta=delta, audio_s=r["audio_s"], tail_s=a.tail_s, K=s.K,
                        raw_reference=r["raw_reference"], reference=r["reference"], hypothesis=hyp,
                        reference_normalized=norm(r["reference"]), hypothesis_normalized=norm(hyp),
                        primary_metric=primary, metrics=score_pair(r["reference"], hyp, r["lang"]),
                        token_ids=ids, emissions=s.emitted, forced=s.forced, flush_rounds=s.flush_rounds,
                        post_audio_tokens=sum((k + 1) * .08 > r["audio_s"] for k, _ in s.emitted),
                        batch_size=len(batch), batch_decode_s=elapsed, batch_encode_s=encoded_s)
                    output.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done.add((r["id"], delta))
                    completed += 1
                output.flush()
                os.fsync(output.fileno())
                print(f"PROGRESS rank={a.rank} predictions={completed}/{len(assigned)*len(conds)} "
                      f"dataset={batch[0]['dataset']} delta={delta} batch={len(batch)} "
                      f"encode_s={encoded_s:.1f} decode_s={elapsed:.1f} elapsed_s={time.monotonic()-start:.1f} "
                      f"peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
            if a.final:                                                         # 같은 인코더 특징(인과)으로 오프라인 모드 전사
                from vapasr.hf.offline_decode import offline_decode, offline_blocked_ids
                from vapasr.data.offline_seq import OfflineFormat
                from vapasr.data.textnorm import score_en, score_ko
                if ofmt is None: ofmt = OfflineFormat(tok); oblk = torch.tensor(offline_blocked_ids(model, ofmt), device=dev)
                dt = time.monotonic()
                for r, f in zip(batch, feats):
                    if (r["id"], "final") in done:
                        continue
                    ids = offline_decode(model, tok, f[0], r["lang"], fmt=ofmt, blocked=oblk)
                    hyp = tok.decode(ids, skip_special_tokens=True).strip()
                    norm = score_en if r["lang"] == "English" else (lambda x: score_ko(x, False))
                    rec = dict(id=r["id"], dataset=r["dataset"], lang=r["lang"], path=r["path"], delta="final", audio_s=r["audio_s"], tail_s=a.tail_s,
                        K=int(f.shape[1]), raw_reference=r["raw_reference"], reference=r["reference"], hypothesis=hyp,
                        reference_normalized=norm(r["reference"]), hypothesis_normalized=norm(hyp),
                        primary_metric="wer" if r["lang"] == "English" else "cer_nospace", metrics=score_pair(r["reference"], hyp, r["lang"]),
                        token_ids=ids, emissions=[], forced=0, flush_rounds=0, post_audio_tokens=0, batch_size=1, batch_decode_s=None, batch_encode_s=encoded_s)
                    output.write(json.dumps(rec, ensure_ascii=False) + "\n"); done.add((r["id"], "final")); completed += 1
                output.flush(); os.fsync(output.fileno())
                print(f"PROGRESS rank={a.rank} predictions={completed}/{len(assigned)*len(conds)} dataset={batch[0]['dataset']} delta=final "
                      f"batch={len(batch)} decode_s={time.monotonic()-dt:.1f} elapsed_s={time.monotonic()-start:.1f}", flush=True)
            pos = end
    write_json(out / f"done-rank{a.rank}.json", dict(predictions=completed, elapsed_s=time.monotonic()-start))
    print(f"DONE rank={a.rank} predictions={completed}", flush=True)


def aa_report(rows):
    """영어 세트·조건별 AA-WER 비교 지표(vapasr/data/aa_wer.py: Whisper 정규화 + 숫자 분리, micro 와 오디오 길이 가중 평균).
    내부 주 지표(groups 의 wer = score_en micro)와 별개의 보조 지표다. 영어 행이 없으면 빈 dict."""
    from vapasr.data.aa_wer import aa_fingerprint, aa_summary
    en = {}
    for r in rows:
        if r["lang"] == "English":
            en.setdefault(f"{r['dataset']}/delta-{r['delta']}", []).append(r)
    if not en:
        return {}
    return dict(aa_wer={k: aa_summary(v) for k, v in sorted(en.items())}, aa_scoring=aa_fingerprint())


def summarize(a):
    out = Path(a.out)
    configs = sorted(out.glob("config-rank*.json"))
    if not configs:
        raise ValueError("No worker configs")
    cfg = json.loads(configs[0].read_text())
    assert all(json.loads(p.read_text()) == cfg for p in configs), "Inconsistent worker protocols"
    rows = [r for p in sorted(out.glob("predictions-rank*.jsonl")) for r in read_jsonl(p)]
    assert len({(r['id'], r['delta']) for r in rows}) == len(rows), "Duplicate predictions"
    expected = cfg["utterances"] * (len(cfg["deltas"]) + int(bool(cfg.get("final"))))
    complete = (len(rows) == expected and len(list(out.glob("done-rank*.json"))) == cfg["world_size"])
    report = dict(complete=complete, predictions=len(rows), expected_predictions=expected,
                  config=cfg, groups=aggregate(rows), **aa_report(rows))
    write_json(out / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, ensure_ascii=False, indent=2))
    if not complete and not a.allow_partial:
        raise SystemExit("Incomplete run: no final benchmark result yet")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--voxpopuli-aa-root", default=None, help="영어 기본 세트: VoxPopuli-Cleaned-AA 스냅숏 루트(jsonl + audio/)")
    q.add_argument("--voxpopuli-aa-revision", default=VOXPOPULI_AA_REVISION, help="coverage.json 에 기록할 HF 리비전")
    q.add_argument("--kspon-root", default=None, help="한국어: KsponSpeech eval_clean/eval_other")
    q.add_argument("--libri-root", default=None, help="legacy 영어 비교: LibriSpeech test-clean/test-other")
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=32)
    q = sub.add_parser("run")
    q.add_argument("--manifest", required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--encoder", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--deltas", type=int, nargs="*", default=[2, 4])
    q.add_argument("--encoder-right-context", type=int, default=None, help="인코더 att_context 우측 프레임을 체크포인트 config 대신 이 값으로(예: 3 → [56,3]). "
                   "multi-lookahead 로 학습된 인코더(--encoder-context-sampling multi)에서만 의미가 있다")
    q.add_argument("--final", action="store_true", help="final(오프라인) 모드도 평가(같은 모델·같은 인코더 특징; offline_frac 으로 학습한 모델) — delta='final'")
    q.add_argument("--tail-s", type=float, default=1.0)
    q.add_argument("--max-flush", type=int, default=8)
    q.add_argument("--batch-size", type=int, default=64)
    q.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    q.add_argument("--cpu-threads", type=int, default=4)
    q.add_argument("--io-workers", type=int, default=16)
    q.add_argument("--rank", type=int, default=0)
    q.add_argument("--world-size", type=int, default=1)
    q.add_argument("--limit", type=int, default=0, help="Smoke test only: deterministic samples per dataset")
    q.add_argument("--verify", action="store_true", help="Require exact serial/batch token and timing parity")
    q = sub.add_parser("summarize")
    q.add_argument("--out", required=True)
    q.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    if a.command == "run":
        assert 0 <= a.rank < a.world_size and a.tail_s >= 0 and a.batch_size > 0 and a.max_flush >= 0
    {"prepare": prepare, "run": run, "summarize": summarize}[a.command](a)


if __name__ == "__main__":
    main()
