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


def prepare(a):
    import soundfile as sf
    from vapasr.data.kspon import read_trn, resolve_path
    rows, missing, coverage = [], [], {}
    for subset in ("test-clean", "test-other"):
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
    for subset in ("eval_clean", "eval_other"):
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
        batch_size=a.batch_size, encoder_batch_size=1, cpu_threads=a.cpu_threads)
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
    todo = [r for r in assigned if not all((r["id"], d) in done for d in a.deltas)]
    # Sort by corpus and duration to minimize dense KV-cache padding/wasted work.
    todo.sort(key=lambda r: (r["dataset"], r["audio_s"], r["id"]))
    print(f"START rank={a.rank} total={len(assigned)} remaining={len(todo)} deltas={a.deltas}", flush=True)
    model, tok = load_model(str(ck), encoder_path=a.encoder, dtype=getattr(torch, a.dtype))
    assert model.config.lanes == 0, "This adapter is for the mono model"
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
        pos = 0
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
                print(f"PROGRESS rank={a.rank} predictions={completed}/{len(assigned)*len(a.deltas)} "
                      f"dataset={batch[0]['dataset']} delta={delta} batch={len(batch)} "
                      f"encode_s={encoded_s:.1f} decode_s={elapsed:.1f} elapsed_s={time.monotonic()-start:.1f} "
                      f"peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
            pos = end
    write_json(out / f"done-rank{a.rank}.json", dict(predictions=completed, elapsed_s=time.monotonic()-start))
    print(f"DONE rank={a.rank} predictions={completed}", flush=True)


def summarize(a):
    out = Path(a.out)
    configs = sorted(out.glob("config-rank*.json"))
    if not configs:
        raise ValueError("No worker configs")
    cfg = json.loads(configs[0].read_text())
    assert all(json.loads(p.read_text()) == cfg for p in configs), "Inconsistent worker protocols"
    rows = [r for p in sorted(out.glob("predictions-rank*.jsonl")) for r in read_jsonl(p)]
    assert len({(r['id'], r['delta']) for r in rows}) == len(rows), "Duplicate predictions"
    expected = cfg["utterances"] * len(cfg["deltas"])
    complete = (len(rows) == expected and len(list(out.glob("done-rank*.json"))) == cfg["world_size"])
    report = dict(complete=complete, predictions=len(rows), expected_predictions=expected,
                  config=cfg, groups=aggregate(rows))
    write_json(out / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, ensure_ascii=False, indent=2))
    if not complete and not a.allow_partial:
        raise SystemExit("Incomplete run: no final benchmark result yet")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--libri-root", required=True)
    q.add_argument("--kspon-root", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=32)
    q = sub.add_parser("run")
    q.add_argument("--manifest", required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--encoder", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--deltas", type=int, nargs="+", default=[2, 4])
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
