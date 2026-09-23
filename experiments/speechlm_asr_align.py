#!/usr/bin/env python3
"""Bounded Arrow -> verbatim Qwen/Whisper -> pair QC -> forced alignment.

prepare/check are CPU-only. worker runs under Slurm (one visible GPU per task).
No source writes, automatic training approval, or silent held-out inclusion.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import fcntl
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest, agreement
from vapasr.data.speechlm_inference import POLICY, TarReader, adapt, decode, decide
from vapasr.data.selection_alignment import project_items, text_sha
from vapasr.data import textnorm


def save(path, value):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def freeze_model(path):
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and p.suffix in (".safetensors", ".bin", ".json", ".txt", ".model")
                   and not any(x.startswith(".") for x in p.relative_to(path).parts))
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError(f"No weights: {path}")
    return dict(path=str(path.resolve()), files={str(p.relative_to(path)): dict(
        sha256=file_digest(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files})


def prepare(a):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    source, out = a.source.resolve(), a.out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Use independent output directory")
    if out.exists():
        raise ValueError("Preparation output exists; use a new root (workers resume separately)")
    original = json.loads((source / "summary.json").read_text())
    out.mkdir(parents=True)
    for d in ("inputs", "results", "locks", "work", "tar-index"):
        (out / d).mkdir()
    jobs, counts, hashes = [], Counter(), {}
    for catalog in sorted(original["catalogs"]):
        files = sorted((source / "catalogs" / catalog).glob("data-*.arrow"))
        if not files:
            raise ValueError(f"Missing Arrow files: {catalog}")
        sampled = 0
        pending = []
        def flush():
            if not pending:
                return
            name = f"part-{len(jobs):06d}"
            dest = out / "inputs" / (name + ".jsonl")
            with dest.open("w") as f:
                for row in pending:
                    line(f, row)
            jobs.append(dict(name=name, catalog=catalog, rows=len(pending),
                             path=str(dest.relative_to(out)), sha256=file_digest(dest)))
            pending.clear()
        for i, path in enumerate(files):
            if a.per_catalog and sampled >= a.per_catalog:
                break
            # Smoke distributes small prefixes across available Arrow shards;
            # it is a coverage check, NOT an unbiased quality estimate.
            quota = math.ceil((a.per_catalog - sampled) / (len(files) - i)) if a.per_catalog else None
            hashes[str(path)] = file_digest(path)
            taken = 0
            with pa.memory_map(str(path), "r") as mm:
                for batch in ipc.open_stream(mm):
                    if quota is not None:
                        batch = batch.slice(0, min(batch.num_rows, quota - taken))
                    for record in batch.to_pylist():
                        row = adapt(record)
                        # Reject NaN metadata explicitly; JSON has no portable NaN.
                        json.dumps(row, allow_nan=False)
                        pending.append(row)
                        counts[row["selection_state"]] += 1
                        sampled += 1
                        taken += 1
                        if len(pending) >= a.shard_rows:
                            flush()
                    if quota is not None and taken >= quota:
                        break
        flush()
        print(f"PREPARED {catalog} rows={sampled}", flush=True)
    total = sum(j["rows"] for j in jobs)
    if not a.per_catalog and total != original["total_rows"]:
        raise ValueError("Full Arrow/source summary row mismatch")
    root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__).resolve(), *(root / "vapasr/data").glob("*.py")]
    package = Path(__import__("qwen_asr").__file__).parent
    config = dict(schema="speechlm-asr-align-v1", source=str(source),
        source_summary_sha256=file_digest(source / "summary.json"), arrow_sha256=hashes,
        scope="smoke" if a.per_catalog else "full", per_catalog=a.per_catalog,
        policy=POLICY, tn=textnorm.fingerprint(), rows=total, states=counts, jobs=jobs,
        batch=a.batch, batch_sec=a.batch_sec, io_workers=a.io_workers, memory_fraction=.9,
        tar_index=str(a.tar_index.resolve()),
        models={name: freeze_model(path) for name, path in
                (("qwen", a.qwen), ("whisper", a.whisper), ("aligner", a.aligner))},
        code={str(p.relative_to(root)): file_digest(p) for p in code_paths},
        qwen_package={str(p.relative_to(package)):file_digest(p) for p in package.rglob("*.py")},
        packages={name: importlib.metadata.version(name) for name in
                  ("torch", "transformers", "qwen-asr", "pyarrow", "soundfile", "soxr")},
        recipe=dict(qwen_max_new_tokens=448, whisper_max_length=448, whisper_dtype="float16",
                    qwen_aligner_dtype="bfloat16", greedy=True, prompt=None,
                    tail_padding_s=0, target_normalization="none"))
    config["fingerprint"] = digest(config)
    save(out / "config.json", config)
    print(f"READY rows={total} shards={len(jobs)} scope={config['scope']}", flush=True)


def read_config(out):
    config = json.loads((out / "config.json").read_text())
    check = dict(config)
    signature = check.pop("fingerprint")
    if digest(check) != signature:
        raise ValueError("Config fingerprint mismatch")
    root = Path(__file__).resolve().parents[1]
    for path, sha in config["code"].items():
        if file_digest(root / path) != sha:
            raise ValueError(f"Code changed: {path}; restore frozen code or use new output root")
    if config["tn"] != textnorm.fingerprint():
        raise ValueError("Comparison TN changed")
    return config


def rows_for(out, job):
    path = out / job["path"]
    if file_digest(path) != job["sha256"]:
        raise ValueError("Input shard changed")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    if len(rows) != job["rows"] or len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Input count/identity mismatch")
    return rows


def done(out, config, job):
    dest = out / "results" / job["name"]
    if not dest.exists():
        return None
    summary = json.loads((dest / "summary.json").read_text())
    if (summary["fingerprint"] != config["fingerprint"] or summary["rows"] != job["rows"]
            or not summary["complete"] or sum(summary["states"].values()) != job["rows"]):
        raise ValueError("Resume summary mismatch")
    for filename, sha in summary["outputs"].items():
        if file_digest(dest / filename) != sha:
            raise ValueError("Resume output changed")
    return summary


def check_audio(a):
    config = read_config(a.out)
    reader = TarReader(config["tar_index"])
    counts, examples = Counter(), []
    for job in config["jobs"]:
        for row in rows_for(a.out, job)[:a.check_rows]:
            if row["reasons"]:
                counts["metadata_hold"] += 1
                continue
            try:
                _, info = decode(row, reader)
                counts["audio_hold" if info["review"] else "ok"] += 1
                examples.append(dict(key=row["key"], source=row["source"], audio=info))
            except Exception as exc:
                counts["error"] += 1
                examples.append(dict(key=row["key"], source=row["source"], error=str(exc)))
    save(a.out / "cpu-check.json", dict(counts=counts, samples=examples, gpu_used=False))
    print(json.dumps(counts), flush=True)
    if counts["error"]:
        raise SystemExit(2)


class Engines:
    def __init__(self, config):
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer
        self.torch = torch
        self.qwen = Qwen3ASRModel.from_pretrained(config["models"]["qwen"]["path"],
            dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=config["batch"],
            max_new_tokens=448)
        self.whisper = AutoModelForSpeechSeq2Seq.from_pretrained(config["models"]["whisper"]["path"],
            torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True,
            attn_implementation="sdpa", local_files_only=True).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(config["models"]["whisper"]["path"], local_files_only=True)
        self.aligner = Qwen3ForcedAligner.from_pretrained(config["models"]["aligner"]["path"],
            dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)
        self.aligner.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config["models"]["qwen"]["path"],
            use_fast=True, fix_mistral_regex=True, local_files_only=True)
        if not self.tokenizer.is_fast:
            raise ValueError("Fast tokenizer required for verbatim offsets")

    def once(self, name, batch):
        try:
            with self.torch.inference_mode():
                audio = [(x, 16000) for r, x in batch]
                if name == "qwen":
                    result = self.qwen.transcribe(audio=audio, language=[r["lang"] for r, x in batch])
                    result = [dict(text=r.text, warning="token_cap" if len(
                        self.tokenizer.encode(r.text, add_special_tokens=False)) >= 430 else None) for r in result]
                elif name == "whisper":
                    inputs = self.processor([x for r, x in batch], sampling_rate=16000,
                        return_tensors="pt", return_attention_mask=True, padding="max_length", truncation=False)
                    ids = self.whisper.generate(input_features=inputs.input_features.to("cuda", dtype=self.torch.float16),
                        attention_mask=inputs.attention_mask.to("cuda"), language=batch[0][0]["lang"].lower(),
                        task="transcribe", max_new_tokens=None, max_length=448, condition_on_prev_tokens=False,
                        return_timestamps=False, do_sample=False, num_beams=1)
                    texts = self.processor.batch_decode(ids, skip_special_tokens=True)
                    result = [dict(text=t, warning="token_cap" if len(self.processor.tokenizer.encode(
                        t, add_special_tokens=False)) >= 430 else None) for t in texts]
                else:
                    outputs = self.aligner.align(audio=audio, text=[r["transcripts"]["qwen_raw"] for r, x in batch],
                        language=[r["lang"] for r, x in batch])
                    result = [dict(items=[dict(text=i.text, start_time=i.start_time, end_time=i.end_time)
                                          for i in res.items]) for res in outputs]
                if len(result) != len(batch):
                    raise ValueError("Model output count mismatch")
                return result, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:500]}"

    def infer(self, name, batch):
        if not batch:
            return []
        result, error = self.once(name, batch)
        if error is None:
            return result
        # Exit exception frame before splitting so CUDA allocations can be freed.
        gc.collect()
        self.torch.cuda.empty_cache()
        if len(batch) == 1:
            return [dict(error=error)]
        print(f"RETRY {name} n={len(batch)} {error}", flush=True)
        mid = len(batch) // 2
        return self.infer(name, batch[:mid]) + self.infer(name, batch[mid:])


def groups(rows, config):
    group, seconds = [], 0.
    for row in sorted(rows, key=lambda r: (r["lang"] or "", r["duration_s"])):
        if group and (row["lang"] != group[0]["lang"] or len(group) >= config["batch"]
                      or seconds + row["duration_s"] > config["batch_sec"]):
            yield group
            group, seconds = [], 0.
        group.append(row)
        seconds += row["duration_s"]
    if group:
        yield group


def worker(a):
    import torch
    config = read_config(a.out)
    rank, world, nodes = (int(os.environ[k]) for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_JOB_NUM_NODES"))
    if world != nodes * 8 or torch.cuda.device_count() != 1 or not 0 <= rank < world:
        raise ValueError("Require 8 GPU tasks/node with one visible GPU/task")
    if int(os.environ["SLURM_CPUS_PER_TASK"]) < config["io_workers"] + 1:
        raise ValueError("Insufficient CPU allocation")
    for name, version in config["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Package version changed: {name}")
    package = Path(__import__("qwen_asr").__file__).parent
    for path, sha in config["qwen_package"].items():
        if file_digest(package / path) != sha:
            raise ValueError("Qwen package code changed")
    for model in config["models"].values():
        for name, expected in model["files"].items():
            stat = (Path(model["path"]) / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
                raise ValueError("Model metadata changed; re-prepare and hash")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(config["memory_fraction"])
    reader, engines = TarReader(config["tar_index"]), None
    def safe_decode(row):
        try:
            x, info = decode(row, reader)
            row["audio_info"] = info
            row["auto_selection"] = dict(expected_waveform_sha256=info["sha256"])
            return row, x
        except Exception as exc:
            row.update(selection_state="HOLD_AUDIO", error=f"{type(exc).__name__}: {exc}")
            return row, None
    with ThreadPoolExecutor(config["io_workers"]) as pool:
        for job in config["jobs"][rank::world]:
            with (a.out / "locks" / (job["name"] + ".lock")).open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if done(a.out, config, job):
                    continue
                rows = rows_for(a.out, job)
                stage = Path(tempfile.mkdtemp(prefix=job["name"] + "-", dir=a.out / "work"))
                counts, aligned, failed, processed = Counter(), 0, 0, 0
                started = time.time()
                with (stage / "records.jsonl").open("w") as records, (stage / "aligned.jsonl").open("w") as aligned_out, (stage / "failed.jsonl").open("w") as failed_out:
                    def emit(row):
                        nonlocal processed
                        counts[row["selection_state"]] += 1
                        processed += 1
                        row["fingerprint"] = config["fingerprint"]
                        line(records, row)
                    eligible = []
                    for row in rows:
                        if row["reasons"]:
                            emit(row)
                        else:
                            eligible.append(row)
                    iterator = iter(groups(eligible, config))
                    def prefetch():
                        return [pool.submit(safe_decode, row) for row in next(iterator, [])]
                    pending = prefetch()
                    while pending:
                        decoded = [f.result() for f in pending]
                        pending = prefetch()
                        batch = []
                        for row, x in decoded:
                            if x is None or row["audio_info"]["review"]:
                                row["selection_state"] = "HOLD_AUDIO"
                                emit(row)
                            else:
                                batch.append((row, x))
                        if not batch:
                            continue
                        if engines is None:
                            engines = Engines(config)
                        qwen, whisper = engines.infer("qwen", batch), engines.infer("whisper", batch)
                        selected = []
                        for (row, x), q, w in zip(batch, qwen, whisper):
                            row["teachers"] = dict(qwen=q, whisper=w)
                            row["transcripts"] = dict(source_raw=row["raw_text"], qwen_raw=q.get("text"), whisper_raw=w.get("text"))
                            metrics = None
                            if not q.get("error") and not w.get("error"):
                                try:
                                    norm = (lambda s: list(textnorm.score_ko(s, False))) if row["lang"] == "Korean" else (lambda s: textnorm.score_en(s).split())
                                    metrics = agreement(norm(row["text"]), norm(q["text"]), norm(w["text"]))
                                except Exception as exc:
                                    row["comparison_error"] = str(exc)
                            row["metrics"] = metrics
                            row["selection_state"] = decide(row, q, w, metrics)
                            if row["selection_state"] == "KEEP_ASR_SILVER":
                                row["recommended_training_target"] = dict(text=q["text"], normalization="none", origin="qwen3_asr_pseudo_label")
                                selected.append((row, x))
                            emit(row)
                        for (row, x), result in zip(selected, engines.infer("align", selected)):
                            text = row["transcripts"]["qwen_raw"]
                            base = dict(key=row["key"], source=row["source"], lang=row["lang"], audio=row["audio"],
                                target_text=text, target_sha256=text_sha(text), waveform_sha256=row["audio_info"]["sha256"],
                                duration_s=len(x)/16000, fingerprint=config["fingerprint"], source_shard=job["name"],
                                training_eligible=False, dedup_status="unverified")
                            try:
                                if result.get("error"):
                                    raise ValueError(result["error"])
                                base["aligner_items"] = result["items"]
                                base.update(project_items(text, result["items"], engines.tokenizer(text,
                                    add_special_tokens=False, return_offsets_mapping=True), base["duration_s"]))
                                base["alignment_ok"] = True
                                line(aligned_out, base)
                                aligned += 1
                            except Exception as exc:
                                base.update(alignment_ok=False, error=str(exc))
                                line(failed_out, base)
                                failed += 1
                        print(f"PROGRESS rank={rank} {job['name']} {processed}/{len(rows)} aligned={aligned} fail={failed}", flush=True)
                if processed != job["rows"] or aligned + failed != counts["KEEP_ASR_SILVER"]:
                    raise ValueError("Output accounting mismatch")
                summary = dict(complete=True, fingerprint=config["fingerprint"], rows=processed, states=counts,
                    aligned=aligned, alignment_failed=failed, elapsed_s=time.time()-started,
                    gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9,
                    outputs={p.name:file_digest(p) for p in stage.glob("*.jsonl")})
                save(stage / "summary.json", summary)
                stage.rename(a.out / "results" / job["name"])
                print(f"SHARD_DONE rank={rank} {job['name']} {json.dumps(counts)}", flush=True)


def summarize(a):
    config = read_config(a.out)
    counts, results = Counter(), []
    catalogs = defaultdict(Counter)
    for job in config["jobs"]:
        result = done(a.out, config, job)
        if result:
            results.append(result)
            counts.update(result["states"])
            catalogs[job["catalog"]].update(result["states"])
    complete = len(results) == len(config["jobs"])
    summary = dict(complete=complete, scope=config["scope"], fingerprint=config["fingerprint"],
        expected_rows=config["rows"], completed_rows=sum(r["rows"] for r in results),
        completed_shards=len(results), total_shards=len(config["jobs"]), states=counts, catalogs=catalogs,
        aligned=sum(r["aligned"] for r in results), alignment_failed=sum(r["alignment_failed"] for r in results),
        training_eligible=False, remaining_gates=["dedup_train_and_heldout", "alignment_qc", "source_balance", "approval"])
    save(a.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not complete:
        raise SystemExit(2)


def index_one(job):
    archive, root = job
    started = time.time()
    try:
        members = TarReader(root, max_indexes=1).index(archive)
        stat = Path(archive).stat()
        return dict(archive=archive, status="ok", members=len(members),
                    bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, elapsed_s=time.time()-started)
    except Exception as exc:
        return dict(archive=archive, status="error", error=f"{type(exc).__name__}: {exc}")


def preindex(a):
    """Only Arrow-referenced tar files; no directory glob of source audio."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    source = a.source.resolve()
    summary = json.loads((source / "summary.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "index-run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths, hashes, rows = set(), {}, 0
        for catalog in sorted(summary["catalogs"]):
            for path in sorted((source / "catalogs" / catalog).glob("data-*.arrow")):
                hashes[str(path)] = file_digest(path)
                with pa.memory_map(str(path), "r") as mm:
                    for batch in ipc.open_stream(mm):
                        rows += batch.num_rows
                        paths.update(batch.column(batch.schema.get_field_index("audio_tar")).unique().to_pylist())
            print(f"INDEX_SCAN {catalog} rows={rows} distinct_tar={len(paths)}", flush=True)
        if rows != summary["total_rows"] or not paths or None in paths:
            raise ValueError("Arrow coverage mismatch or invalid tar path")
        save(a.out / "index-input.json", dict(source=str(source), source_rows=rows,
             source_summary_sha256=file_digest(source / "summary.json"), arrow_sha256=hashes,
             tar_index=str(a.tar_index.resolve()), archives=sorted(paths)))
        counts, members, started = Counter(), 0, time.time()
        # Append-only attempt log; atomic summary is written only on completion.
        with (a.out / "index-results.jsonl").open("a") as results, ProcessPoolExecutor(a.index_workers) as pool:
            for result in pool.map(index_one, ((p, a.tar_index) for p in sorted(paths)), chunksize=1):
                line(results, result)
                counts[result["status"]] += 1
                members += result.get("members", 0)
                if sum(counts.values()) % 100 == 0:
                    results.flush()
                    print(f"INDEX {sum(counts.values())}/{len(paths)} {dict(counts)}", flush=True)
        save(a.out / "index-summary.json", dict(complete=True, ready=counts["error"] == 0,
             expected_archives=len(paths), counts=counts, members=members,
             elapsed_s=time.time()-started, source_rows=rows, gpu_used=False,
             note="member offsets only; audio decoding/member coverage checked separately"))
        if counts["error"]:
            raise SystemExit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "check", "worker", "summarize", "index"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("/soundai/users/tskim/VAPKT-data/data/speechlm-asr-single-speaker-v1-20260922"))
    p.add_argument("--per-catalog", type=int, default=32, help="0 = all rows; default is smoke only")
    p.add_argument("--shard-rows", type=int, default=4096)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--batch-sec", type=float, default=480.)
    p.add_argument("--io-workers", type=int, default=7)
    p.add_argument("--check-rows", type=int, default=2)
    p.add_argument("--index-workers", type=int, default=16)
    p.add_argument("--tar-index", type=Path, default=Path("/soundai/users/tskim/VAPKT-data/data/speechlm-asr-tar-index-v1"))
    p.add_argument("--qwen", type=Path, default=Path("/soundai/Model/Qwen3-ASR-0.6B"))
    p.add_argument("--whisper", type=Path, default=Path("/soundai/users/tskim/VAPKT-data/baselines/whisper-large-v2"))
    p.add_argument("--aligner", type=Path, default=Path("/soundai/Model/Qwen3-ForcedAligner-0.6B"))
    a = p.parse_args()
    if a.per_catalog < 0 or min(a.batch, a.batch_sec, a.shard_rows, a.io_workers, a.check_rows) <= 0 or not 1 <= a.index_workers <= 48:
        p.error("Invalid resource/row limit")
    {"prepare":prepare, "check":check_audio, "worker":worker, "summarize":summarize, "index":preindex}[a.mode](a)


if __name__ == "__main__":
    main()
