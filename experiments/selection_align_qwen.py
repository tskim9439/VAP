#!/usr/bin/env python3
"""Selected Qwen verbatim targets: prepare / GPU worker / CPU summary.

Input is results/part-*/keep-asr.jsonl, NEVER the old row['text'] or decisions.
No TN, no source mutation, no inferred EOT/speaker labels. See the Slurm runbook.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest
from vapasr.data.selection_alignment import SCHEMA, project_items, target_text, text_sha


def save(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def locked(path):
    handle = path.open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def tree_hash(path, suffixes):
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes
                   and not any(part.startswith(".") for part in p.relative_to(path).parts))
    if not files:
        raise ValueError(f"No model/package files: {path}")
    return {str(p.relative_to(path)): file_digest(p) for p in files}


def prepare(a):
    source, out = a.source.resolve(), a.out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Output must be independent of the selection input")
    summary = json.loads((source / "summary.json").read_text())
    if not summary["complete"] or summary["policy"]["version"] != "asr-pair-select-v2":
        raise ValueError("Expected completed pair-v2 selection")
    if not 0 < a.mem_frac <= 0.95 or min(a.batch, a.batch_sec, a.io_threads) <= 0:
        raise ValueError("Invalid batch/CPU/memory settings")
    if min(a.max_shards, a.max_rows) < 0:
        raise ValueError("Negative smoke limit")
    shards = sorted(summary["shards"], key=lambda s: s["name"])
    if not shards:
        raise ValueError("Empty selection shard list")
    if len({s["name"] for s in shards}) != len(shards):
        raise ValueError("Duplicate shard name")
    if a.max_shards and a.max_shards < len(shards):
        # Smoke across the source list, rather than only the first corpus.
        shards = [shards[round(i * (len(shards) - 1) / max(1, a.max_shards - 1))]
                  for i in range(a.max_shards)]
    jobs = []
    for shard in shards:
        name = shard["name"]
        if Path(name).name != name or not name.startswith("part-"):
            raise ValueError("Unsafe shard name")
        detail = json.loads((source / "results" / name / "summary.json").read_text())
        if not detail["complete"] or detail["outputs"] != shard["outputs"]:
            raise ValueError(f"Selection shard summary mismatch: {name}")
        count = sum(c.get("KEEP_ASR_SILVER", 0) for c in detail["sources"].values())
        jobs.append(dict(name=name, input_sha256=shard["outputs"]["keep-asr.jsonl"],
                         selected_rows=count, rows=min(count, a.max_rows) if a.max_rows else count))
    if not a.max_shards:
        expected = sum(c.get("KEEP_ASR_SILVER", 0) for c in summary["sources"].values())
        if sum(s["selected_rows"] for s in jobs) != expected:
            raise ValueError("Root/shard selected-row counts disagree")
    root = Path(__file__).resolve().parents[1]
    package = Path(importlib.util.find_spec("qwen_asr").origin).parent
    code = [Path(__file__).resolve(), root / "vapasr/data/selection_alignment.py",
            root / "vapasr/data/selection_audio.py", root / "vapasr/data/archive.py",
            root / "vapasr/data/selection.py"]
    config = dict(schema=SCHEMA, source=str(source), source_summary_sha256=file_digest(source / "summary.json"),
                  aligner=str(a.aligner.resolve()), tokenizer=str(a.tokenizer.resolve()),
                  aligner_files=tree_hash(a.aligner, {".json", ".safetensors", ".bin", ".txt", ".model"}),
                  tokenizer_files=tree_hash(a.tokenizer, {".json", ".txt", ".model"}),
                  package_files=tree_hash(package, {".py", ".dict"}),
                  versions={p: importlib.metadata.version(p) for p in ("qwen-asr", "torch", "transformers", "numpy", "soundfile", "soxr")},
                  code={str(p.relative_to(root)): file_digest(p) for p in code},
                  tokenizer_options=dict(fix_mistral_regex=True, add_special_tokens=False),
                  target_normalization="none", dtype="bfloat16", batch=a.batch, batch_sec=a.batch_sec,
                  io_threads=a.io_threads, mem_frac=a.mem_frac, max_rows=a.max_rows,
                  scope="smoke" if a.max_shards or a.max_rows else "full", shards=jobs)
    config["fingerprint"] = digest(config)
    out.mkdir(parents=True, exist_ok=True)
    with locked(out / "prepare.lock"):
        config_path = out / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Resume fingerprint changed; use a NEW output root")
        elif any(p.name != "prepare.lock" for p in out.iterdir()):
            raise ValueError("Existing output without fingerprint")
        save(config_path, config)
        for directory in ("results", "work", "locks"):
            (out / directory).mkdir(exist_ok=True)
    print(f"PREPARED scope={config['scope']} shards={len(jobs)} rows={sum(s['rows'] for s in jobs)}", flush=True)


def completed(out, spec, fingerprint):
    dest = out / "results" / spec["name"]
    if not dest.exists():
        return None
    result = json.loads((dest / "summary.json").read_text())
    if (not result["complete"] or result["fingerprint"] != fingerprint
            or result["input_sha256"] != spec["input_sha256"] or result["rows"] != spec["rows"]
            or result["ok"] + result["failed"] != result["rows"]
            or set(result["outputs"]) != {"aligned.jsonl", "failed.jsonl"}):
        raise ValueError(f"Invalid resume result: {dest}")
    for name, expected in result["outputs"].items():
        if file_digest(dest / name) != expected:
            raise ValueError(f"Changed output: {dest / name}")
    return result


def align_once(aligner, batch):
    # Drop exception traceback before OOM retry, so tensors can actually be freed.
    try:
        outputs = aligner.align(audio=[(x[1], 16000) for x in batch],
                                text=[x[0]["target_text"] for x in batch],
                                language=[x[0]["lang"] for x in batch])
        if len(outputs) != len(batch):
            raise RuntimeError("aligner_result_count_mismatch")
        return outputs, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:400]}"


def align_batch(aligner, batch, torch):
    outputs, error = align_once(aligner, batch)
    if error is None:
        return outputs
    print(f"ALIGN_RETRY n={len(batch)} {error}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    if len(batch) == 1:
        return [error]
    middle = len(batch) // 2
    return align_batch(aligner, batch[:middle], torch) + align_batch(aligner, batch[middle:], torch)


def load_audio(row):
    from vapasr.data.selection_audio import decode
    base = dict(schema=SCHEMA, key=row["key"], source=row["source"], lang=row["lang"],
                audio=row["audio"], target_text=row.get("recommended_training_target", {}).get("text"),
                training_eligible=False)
    try:
        text = target_text(row)
        base["target_sha256"] = text_sha(text)
        waveform, info = decode(row)
        if info["sha256"] != row["auto_selection"]["expected_waveform_sha256"]:
            raise ValueError("selection_waveform_sha256_mismatch")
        base.update(waveform_sha256=info["sha256"], duration_s=len(waveform) / 16000,
                    audio_review=info["review"])
        return base, waveform, None
    except Exception as exc:
        return base, None, f"{type(exc).__name__}: {str(exc)[:400]}"


def batches(rows, config):
    current, seconds = [], 0.0
    for row in sorted(rows, key=lambda r: (r["lang"], r["duration_s"])):
        if current and (len(current) >= config["batch"] or seconds + row["duration_s"] > config["batch_sec"]):
            yield current
            current, seconds = [], 0.0
        current.append(row)
        seconds += row["duration_s"]
    if current:
        yield current


def worker(a):
    import torch
    from qwen_asr import Qwen3ForcedAligner
    from transformers import AutoTokenizer
    config = json.loads((a.out / "config.json").read_text())
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "0"))
    if nodes < 1 or world != nodes * 8 or not 0 <= rank < world or torch.cuda.device_count() != 1:
        raise ValueError("Expected 8 Slurm tasks per node, each bound to exactly one allocated GPU")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["mem_frac"], 0)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True,
                                             use_fast=True, fix_mistral_regex=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer offsets required")
    aligner = None  # Resume-complete workers do not need to load weights.
    with ThreadPoolExecutor(config["io_threads"]) as pool:
        for spec in config["shards"][rank::world]:
            with locked(a.out / "locks" / (spec["name"] + ".lock")):
                source = Path(config["source"]) / "results" / spec["name"] / "keep-asr.jsonl"
                if file_digest(source) != spec["input_sha256"]:
                    raise ValueError(f"Selection input changed: {source}")
                if completed(a.out, spec, config["fingerprint"]):
                    print(f"SKIP rank={rank} {spec['name']}", flush=True)
                    continue
                rows = [json.loads(line) for line in source.read_text().splitlines()]
                if len(rows) != spec["selected_rows"] or len({r["key"] for r in rows}) != len(rows):
                    raise ValueError(f"Selection row accounting failed: {source}")
                rows = rows[:spec["rows"]]
                stage = Path(tempfile.mkdtemp(prefix=spec["name"] + "-", dir=a.out / "work"))
                counts, errors = Counter(), Counter()
                started = time.time()
                with (stage / "aligned.jsonl").open("w", encoding="utf-8") as ok, (stage / "failed.jsonl").open("w", encoding="utf-8") as bad:
                    def emit(base, error=None):
                        if error:
                            base.update(alignment_ok=False, error=error)
                            errors[error.split(":", 1)[0]] += 1
                        else:
                            base["alignment_ok"] = True
                        base.update(fingerprint=config["fingerprint"], source_shard=spec["name"])
                        (bad if error else ok).write(json.dumps(base, ensure_ascii=False) + "\n")
                        counts["failed" if error else "ok"] += 1
                    for group in batches(rows, config):
                        decoded = []
                        for base, waveform, error in pool.map(load_audio, group):
                            if error:
                                emit(base, error)
                            else:
                                decoded.append((base, waveform))
                        if not decoded:
                            continue
                        if aligner is None:
                            aligner = Qwen3ForcedAligner.from_pretrained(config["aligner"],
                                dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True)
                            aligner.model.eval()
                        outputs = align_batch(aligner, decoded, torch)
                        for (base, _), result in zip(decoded, outputs):
                            if isinstance(result, str):
                                emit(base, result)
                                continue
                            try:
                                items = [dict(text=it.text, start_time=it.start_time, end_time=it.end_time) for it in result.items]
                                base["aligner_items"] = items
                                encoding = tokenizer(base["target_text"], add_special_tokens=False, return_offsets_mapping=True)
                                base.update(project_items(base["target_text"], items, encoding, base["duration_s"]))
                            except Exception as exc:
                                emit(base, f"{type(exc).__name__}: {str(exc)[:400]}")
                            else:
                                emit(base)
                        print(f"PROGRESS rank={rank} {spec['name']} {sum(counts.values())}/{len(rows)} ok={counts['ok']} fail={counts['failed']}", flush=True)
                result = dict(complete=True, name=spec["name"], fingerprint=config["fingerprint"],
                              input_sha256=spec["input_sha256"], rows=len(rows), ok=counts["ok"],
                              failed=counts["failed"], errors=errors, elapsed_s=time.time() - started,
                              outputs={name: file_digest(stage / name) for name in ("aligned.jsonl", "failed.jsonl")})
                if sum(counts.values()) != len(rows):
                    raise RuntimeError("Alignment row accounting failed")
                save(stage / "summary.json", result)
                stage.rename(a.out / "results" / spec["name"])
                print(f"SHARD_DONE rank={rank} {spec['name']} {dict(counts)}", flush=True)


def summarize(a):
    config = json.loads((a.out / "config.json").read_text())
    results = [completed(a.out, s, config["fingerprint"]) for s in config["shards"]]
    finished = [r for r in results if r is not None]
    counts = {k: sum(r[k] for r in finished) for k in ("rows", "ok", "failed")}
    complete = len(finished) == len(results)
    result = dict(schema=SCHEMA, fingerprint=config["fingerprint"], scope=config["scope"],
                  complete=complete, completed_shards=len(finished), total_shards=len(results),
                  expected_rows=sum(s["rows"] for s in config["shards"]), **counts,
                  qc_status="required", training_eligible=False,
                  status="incomplete" if not complete else "complete_with_failures" if counts["failed"] else "complete_pending_qc")
    save(a.out / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not complete:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "worker", "summarize"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path)
    ap.add_argument("--aligner", type=Path, default=Path("/soundai/Model/Qwen3-ForcedAligner-0.6B"))
    ap.add_argument("--tokenizer", type=Path, default=Path("/soundai/Model/Qwen3-ASR-0.6B"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batch-sec", type=float, default=480)
    ap.add_argument("--io-threads", type=int, default=3)
    ap.add_argument("--mem-frac", type=float, default=.9)
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "prepare" and a.source is None:
        ap.error("prepare requires --source")
    {"prepare": prepare, "worker": worker, "summarize": summarize}[a.mode](a)


if __name__ == "__main__":
    main()
