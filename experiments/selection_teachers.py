#!/usr/bin/env python3
"""One GPU teacher worker. Coordinator limits concurrency to four devices.

Append-only resumable outputs are fingerprint-bound to input, code, TN and model
weights. No record is promoted to training by this program.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import gc
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest, digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--teacher", choices=["qwen", "whisper"], required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--decode-workers", type=int, default=1)
    ap.add_argument("--memory-fraction", type=float, default=.85)
    a = ap.parse_args()
    if not (0 < a.memory_fraction <= .9) or a.batch < 1 or not 1 <= a.decode_workers <= 8:
        raise ValueError("invalid resource limit")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"START teacher={a.teacher} gpu={a.gpu} batch={a.batch}", flush=True)
    import torch
    from vapasr.data.selection_audio import decode
    from vapasr.data import textnorm
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    # Hash actual local weights, not just a mutable model-directory name.
    model_files = sorted(set(a.model.glob("*.safetensors")) | set(a.model.glob("*.json")))
    if not any(p.suffix == ".safetensors" for p in model_files):
        raise ValueError("No local safetensors model found")
    print("hashing model and input", flush=True)
    root = Path(__file__).resolve().parents[1]
    fp = dict(input_sha256=file_digest(a.input), teacher=a.teacher,
              model=str(a.model), model_hashes={p.name: file_digest(p) for p in model_files},
              code={str(p.relative_to(root)): file_digest(p) for p in
                    [Path(__file__).resolve(), *sorted((root / "vapasr/data").glob("*.py"))]},
              tn=textnorm.fingerprint(), packages={p: importlib.metadata.version(p) for p in
                    ("torch", "transformers", "qwen-asr", "num2words", "soundfile")},
              decoding=dict(batch=a.batch, decode_workers=a.decode_workers, qwen_max_new_tokens=448,
                            whisper_max_length=448, whisper_condition_on_prev_tokens=False,
                            timestamps="whisper_long_only", language="manifest",
                            task="transcribe", prompt=None))
    signature = digest(fp)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fp_path = a.output.with_suffix(".fingerprint.json")
    if fp_path.exists():
        if digest(json.loads(fp_path.read_text())) != signature:
            raise ValueError("Resume fingerprint mismatch; use new output path")
    else:
        if a.output.exists():
            raise ValueError("Unfingerprinted existing output")
        fp_path.write_text(json.dumps(fp, ensure_ascii=False, indent=2))
    done = set()
    if a.output.exists():
        with a.output.open() as f:
            for line in f:
                r = json.loads(line)  # truncated output fails closed
                done.add(r["key"])
    rows = []
    with a.input.open() as f:
        for line in f:
            r = json.loads(line)
            if r["key"] not in done and r["selection_state"] == "PENDING_AUDIO_TEACHERS":
                rows.append(r)
    rows.sort(key=lambda r: (r["lang"], r["duration_s"], r["key"]))
    if any(r["lang"] not in ("English", "Korean") for r in rows):
        raise ValueError("Unsupported input language; refusing to silently skip")
    print(f"pending={len(rows)} completed={len(done)}", flush=True)
    if not rows:
        return
    if a.teacher == "qwen":
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(str(a.model), dtype=torch.bfloat16,
            device_map="cuda", max_inference_batch_size=a.batch, max_new_tokens=448)
        processor = None
    else:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(str(a.model),
                    torch_dtype=torch.float16, low_cpu_mem_usage=True,
                    use_safetensors=True, attn_implementation="sdpa", local_files_only=True).cuda().eval()
        processor = AutoProcessor.from_pretrained(str(a.model), local_files_only=True)
    started, processed, audio_seconds = time.monotonic(), 0, 0.0

    def infer(batch):
        try:
            with torch.inference_mode():
                if a.teacher == "qwen":
                    res = model.transcribe(audio=[(x, 16000) for _, x, _ in batch],
                                           language=[r["lang"] for r, _, _ in batch])
                    return [(r.text, "generation_near_token_cap" if
                             len(model.processor.tokenizer.encode(r.text, add_special_tokens=False)) >= 430
                             else None) for r in res]
                long = any(len(x) > 30 * 16000 for _, x, _ in batch)
                inputs = processor([x for _, x, _ in batch], sampling_rate=16000,
                    return_tensors="pt", return_attention_mask=True,
                    truncation=False, padding="longest" if long else "max_length")
                features = inputs.input_features.to("cuda", dtype=torch.float16)
                result = model.generate(input_features=features,
                    attention_mask=inputs.attention_mask.to("cuda"),
                    language=batch[0][0]["lang"].lower(), task="transcribe",
                    return_timestamps=long, max_new_tokens=None, max_length=448,
                    condition_on_prev_tokens=False, do_sample=False, num_beams=1)
                texts = processor.batch_decode(result, skip_special_tokens=True)
                # Approximate token-cap warning, not a calibrated confidence.
                return [(t, ("long_form_requires_review" if long else
                             "generation_near_token_cap" if len(ids) >= 448 else None))
                        for t, ids in zip(texts, result)]
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
        # Leave the exception scope before retrying: its traceback can retain
        # the failed GPU tensors and make even a smaller batch fail again.
        if 'features' in locals(): del features
        if 'inputs' in locals(): del inputs
        gc.collect()
        torch.cuda.empty_cache()
        middle = len(batch) // 2
        print(f"OOM split {len(batch)} -> {middle}", flush=True)
        return infer(batch[:middle]) + infer(batch[middle:])

    def decode_safe(r):
        try:
            x, info = decode(r)
            return r, x, info, None
        except Exception as e:
            return r, None, None, f"{type(e).__name__}: {e}"

    with a.output.open("a", buffering=1) as out, ThreadPoolExecutor(max_workers=a.decode_workers) as pool:
        def emit(r):
            r["fingerprint"] = signature
            out.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")

        for lang, long_group in [(lang, long_group) for lang in ("English", "Korean")
                                 for long_group in (False, True)]:
            lang_rows = [r for r in rows if r["lang"] == lang
                         and (r["duration_s"] > 30) == long_group]
            pending_decode = [pool.submit(decode_safe, r) for r in lang_rows[:a.batch]]
            for start in range(0, len(lang_rows), a.batch):
                batch = []
                decoded = [future.result() for future in pending_decode]
                # Read the next batch while CUDA runs inference on this batch.
                pending_decode = [pool.submit(decode_safe, r) for r in
                                  lang_rows[start+a.batch:start+2*a.batch]]
                for r, x, info, error in decoded:
                    if error:
                        emit(dict(key=r["key"], teacher=a.teacher, status="audio_error",
                                  error=error))
                    else:
                        batch.append((r, x, info))
                if not batch:
                    continue
                try:
                    results = infer(batch)
                except Exception as e:
                    # Keep each failure explicit; never silently omit a batch.
                    for r, _, info in batch:
                        emit(dict(key=r["key"], teacher=a.teacher, status="inference_error",
                                  audio=info, error=f"{type(e).__name__}: {e}"))
                    print(f"INFERENCE ERROR: {e}", flush=True)
                    continue
                if len(results) != len(batch):
                    raise RuntimeError("teacher returned wrong batch size")
                for (r, _, info), (hyp, warning) in zip(batch, results):
                    flags = sorted(textnorm.target_flags(r["text"], r["lang"]))
                    emit(dict(key=r["key"], teacher=a.teacher, status="ok", hyp=hyp,
                              audio=info, tn_flags=flags, warning=warning,
                              confidence=None, confidence_note="not exported/calibrated"))
                    processed += 1
                    audio_seconds += info["duration_s"]
                print(json.dumps(dict(processed=processed, audio_h=audio_seconds/3600,
                    elapsed_s=time.monotonic()-started,
                    batch_rows=len(batch), audio_realtime_factor=audio_seconds/max(1e-9,time.monotonic()-started),
                    gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9,
                    gpu_reserved_peak_gb=torch.cuda.max_memory_reserved()/1e9)), flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
