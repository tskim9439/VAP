#!/usr/bin/env python
"""VibeVoice 공식 step API로 청크별 추론. final speaker text와 chunk 가용 시각 보존.

speed=throughput: 최대 속도 처리(RTF만). speed=realtime: 관측 가능 시점까지 기다림.
청크 emit 시각은 단어 timestamp/EOT가 아니다. scorer는 별도 CPU 실행 가능.
"""
import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.baseline_eval import (CODE_REVISION, MODEL_REVISION, parse_speakers,
                                      read_manifest, sha256)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--vibevoice-source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--speed", choices=("throughput", "realtime"), default="throughput")
    p.add_argument("--max-new-tokens", type=int, default=256)
    a = p.parse_args()
    if a.max_new_tokens < 1: p.error("max-new-tokens must be positive")
    rows = read_manifest(a.manifest)
    rev = subprocess.check_output(["git", "-C", str(a.vibevoice_source), "rev-parse", "HEAD"], text=True).strip()
    if rev != CODE_REVISION:
        raise ValueError("VibeVoice code revision mismatch")
    subprocess.run(["git", "-C", str(a.vibevoice_source), "diff", "--quiet", "HEAD", "--", "vibevoice"], check=True)
    provenance = json.loads((a.model / "baseline-provenance.json").read_text())
    if provenance["revision"] != MODEL_REVISION:
        raise ValueError("Model revision mismatch")
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from scipy.signal import resample_poly
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    source = Path(inspect.getfile(VibeVoiceASRForConditionalGeneration)).resolve()
    if not source.is_relative_to(a.vibevoice_source.resolve()):
        raise ValueError(f"Unexpected imported VibeVoice source: {source}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; use scorer/tests for CPU-only validation")
    cfg = json.loads((a.model / "preprocessor_config.json").read_text())
    sr = cfg["target_sample_rate"]
    chunk_n = cfg["chunk_frames"] * cfg["speech_tok_compress_ratio"]
    look_n = cfg["lookahead_frames"] * cfg["speech_tok_compress_ratio"]
    run = dict(manifest_sha256=sha256(a.manifest), model_revision=MODEL_REVISION,
               code_revision=rev, model_path=str(a.model.resolve()), source_sha256=sha256(source),
               runner_sha256=sha256(__file__),
               adapter_sha256=sha256(Path(__file__).resolve().parents[1] / "vapasr/data/baseline_eval.py"),
               preprocessor=cfg, speed=a.speed, max_new_tokens=a.max_new_tokens, seed=0,
               timestamp_semantics="chunk_complete_emission_not_speech_boundary",
               versions={n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy", "scipy")})
    a.out.mkdir(parents=True, exist_ok=True)
    run_path = a.out / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Output contains different run; choose a new --out")
    run_path.write_text(json.dumps(run, indent=2))
    torch.manual_seed(0)
    processor = VibeVoiceASRProcessor.from_pretrained(str(a.model))
    if processor.tokenizer.text_chunk_end_id is None:
        raise ValueError("Not a streaming tokenizer")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        str(a.model), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    class TokenizerTrace:
        def __init__(self, tok): self.tok, self.n = tok, 0
        def __getattr__(self, key): return getattr(self.tok, key)
        def decode(self, ids, **kw):
            self.n = len(ids)
            return self.tok.decode(ids, **kw)

    tok = TokenizerTrace(processor.tokenizer)
    for index, row in enumerate(rows):
        dest = a.out / f"session-{index:04d}.json"
        if dest.exists():
            old = json.loads(dest.read_text())
            if old["session_id"] != row["session_id"] or old["audio_sha256"] != row["audio_sha256"]:
                raise ValueError("Existing session result mismatch")
            print("skip complete", row["session_id"], flush=True)
            continue
        if sha256(row["audio"]) != row["audio_sha256"]:
            raise ValueError("Audio hash changed")
        wav, input_sr = sf.read(row["audio"], dtype="float32")
        if wav.ndim != 1: raise ValueError("Only shared mono input allowed")
        if abs(len(wav)/input_sr - row["duration_s"]) > 1/input_sr:
            raise ValueError("Duration mismatch")
        if input_sr != sr:
            g = math.gcd(input_sr, sr)
            wav = resample_poly(wav, sr//g, input_sr//g)
        # 오디오 resampling은 공통 입력 준비. 이를 live capture 지연이라고 보고하지 않는다.
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = model.init_streaming_state(tok, context_info=None)
        torch.cuda.synchronize()
        setup_s = time.perf_counter() - t0
        chunks = []
        for k, start in enumerate(range(0, len(wav), chunk_n)):
            stop = min(start + chunk_n + look_n, len(wav))
            observed = stop / sr
            if a.speed == "realtime":
                time.sleep(max(0.0, t0 + observed - time.perf_counter()))
            x = torch.from_numpy(np.ascontiguousarray(wav[start:stop])).unsqueeze(0).cuda()
            x = F.pad(x, (0, chunk_n + look_n - x.shape[-1]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            first = []
            def first_token(_): first.append(time.perf_counter() - t0)
            with torch.inference_mode():
                features = model.encode_speech(x)
                text, state = model.streaming_generate_step(features, state, tok,
                    max_new_tokens=a.max_new_tokens, temperature=0.0, on_first_token=first_token)
            torch.cuda.synchronize()
            end = time.perf_counter()
            chunks.append(dict(chunk=k, audio_observed_until_s=observed, text=text,
                               emitted_elapsed_s=end-t0, first_token_elapsed_s=first[0] if first else None,
                               compute_s=end-begin, text_tokens=tok.n, cap_hit=tok.n >= a.max_new_tokens,
                               realtime_release_lag_s=end-t0-observed if a.speed == "realtime" else None))
            print(json.dumps(dict(session=row["session_id"], chunk=k, text=text,
                                  compute_s=round(end-begin, 3)), ensure_ascii=False), flush=True)
        elapsed = time.perf_counter()-t0
        result = dict(session_id=row["session_id"], audio_sha256=row["audio_sha256"],
                      chunks=chunks, segments=parse_speakers([c["text"] for c in chunks]),
                      duration_s=row["duration_s"], elapsed_s=elapsed, setup_s=setup_s,
                      throughput_rtf=elapsed/row["duration_s"] if a.speed == "throughput" else None,
                      peak_vram_bytes=torch.cuda.max_memory_allocated(),
                      native_metrics=dict(der=None, onset=None, offset=None, eot=None),
                      word_latency=None, word_latency_reason="no_native_word_timestamps_or_alignment")
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        os.replace(tmp, dest)
        del state
    (a.out / "INFERENCE_DONE").write_text("all manifest sessions finished\n")


if __name__ == "__main__":
    main()
