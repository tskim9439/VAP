#!/usr/bin/env python
"""Hugging Face Whisper 단일 오디오 추론과 선택적 TN/CER 기록."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import wave


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def edit_distance(hyp, ref):
    prev = list(range(len(ref) + 1))
    for i, x in enumerate(hyp, 1):
        cur = [i] + [0] * len(ref)
        for j, y in enumerate(ref, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def load_pcm_wave(path):
    """ffmpeg 없이 PCM WAV를 float32 mono 배열로 읽는다."""
    import numpy as np

    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        sampling_rate = f.getframerate()
        frames = f.readframes(f.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sampling_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="openai/whisper-large-v2")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--language", default="korean")
    p.add_argument("--reference-window", type=Path, default=None,
                   help="p2_eval_lanes window JSON; ref.episodes text를 연결")
    p.add_argument("--chunk-length", type=float, default=30.0)
    p.add_argument("--stride", type=float, default=5.0)
    p.add_argument("--native-long-form", action="store_true",
                   help="pipeline chunking 대신 Whisper generate의 long-form 경로 사용")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.out}")

    import torch
    from huggingface_hub import model_info, snapshot_download
    from transformers import (AutoModelForSpeechSeq2Seq, AutoProcessor,
                              pipeline)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    info = model_info(a.model)
    revision = info.sha
    model_path = snapshot_download(
        a.model, revision=revision,
        local_dir=str(a.model_dir) if a.model_dir else None,
    )
    dtype = torch.float16
    t0 = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        use_safetensors=True,
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(model_path)
    asr = None
    if not a.native_long_form:
        asr = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype, device=0,
            chunk_length_s=a.chunk_length,
            stride_length_s=(a.stride, a.stride),
        )
    loaded_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    audio, sampling_rate = load_pcm_wave(a.audio)
    if a.native_long_form:
        processed = processor(
            [audio],
            sampling_rate=sampling_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        input_features = processed.input_features.to(device="cuda", dtype=dtype)
        attention_mask = processed.attention_mask.to(device="cuda")
        generated = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language=a.language,
            task="transcribe",
            return_timestamps=True,
        )
        result = {
            "text": processor.batch_decode(
                generated, skip_special_tokens=True,
                decode_with_timestamps=False,
            )[0],
            "chunks": None,
        }
    else:
        result = asr(
            {"array": audio, "sampling_rate": sampling_rate},
            generate_kwargs={"language": a.language, "task": "transcribe"},
            return_timestamps=True,
        )
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - t1

    record = {
        "model": a.model,
        "revision": revision,
        "audio": str(a.audio.resolve()),
        "audio_sha256": sha256(a.audio),
        "language": a.language,
        "task": "transcribe",
        "decoding_mode": "native-long-form" if a.native_long_form else "pipeline-chunked",
        "chunk_length_s": None if a.native_long_form else a.chunk_length,
        "stride_length_s": None if a.native_long_form else [a.stride, a.stride],
        "raw_text": result["text"].strip(),
        "chunks": result.get("chunks"),
        "loaded_s": round(loaded_s, 3),
        "inference_s": round(inference_s, 3),
        "torch": torch.__version__,
    }
    if a.reference_window:
        from vapasr.data.textnorm import score_ko
        source = json.loads(a.reference_window.read_text())
        reference = " ".join(x["text"] for x in source["ref"]["episodes"])
        ref_norm = score_ko(reference, spaces=False)
        hyp_norm = score_ko(record["raw_text"], spaces=False)
        errors = edit_distance(list(hyp_norm), list(ref_norm))
        record.update(
            reference=reference,
            reference_normalized=ref_norm,
            hypothesis_normalized=hyp_norm,
            cer=errors / len(ref_norm) if ref_norm else None,
            cer_errors=errors,
            cer_reference_characters=len(ref_norm),
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
