#!/usr/bin/env python3
"""Second teacher for the YODAS-Granary German check: Whisper on the rows sampled by yodas_de_qwen_check.py.

  python experiments/yodas_de_whisper_check.py --root <yodas_granary/data> --results <check dir>/results.jsonl \
      --model <whisper dir> --name whisper-large-v3 --out <check dir>/whisper-large-v3.jsonl

Greedy decoding, language German, task transcribe. Segments longer than 30 s are decoded in long-form mode
(timestamps on, no truncation) so that Whisper sees the whole segment. Raw outputs are stored unchanged.
"""
import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path


def load_audio(root, rows):
    """{utt_id: (wave, sr)} for the sampled rows, read from their parquet files."""
    import pyarrow.parquet as pq
    import soundfile as sf
    want = defaultdict(set)
    for r in rows:
        want[(r["subset"], r["task"], r["file"])].add(r["utt_id"])
    out = {}
    for (sub, task, name), ids in want.items():
        f = pq.ParquetFile(Path(root) / sub / task / name)
        for g in range(f.metadata.num_row_groups):
            uids = f.read_row_group(g, columns=["utt_id"]).column(0).to_pylist()
            hit = [i for i, u in enumerate(uids) if u in ids]
            if not hit:
                continue
            col = f.read_row_group(g, columns=["audio"]).column(0)
            for i in hit:
                x, sr = sf.read(io.BytesIO(col[i].as_py()["bytes"]), dtype="float32")
                out[uids[i]] = (x.mean(axis=1) if x.ndim > 1 else x, sr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    rows = [json.loads(line) for line in open(a.results, encoding="utf-8")]
    t0 = time.monotonic()
    audio = load_audio(a.root, rows)
    print(f"loaded {len(audio)}/{len(rows)} clips in {time.monotonic() - t0:.0f} s", flush=True)
    proc = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(a.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                      local_files_only=True).cuda().eval()
    order = sorted((r["utt_id"] for r in rows if r["utt_id"] in audio), key=lambda u: len(audio[u][0]))
    hyp, t1 = {}, time.monotonic()
    for b in range(0, len(order), a.batch):
        ids = order[b:b + a.batch]
        wavs = [audio[u][0] for u in ids]
        long = any(len(w) > 30 * 16000 for w in wavs)
        inp = proc(wavs, sampling_rate=16000, return_tensors="pt", return_attention_mask=True,
                   truncation=not long, padding="longest" if long else "max_length")
        with torch.inference_mode():
            y = model.generate(input_features=inp.input_features.to("cuda", torch.float16),
                               attention_mask=inp.attention_mask.to("cuda"), language="de", task="transcribe",
                               return_timestamps=long, condition_on_prev_tokens=False, do_sample=False, num_beams=1)
        for u, t in zip(ids, proc.batch_decode(y, skip_special_tokens=True)):
            hyp[u] = t.strip()
    print(f"decoded {len(hyp)} clips in {time.monotonic() - t1:.0f} s", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"utt_id": r["utt_id"], "model": a.name, "hyp": hyp.get(r["utt_id"])},
                               ensure_ascii=False) + "\n")
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
