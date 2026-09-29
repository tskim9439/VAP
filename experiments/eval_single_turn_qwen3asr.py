#!/usr/bin/env python3
"""single-turn v1 manifest 를 오프라인 Qwen3-ASR(qwen-asr 패키지, transformers 백엔드, bf16)로 전사해 같은 채점으로 남긴다 — 스트리밍 모델의 상한 참고선.

입력은 eval_single_turn_asr.read_audio 와 같다(원 발화 + tail 디지털 무음). language 는 manifest 의 lang 으로 고정한다.
결과 폴더는 `eval_single_turn_asr.py run` 과 같은 형식(config-rank0.json · predictions-rank0.jsonl · done-rank0.json, delta="offline")이고,
끝나면 같은 summarize(micro groups + 영어 AA-WER)를 부른다. 같은 명령을 다시 돌리면 끝난 발화는 건너뛴다(설정이 다르면 거부).
  CUDA_VISIBLE_DEVICES=0 python experiments/eval_single_turn_qwen3asr.py --manifest <manifest.jsonl> --model /soundai/Model/Qwen3-ASR-1.7B --out <dir>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vapasr.data.single_turn_eval import PROTOCOL, digest, fingerprint, read_jsonl, score_pair, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True, help="Qwen3-ASR 디렉토리(예: /soundai/Model/Qwen3-ASR-1.7B)")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tail-s", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--datasets", default="", help="쉼표 목록이면 그 세트만(예: voxpopuli-aa-test)")
    a = p.parse_args()
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("est", ROOT / "experiments/eval_single_turn_asr.py")
    est = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(est)
    from vapasr.data.textnorm import score_en, score_ko

    rows = read_jsonl(a.manifest)
    if a.datasets:
        keep = set(a.datasets.split(","))
        rows = [r for r in rows if r["dataset"] in keep]
        assert rows, f"--datasets {a.datasets}: manifest 에 없음"
    md = Path(a.model)
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in
            ["experiments/eval_single_turn_qwen3asr.py", "experiments/eval_single_turn_asr.py", "vapasr/data/single_turn_eval.py"]}
    config = dict(protocol=PROTOCOL, system="qwen3-asr-offline", checkpoint=str(md),
                  checkpoint_weights={q.name: dict(size=q.stat().st_size, mtime_ns=q.stat().st_mtime_ns) for q in sorted(md.glob("*.safetensors"))},
                  manifest_digest=digest(rows), utterances=len(rows), deltas=["offline"], tail_s=a.tail_s, leading_silence_s=0,
                  dtype="bfloat16", max_new_tokens=a.max_new_tokens, textnorm=fingerprint(), code=code, world_size=1, batch_size=a.batch_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cp = out / "config-rank0.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == config, "Resume protocol mismatch; use a fresh output directory"
    else:
        write_json(cp, config)
    dest = out / "predictions-rank0.jsonl"
    done = {r["id"] for r in read_jsonl(dest)} if dest.exists() else set()
    todo = sorted((r for r in rows if r["id"] not in done), key=lambda r: -r["audio_s"])      # 긴 것부터: 메모리 부족을 초반에 드러낸다
    from qwen_asr import Qwen3ASRModel
    t0 = time.monotonic()
    model = Qwen3ASRModel.from_pretrained(str(md), dtype=torch.bfloat16, device_map="cuda:0",
                                          max_inference_batch_size=a.batch_size, max_new_tokens=a.max_new_tokens)
    print(f"START model={md} load_s={time.monotonic() - t0:.0f} remaining={len(todo)}/{len(rows)}", flush=True)
    with dest.open("a", buffering=1) as f:
        for i in range(0, len(todo), a.batch_size):
            batch = todo[i:i + a.batch_size]
            bt = time.monotonic()
            res = model.transcribe(audio=[(est.read_audio(r, a.tail_s), 16000) for r in batch], language=[r["lang"] for r in batch])
            for r, t in zip(batch, res):
                hyp = (t.text or "").strip()
                norm = score_en if r["lang"] == "English" else (lambda x: score_ko(x, False))
                f.write(json.dumps(dict(id=r["id"], dataset=r["dataset"], lang=r["lang"], path=r["path"], delta="offline",
                                        audio_s=r["audio_s"], tail_s=a.tail_s, raw_reference=r["raw_reference"], reference=r["reference"],
                                        hypothesis=hyp, reference_normalized=norm(r["reference"]), hypothesis_normalized=norm(hyp),
                                        primary_metric="wer" if r["lang"] == "English" else "cer_nospace",
                                        metrics=score_pair(r["reference"], hyp, r["lang"]), forced=0,
                                        detected_language=getattr(t, "language", None)), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"PROGRESS {min(i + a.batch_size, len(todo))}/{len(todo)} batch_s={time.monotonic() - bt:.1f} "
                  f"elapsed_s={time.monotonic() - t0:.0f}", flush=True)
    write_json(out / "done-rank0.json", dict(predictions=len(read_jsonl(dest)), elapsed_s=time.monotonic() - t0))
    est.summarize(argparse.Namespace(out=str(out), allow_partial=False))


if __name__ == "__main__":
    main()
