#!/usr/bin/env python3
"""Offline baselines with oracle turn segmentation: quality upper bounds for `cstbench eval`.

Every turn is cut from the session mix with its reference start/end and translated after it ends.

  systems (--systems, comma separated)
    asr       Qwen3-ASR with the turn language given -> asr.jsonl (input of `cascade`)
    gold-mt   reference transcript -> LLM translation (no recognition errors)
    cascade   asr.jsonl transcripts -> LLM translation
    whisper   Whisper task=translate; X->English only (German turns)
`asr` and `cascade` may run in separate processes (e.g. different Python environments) on the same --out-dir.

Outputs in --out-dir: hyp-<system>.jsonl (with turn_id, t = turn end, elapsed = t + measured compute
time per turn) and asr.jsonl (cascade transcripts with WER against the corpus transcript).

  python baselines/offline_oracle.py --sessions <out>/taxi-natural/sessions.jsonl --out-dir runs/offline \
      --systems asr,gold-mt,cascade,whisper --asr-model Qwen/Qwen3-ASR-1.7B --llm-model <LLM> \
      --whisper-model openai/whisper-large-v3
Requirements beyond cst-bench: torch, transformers, qwen-asr (cascade).
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cstbench.audio import read_mono  # noqa: E402
from cstbench.evaluate import align, normalize  # noqa: E402

WHISPER_CODES = {"German": "de", "English": "en", "Korean": "ko", "Chinese": "zh"}


def prompt(src_lang, tgt_lang, text):
    return [{"role": "user", "content": f"Translate this {src_lang} utterance from a phone conversation into "
                                        f"{tgt_lang}. Output only the {tgt_lang} translation.\n\n{text}"}]


def load_turns(sessions_path, limit=0):
    """-> list of turn dicts with session, audio (float32, 16 kHz) cut from mix.wav."""
    root = Path(sessions_path).parent
    out = []
    for line in open(sessions_path, encoding="utf-8"):
        s = json.loads(line)
        x, sr = read_mono(root / s["mix"])
        for t in s["turns"]:
            if not t.get("translation"):
                continue
            a = x[int(round(t["start_s"] * sr)):int(round(t["end_s"] * sr))]
            out.append(dict(t, session=s["session_id"], audio=a, sr=sr))
            if limit and len(out) >= limit:
                return out
    return out


def wer(hyp, ref, lang):
    h, r = normalize(hyp, lang).split(), normalize(ref, lang).split()
    a = align(h, r)
    hits = sum(j >= 0 and h[i] == r[j] for i, j in enumerate(a))
    ins = sum(j < 0 for j in a)
    return (len(r) - hits + ins) / max(len(r), 1)              # (S + D + I) / N


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_asr(turns, model_path, device, batch):
    import torch
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device,
                                      max_inference_batch_size=batch, max_new_tokens=256)
    texts, secs = [], []
    for b in batched(turns, batch):
        t0 = time.monotonic()
        with torch.inference_mode():
            res = m.transcribe(audio=[(t["audio"], t["sr"]) for t in b], language=[t["lang"] for t in b])
        dt = (time.monotonic() - t0) / len(b)
        texts += [r.text for r in res]
        secs += [dt] * len(b)
    del m
    free()
    return texts, secs


def load_llm(path, device):
    import torch
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    arch = (transformers.AutoConfig.from_pretrained(path).architectures or ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch, None) if "CausalLM" not in arch else transformers.AutoModelForCausalLM
    cls = cls or transformers.AutoModelForCausalLM          # e.g. Qwen3.5-family ConditionalGeneration, text only
    model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device}, low_cpu_mem_usage=True).eval()
    return tok, model


def run_mt(tok, model, items, batch, max_new_tokens=160):
    """items: (src_lang, tgt_lang, text) -> (translations, seconds per item)."""
    import torch
    from transformers import GenerationConfig
    gen = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                           eos_token_id=model.generation_config.eos_token_id)
    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    out, secs = [None] * len(items), [0.0] * len(items)
    for b in batched(order, batch):
        texts = [tok.apply_chat_template(prompt(*items[i]), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for i in b]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.monotonic()
        with torch.inference_mode():
            y = model.generate(**enc, generation_config=gen)
        dt = (time.monotonic() - t0) / len(b)
        for i, row in zip(b, y[:, enc["input_ids"].shape[1]:]):
            s = tok.decode(row, skip_special_tokens=True).split("</think>")[-1].strip()
            out[i], secs[i] = (s.splitlines() or [""])[0].strip(), dt
    return out, secs


def run_whisper(turns, path, device, batch):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    m = AutoModelForSpeechSeq2Seq.from_pretrained(path, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    texts, secs = [], []
    for b in batched(turns, batch):
        feats = proc([t["audio"] for t in b], sampling_rate=b[0]["sr"], return_tensors="pt").input_features
        t0 = time.monotonic()
        with torch.inference_mode():
            y = m.generate(input_features=feats.to(device, dtype), task="translate",
                           language=WHISPER_CODES[b[0]["lang"]], max_new_tokens=200)
        dt = (time.monotonic() - t0) / len(b)
        texts += [s.strip() for s in proc.batch_decode(y, skip_special_tokens=True)]
        secs += [dt] * len(b)
    del m
    free()
    return texts, secs


def write_hyp(path, turns, texts, secs, system):
    with open(path, "w", encoding="utf-8") as f:
        for t, x, dt in zip(turns, texts, secs):
            f.write(json.dumps(dict(session=t["session"], turn_id=t["turn_id"], lang=t["translation_lang"], t=t["end_s"],
                                    elapsed=round(t["end_s"] + dt, 3), text=x, system=system), ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sessions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--systems", default="asr,gold-mt,cascade,whisper")
    p.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--llm-model")
    p.add_argument("--whisper-model", default="openai/whisper-large-v3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="first N turns only (smoke test)")
    a = p.parse_args()
    systems = a.systems.split(",")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    turns = load_turns(a.sessions, a.limit)
    print(f"turns {len(turns)} systems {systems}", flush=True)
    meta = dict(sessions=a.sessions, systems=systems, asr_model=a.asr_model, llm_model=a.llm_model,
                whisper_model=a.whisper_model, turns=len(turns))

    asr, asr_s = None, None
    if "asr" in systems:
        asr, asr_s = run_asr(turns, a.asr_model, a.device, a.batch)
        with open(out / "asr.jsonl", "w", encoding="utf-8") as f:
            for t, x, dt in zip(turns, asr, asr_s):
                f.write(json.dumps(dict(session=t["session"], turn_id=t["turn_id"], lang=t["lang"], ref=t["transcript"],
                                        hyp=x, wer=round(wer(x, t["transcript"], t["lang"]), 4), seconds=round(dt, 4)),
                                   ensure_ascii=False) + "\n")
        for lang in sorted({t["lang"] for t in turns}):
            idx = [i for i, t in enumerate(turns) if t["lang"] == lang]
            errs = sum(wer(asr[i], turns[i]["transcript"], lang) * max(len(normalize(turns[i]["transcript"], lang).split()), 1)
                       for i in idx)
            words = sum(max(len(normalize(turns[i]["transcript"], lang).split()), 1) for i in idx)
            meta[f"asr_wer_{lang}"] = round(errs / words, 4)
            print(f"ASR WER {lang}: {meta[f'asr_wer_{lang}']:.4f} ({len(idx)} turns)", flush=True)

    if "cascade" in systems and asr is None:
        rows = {(r["session"], r["turn_id"]): r for r in map(json.loads, open(out / "asr.jsonl", encoding="utf-8"))}
        asr = [rows[(t["session"], t["turn_id"])]["hyp"] for t in turns]
        asr_s = [rows[(t["session"], t["turn_id"])]["seconds"] for t in turns]
    if "gold-mt" in systems or "cascade" in systems:
        tok, model = load_llm(a.llm_model, a.device)
        for name, src in (("gold-mt", [t["transcript"] for t in turns]), ("cascade", asr)):
            if name not in systems:
                continue
            ys, secs = run_mt(tok, model, [(t["lang"], t["translation_lang"], x) for t, x in zip(turns, src)], a.batch)
            if name == "cascade":
                secs = [s + r for s, r in zip(secs, asr_s)]
            write_hyp(out / f"hyp-{name}.jsonl", turns, ys, secs, name)
            print(f"wrote hyp-{name}.jsonl", flush=True)
        del model
        free()

    if "whisper" in systems:
        sel = [t for t in turns if t["translation_lang"] == "English"]
        ys, secs = run_whisper(sel, a.whisper_model, a.device, a.batch)
        write_hyp(out / "hyp-whisper.jsonl", sel, ys, secs, "whisper")
        print("wrote hyp-whisper.jsonl", flush=True)
    (out / f"run_info-{'-'.join(systems)}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
