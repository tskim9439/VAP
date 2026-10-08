#!/usr/bin/env python3
"""YODAS-Granary German, GPU stage: Qwen3-ASR transcription, selection, forced alignment and export.

Input: <out>/prefilter/prefilter.parquet from `yodas_de_prepare.py prefilter` (rows without flags are candidates;
both the train and the held-out dev split are processed). One shard = one source parquet file.
Per shard:
  1. Qwen3-ASR-1.7B (language German); the raw output is stored unchanged.
  2. keep iff WER(norm_de(Granary text), norm_de(Qwen text)) <= --max-wer (0.03); the target is the raw Qwen text
     (decisions: wiki/outputs/output-yodas-granary-de-transcript-check-20261008.md).
  3. Qwen3-ForcedAligner-0.6B on the Qwen text (language German) -> vapasr.data.selection_alignment.project_items;
     a clip-end overshoot of at most 80 ms is clamped (selection_alignment_qc.classify_failed, as for English/Korean).
  4. words rows built with the speechlm candidate builder's checks (build_candidate) and set "yodas-de"; the WAV
     bytes of kept rows are copied unchanged into an uncompressed tar with <tar>.index.json next to it
     ({member: [data offset, size]}, the format vapasr.data.archive reads).
Passes: the main pass takes unflagged prefilter rows (1-30 s). `--select-flags dur_long --max-dur 40` is the long pass:
rows flagged only dur_long (30-40 s, no other flag) in shards named <shard>-dur_long, summary align-summary-dur_long.json.
The two passes write disjoint shard folders under the same <out> and may run at the same time.
Outputs <out>/results/<shard>/: records.jsonl (every candidate: texts, WER, state), words-train.jsonl,
words-dev.jsonl, failed.jsonl, audio.tar, audio.tar.index.json, summary.json. A shard is staged in <out>/work/ and
renamed into results/ when complete, so a rerun (e.g. after preemption) skips finished shards. A shard that raises is
recorded in <out>/errors/ and retried by the next run. Nothing is deleted: an interrupted attempt leaves its staging
folder under work/.

  python experiments/yodas_de_align.py worker --out <dir> [--rank R --world W]   (SLURM_PROCID/SLURM_NTASKS by default)
  python experiments/yodas_de_align.py summarize --out <dir>
"""
import argparse
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from yodas_de_qwen_check import edit_ops, norm_de  # noqa: E402

SOURCE = "/soundai/users/tskim/VAPKT-data/data/corpora/data/yodas_granary/data"
SET_NAME, LANG = "yodas-de", "German"


CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def flag_set(flags):
    return frozenset(x for x in (flags or "").split(",") if x)


def pass_tag(select_flags):
    """'' for the main pass (unflagged rows); e.g. 'dur_long' for the pass over rows flagged only dur_long."""
    return "+".join(sorted(flag_set(select_flags)))


def shard_name(subset, task, file, tag=""):
    return f"{subset}-{task}-{Path(file).stem}" + (f"-{tag}" if tag else "")


def load_candidates(out, select_flags=""):
    """Candidate rows grouped by source file: rows whose prefilter flags are exactly `select_flags` ('' = unflagged)."""
    import pyarrow.parquet as pq
    t = pq.read_table(Path(out) / "prefilter" / "prefilter.parquet").to_pydict()
    want = flag_set(select_flags)
    shards = defaultdict(list)
    for i in range(len(t["utt_id"])):
        if flag_set(t["flags"][i]) != want:
            continue
        shards[(t["subset"][i], t["task"][i], t["file"][i])].append(
            dict(row_group=t["row_group"][i], row=t["row"][i], utt_id=t["utt_id"][i], duration=t["duration"][i],
                 split=t["split"][i]))
    return dict(sorted(shards.items()))


def read_shard(source, key, cands):
    """Audio bytes, decoded 16 kHz waveform and Granary text of the candidate rows of one source file."""
    import pyarrow.parquet as pq
    import soundfile as sf
    sub, task, name = key
    f = pq.ParquetFile(Path(source) / sub / task / name)
    by_group = defaultdict(list)
    for c in cands:
        by_group[c["row_group"]].append(c)
    items = []
    for g, cs in sorted(by_group.items()):
        t = f.read_row_group(g, columns=["utt_id", "audio", "text"])
        uids, audio, text = t.column("utt_id"), t.column("audio"), t.column("text")
        for c in cs:
            if uids[c["row"]].as_py() != c["utt_id"]:
                raise ValueError(f"row mismatch {name} rg{g} row{c['row']}")
            wav = audio[c["row"]].as_py()["bytes"]
            x, sr = sf.read(io.BytesIO(wav), dtype="float32")
            if sr != 16000 or x.ndim != 1:
                items.append(dict(c, ref=text[c["row"]].as_py() or "", wav=None, x=None, error=f"audio sr={sr} ndim={x.ndim}"))
                continue
            items.append(dict(c, ref=text[c["row"]].as_py() or "", wav=wav, x=x, error=None))
    return items


def batches(items, max_n, max_sec):
    """Duration-sorted batches with at most max_n items and max_sec seconds of audio."""
    order = sorted(items, key=lambda it: len(it["x"]))
    cur, sec = [], 0.0
    for it in order:
        d = len(it["x"]) / 16000
        if cur and (len(cur) >= max_n or sec + d > max_sec):
            yield cur
            cur, sec = [], 0.0
        cur.append(it)
        sec += d
    if cur:
        yield cur


def run_split(fn, batch):
    """Run fn on a batch; on failure split it in half and retry, down to single items (errors are returned)."""
    try:
        return fn(batch)
    except Exception as exc:  # noqa: BLE001
        if len(batch) == 1:
            return [dict(error=f"{type(exc).__name__}: {str(exc)[:300]}")]
        mid = len(batch) // 2
        return run_split(fn, batch[:mid]) + run_split(fn, batch[mid:])


class Engines:
    def __init__(self, a):
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        from transformers import AutoTokenizer
        from vapasr.data.semcommit_words import PieceVocab
        self.torch = torch
        self.asr = Qwen3ASRModel.from_pretrained(a.asr_model, dtype=torch.bfloat16, device_map="cuda:0",
                                                 max_inference_batch_size=a.batch, max_new_tokens=448)
        self.aligner = Qwen3ForcedAligner.from_pretrained(a.aligner_model, dtype=torch.bfloat16, device_map="cuda:0",
                                                          local_files_only=True)
        self.aligner.model.eval()
        self.tok = AutoTokenizer.from_pretrained(a.asr_model, use_fast=True, fix_mistral_regex=True, local_files_only=True)
        if not self.tok.is_fast:
            raise ValueError("fast tokenizer required for character offsets")
        self.pieces = PieceVocab(a.asr_model)

    def transcribe(self, batch):
        with self.torch.inference_mode():
            res = self.asr.transcribe(audio=[(it["x"], 16000) for it in batch], language=[LANG] * len(batch))
        out = []
        for r in res:
            cap = len(self.tok.encode(r.text, add_special_tokens=False)) >= 430
            out.append(dict(text=r.text, warning="token_cap" if cap else None))
        if len(out) != len(batch):
            raise ValueError("ASR output count mismatch")
        return out

    def align(self, batch):
        with self.torch.inference_mode():
            res = self.aligner.align(audio=[(it["x"], 16000) for it in batch], text=[it["hyp"] for it in batch],
                                     language=[LANG] * len(batch))
        out = [dict(items=[dict(text=i.text, start_time=i.start_time, end_time=i.end_time) for i in r.items]) for r in res]
        if len(out) != len(batch):
            raise ValueError("aligner output count mismatch")
        return out


def wer(ref, hyp):
    r, h = norm_de(ref), norm_de(hyp)
    return edit_ops(r, h) / max(1, len(r)), len(r)


def words_row(it, items, eng, tar_path, duration):
    """Projected timing + speechlm candidate checks -> words row, or (None, reason)."""
    from semcommit_build_speechlm_candidates import build_candidate
    from vapasr.data.selection_alignment import project_items
    from vapasr.data.selection_alignment_qc import classify_failed
    text = it["hyp"]
    enc = eng.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    qc = None
    try:
        proj = project_items(text, items, enc, duration)
    except ValueError as exc:
        # Same policy as the English/Korean pipeline (qwen-verbatim-align-qc-v1): a clip-end overshoot of at most
        # 80 ms is clamped; 80-320 ms needs review and anything else is rejected.
        res = classify_failed(dict(alignment_ok=False, error=f"ValueError: {exc}", duration_s=duration,
                                   aligner_items=items, target_text=text), enc)
        if res["state"] != "ACCEPT_CLAMPED_80MS":
            return None, f"{res['state']}:{res['reason']}"
        proj, qc = res["corrected"], res["corrected"]["qc"]
    aligned = dict(key=it["utt_id"], source=SET_NAME, lang=LANG, audio=dict(path=f"{tar_path}::{it['utt_id']}.wav"),
                   target_text=text, duration_s=duration, alignment_ok=True, training_eligible=False,
                   dedup_status="unverified", source_shard=it["shard"], timing_quality=proj.get("timing_quality"),
                   **{k: proj[k] for k in ("words", "tokens")})
    record = dict(key=it["utt_id"], selection_state="KEEP_ASR_SILVER", split="train",
                  source_manifest=dict(split="train", num_channels=1, corpus="yodas-granary-de",
                                       original_transcript=it["ref"]))
    row, why = build_candidate(aligned, record, tokenizer=eng.pieces, max_duration_s=it["max_dur"])
    if row is None:
        return None, why
    if qc:
        row["alignment_qc"] = qc
    row.update(id=f"yde-{it['utt_id']}", set=SET_NAME, candidate_only=False, training_eligible=True,
               selection="qwen_vs_granary_wer_le_%g" % it["max_wer"], wer_vs_granary=it["wer"],
               source_corpus="yodas-granary-de", split=it["split"])
    return row, None


def write_tar(tar_path, members):
    """members: [(name, bytes)] -> uncompressed tar + <tar>.index.json {name: [data offset, size]}."""
    with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
    idx = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.isfile():
                idx[m.name] = [m.offset_data, m.size]
    Path(str(tar_path) + ".index.json").write_text(json.dumps(idx))
    return idx


def jline(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_shard(key, cands, a, eng, final_dir):
    name = shard_name(*key, tag=a.tag)
    # One staging folder per attempt; an attempt cut short by preemption leaves its folder behind (never deleted here).
    stage = Path(a.out) / "work" / f"{name}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.{int(time.time())}"
    stage.mkdir(parents=True)
    t0 = time.monotonic()
    items = read_shard(a.source, key, cands)
    for it in items:
        it.update(shard=name, max_wer=a.max_wer, max_dur=a.max_dur)
    ok = [it for it in items if it["x"] is not None]
    for b in batches(ok, a.batch, a.batch_sec):
        for it, r in zip(b, run_split(eng.transcribe, b)):
            it["hyp"], it["asr_warning"], it["asr_error"] = r.get("text"), r.get("warning"), r.get("error")
    states = Counter()
    keep = []
    for it in items:
        if it["x"] is None:
            it["state"] = "HOLD_AUDIO"
        elif it.get("asr_error") or it.get("hyp") is None:
            it["state"] = "HOLD_ASR_ERROR"
        elif it.get("asr_warning"):
            it["state"] = "HOLD_TOKEN_CAP"
        else:
            it["wer"], it["ref_words"] = wer(it["ref"], it["hyp"])
            it["wer"] = round(it["wer"], 4)
            if not norm_de(it["hyp"]):
                it["state"] = "DROP_EMPTY"
            elif it["wer"] <= a.max_wer:
                it["state"] = "KEEP"
                keep.append(it)
            else:
                it["state"] = "DROP_DISAGREE"
        states[it["state"]] += 1
    tar_rel = final_dir / "audio.tar"
    rows, failed, members = {"train": [], "dev": []}, [], []
    for b in batches(keep, a.batch, a.batch_sec):
        for it, r in zip(b, run_split(eng.align, b)):
            duration = len(it["x"]) / 16000
            if r.get("error"):
                failed.append(dict(utt_id=it["utt_id"], reason="aligner_error", error=r["error"]))
                it["state"] = "FAIL_ALIGN"
                continue
            try:
                row, why = words_row(it, r["items"], eng, tar_rel, duration)
            except Exception as exc:  # noqa: BLE001  (project_items fails closed with ValueError)
                row, why = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            if row is None:
                failed.append(dict(utt_id=it["utt_id"], reason=why, duration_s=duration, target_text=it["hyp"],
                                   aligner_items=r["items"]))
                it["state"] = "FAIL_ALIGN"
                continue
            rows[it["split"]].append(row)
            members.append((f"{it['utt_id']}.wav", it["wav"]))
            it["state"] = "ALIGNED"
    states = Counter(it["state"] for it in items)
    if members:
        write_tar(stage / "audio.tar", members)
    with open(stage / "records.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            jline(f, {k: it.get(k) for k in ("utt_id", "split", "duration", "ref", "hyp", "wer", "ref_words", "state",
                                             "asr_warning", "asr_error", "error")})
    for split in ("train", "dev"):
        with open(stage / f"words-{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows[split]:
                jline(f, row)
    with open(stage / "failed.jsonl", "w", encoding="utf-8") as f:
        for x in failed:
            jline(f, x)
    hours = defaultdict(float)
    for it in items:
        hours[it["state"]] += it["duration"] / 3600
    summary = dict(complete=True, shard=name, source=list(key), candidates=len(items), states=dict(states),
                   hours={k: round(v, 4) for k, v in hours.items()}, words={k: len(v) for k, v in rows.items()},
                   failed_reasons=dict(Counter(x["reason"] for x in failed)), max_wer=a.max_wer,
                   select_flags=a.tag, max_dur=a.max_dur,
                   asr_model=a.asr_model, aligner_model=a.aligner_model, elapsed_s=round(time.monotonic() - t0, 1),
                   code_sha256=CODE_SHA256)
    (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    final = Path(a.out) / "results" / name
    if final.exists():
        raise FileExistsError(f"{final} already exists; refusing to replace a finished shard")
    stage.rename(final)
    return summary


def cmd_worker(a):
    rank = a.rank if a.rank is not None else int(os.environ.get("SLURM_PROCID", 0))
    world = a.world if a.world is not None else int(os.environ.get("SLURM_NTASKS", 1))
    a.tag = pass_tag(a.select_flags)
    shards = load_candidates(a.out, a.tag)
    keys = list(shards)[rank::world]
    todo = [k for k in keys if not (Path(a.out) / "results" / shard_name(*k, tag=a.tag) / "summary.json").exists()]
    print(f"rank {rank}/{world}: pass '{a.tag or 'main'}' max_dur {a.max_dur}: {len(keys)} shards, {len(todo)} to do",
          flush=True)
    if not todo:
        return
    eng = Engines(a)
    (Path(a.out) / "results").mkdir(parents=True, exist_ok=True)
    for k in todo:
        name = shard_name(*k, tag=a.tag)
        final_dir = Path(a.out) / "results" / name
        try:
            s = process_shard(k, shards[k], a, eng, final_dir)
        except Exception as exc:  # noqa: BLE001
            # One bad shard must not stop the other ranks (srun --kill-on-bad-exit): record it and go on. The shard has
            # no results/ folder, so the next submission retries it.
            err = Path(a.out) / "errors"
            err.mkdir(exist_ok=True)
            (err / f"{name}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.json").write_text(
                json.dumps(dict(shard=name, error=f"{type(exc).__name__}: {exc}",
                                traceback=traceback.format_exc()), ensure_ascii=False, indent=1))
            print(f"SHARD_ERROR rank={rank} {name} {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        print(f"SHARD_DONE rank={rank} {s['shard']} cand={s['candidates']} states={json.dumps(s['states'])} "
              f"{s['elapsed_s']} s", flush=True)


def cmd_summarize(a):
    tag = pass_tag(a.select_flags)
    shards = load_candidates(a.out, tag)
    states, hours, words, reasons = Counter(), Counter(), Counter(), Counter()
    done = 0
    for k in shards:
        p = Path(a.out) / "results" / shard_name(*k, tag=tag) / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        done += 1
        states.update(s["states"])
        hours.update(s["hours"])
        words.update(s["words"])
        reasons.update(s["failed_reasons"])
    errors = sorted(p.name for p in (Path(a.out) / "errors").glob("*.json")) if (Path(a.out) / "errors").exists() else []
    errors = [e for e in errors if e.split(".")[0] in {shard_name(*k, tag=tag) for k in shards}]
    out = dict(pass_name=tag or "main", shards=len(shards), done=done, remaining=len(shards) - done, error_files=len(errors), states=dict(states), hours={k: round(v, 2) for k, v in hours.items()},
               words=dict(words), failed_reasons=dict(reasons.most_common()))
    (Path(a.out) / (f"align-summary-{tag}.json" if tag else "align-summary.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("worker")
    q.add_argument("--out", required=True)
    q.add_argument("--source", default=SOURCE)
    q.add_argument("--rank", type=int)
    q.add_argument("--world", type=int)
    q.add_argument("--max-wer", type=float, default=0.03)
    q.add_argument("--select-flags", default="", help="rows whose prefilter flags are exactly these (comma list); "
                   "'' = unflagged rows (main pass), 'dur_long' = the long pass (30-40 s rows with no other flag)")
    q.add_argument("--max-dur", type=float, default=30.0, help="longest kept row in seconds (40 for the long pass)")
    q.add_argument("--asr-model", default="/soundai/Model/Qwen3-ASR-1.7B")
    q.add_argument("--aligner-model", default="/soundai/Model/Qwen3-ForcedAligner-0.6B")
    q.add_argument("--batch", type=int, default=64)
    q.add_argument("--batch-sec", type=float, default=960.0)
    q.set_defaults(fn=cmd_worker)
    q = sub.add_parser("summarize")
    q.add_argument("--out", required=True)
    q.add_argument("--select-flags", default="")
    q.set_defaults(fn=cmd_summarize)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
