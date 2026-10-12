#!/usr/bin/env python3
"""Common Voice 4.0 English read speech (CoVoST 2 en_de train, fixie-ai/covost2 parquet) → ASR words rows (2026-10-11, 사용자 결정:
영어 낭독 데이터 추가). 구조는 YODAS 독일어 파이프라인(experiments/yodas_de_prepare.py · yodas_de_align.py)과 같다.

  prefilter  (CPU, 오디오 없이) parquet 메타데이터 → <out>/prefilter/prefilter.parquet + summary.json
             타깃 = vapasr.data.textnorm.target_en(sentence)(LibriSpeech 와 같은 asr-tn-v1 lexical, 숫자 변환 없음).
             flags: tn_<사유>(target_flags: digit·charset·empty 등) · eval_overlap(평가 문장과 단어 8-gram 공유) ·
             eval_exact(평가 문장과 정규형 전체 일치 — CV 문장은 짧아 8-gram 만으로는 놓친다) · duplicate_text(같은 화자·같은 문장).
             split: --dev-speakers 명의 화자(client_id 의 안정 해시 순)는 dev, 나머지 train. CoVoST 2 의 test·dev 는 받지 않았다(화자 분리).
  worker     (GPU) shard = (parquet 파일, row group 16 개 묶음). mp3 → 16 kHz 모노 → 길이 [min_dur, max_dur] 확인 → Qwen3-ASR-1.7B(English) →
             WER(score_en(타깃), score_en(Qwen)) ≤ --max-wer 이면 KEEP(타깃은 사람 문장의 lexical 형, Qwen 출력은 검증용) →
             Qwen3-ForcedAligner-0.6B(타깃) → words 행(build_candidate, YODAS 와 같은 검사·80 ms 끝 넘침 보정) + 16 kHz PCM16 wav 의 tar.
             결과는 <out>/results/<shard>/ (records·words-train·words-dev·failed·audio.tar(+index)·summary). 끝난 shard 는 건너뛰고
             (선점 재시작), 실패 shard 는 <out>/errors/ 에 남겨 다음 실행이 다시 한다. 아무것도 지우지 않는다.
  summarize  결과 합계 → <out>/align-summary.json
  lists      words 목록 → <out>/lists/cv4-en-words-{train,dev}.list (학습 --asr-list 에 덧붙일 한 줄 한 파일)

  python experiments/cv_en_align.py prefilter --source <covost2-fixie/en_de> --out <dir> --decon-manifest M [M ...]
  python experiments/cv_en_align.py worker --out <dir> [--rank R --world W]     (SLURM_PROCID/SLURM_NTASKS 기본)
  python experiments/cv_en_align.py summarize --out <dir>;  python experiments/cv_en_align.py lists --out <dir>
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import yodas_de_align as Y  # noqa: E402  (batches, run_split, write_tar, jline, Engines)
from yodas_de_qwen_check import edit_ops  # noqa: E402

SOURCE = "/soundai/users/tskim/VAPKT-data/data/corpora/covost2-fixie/en_de"
SET_NAME, LANG, CORPUS = "cv4-en", "English", "common-voice-4-en(covost2-train)"
Y.LANG = LANG                                                                     # Engines.transcribe/align 이 모듈 전역 LANG 을 읽는다
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
COLS = ["file", "row_group", "row", "id", "client_id", "sentence"]
RG_PER_SHARD = 16                                                                  # parquet row group 이 100 행이라 16 개(약 1,600 행)씩 묶는다


def _ngrams(words, n=8):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _hash(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


def shard_name(file, block):
    return f"{Path(file).stem}-b{int(block):03d}"


def eval_texts(manifests):
    """평가 manifest 의 영어 문장(원문 raw_reference 가 영어인 행 + 번역 참조가 영어인 행) — 학습에서 뺀다."""
    out = []
    for m in manifests:
        for line in open(m, encoding="utf-8"):
            r = json.loads(line)
            if r.get("lang") == "English":
                out.append(r.get("raw_reference") or r.get("reference") or "")
            if r.get("tgt_lang") == "English" and r.get("translation_reference"):
                out.append(r["translation_reference"])
    return [t for t in out if t]


def cmd_prefilter(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vapasr.data.textnorm import TEXTNORM_VERSION, target_en, target_flags
    files = sorted(Path(a.source).glob("train-*.parquet"))
    rows = {c: [] for c in COLS}
    for p in files:
        f = pq.ParquetFile(p)
        for g in range(f.metadata.num_row_groups):
            t = f.read_row_group(g, columns=["id", "client_id", "sentence"])
            n = t.num_rows
            rows["file"] += [p.name] * n; rows["row_group"] += [g] * n; rows["row"] += list(range(n))
            for c in ("id", "client_id", "sentence"):
                rows[c] += t.column(c).to_pylist()
    n = len(rows["id"])
    ev = eval_texts(a.decon_manifest or [])
    ev_norm = [target_en(t, "librispeech").split() for t in ev]
    ev_grams = set().union(*(_ngrams(w) for w in ev_norm)) if ev_norm else set()
    ev_exact = {" ".join(w) for w in ev_norm if w}
    print(f"rows {n} from {len(files)} files · eval sentences {len(ev)} · 8-grams {len(ev_grams)}", flush=True)
    speakers = sorted(set(rows["client_id"]), key=_hash)
    dev = set(speakers[:a.dev_speakers])
    flags, targets, seen = [], [], set()
    for i in range(n):
        s = rows["sentence"][i] or ""
        tgt = target_en(s, "librispeech"); w = tgt.split(); f = [f"tn_{x}" for x in sorted(target_flags(tgt, LANG))]
        if " ".join(w) in ev_exact: f.append("eval_exact")
        elif ev_grams and _ngrams(w) & ev_grams: f.append("eval_overlap")
        key = (rows["client_id"][i], " ".join(w))
        if w and key in seen: f.append("duplicate_text")
        seen.add(key); flags.append(",".join(f)); targets.append(tgt)
    split = ["dev" if c in dev else "train" for c in rows["client_id"]]
    out = Path(a.out) / "prefilter"; out.mkdir(parents=True, exist_ok=True)
    tab = pa.table({**{c: rows[c] for c in COLS}, "target": targets, "flags": flags, "split": split})
    tmp = out / "prefilter.parquet.tmp"; pq.write_table(tab, tmp, compression="zstd"); tmp.replace(out / "prefilter.parquet")
    fc = Counter(x for f in flags for x in f.split(",") if x)
    cand = Counter(split[i] for i in range(n) if not flags[i])
    summary = dict(rows=n, files=[p.name for p in files], speakers=len(speakers), dev_speakers=len(dev), textnorm=TEXTNORM_VERSION,
                   decon_manifests=a.decon_manifest or [], eval_sentences=len(ev), eval_8grams=len(ev_grams),
                   flag_rows=dict(fc.most_common()), candidates=dict(cand), code_sha256=CODE_SHA256)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


def load_candidates(out):
    import pyarrow.parquet as pq
    t = pq.read_table(Path(out) / "prefilter" / "prefilter.parquet").to_pydict()
    shards = defaultdict(list)
    for i in range(len(t["id"])):
        if t["flags"][i]: continue
        shards[(t["file"][i], t["row_group"][i] // RG_PER_SHARD)].append(
            dict(row_group=t["row_group"][i], row=t["row"][i], utt_id=t["id"][i], client_id=t["client_id"][i], ref=t["sentence"][i],
                 target=t["target"][i], split=t["split"][i]))
    return dict(sorted(shards.items()))


def decode(mp3):
    """mp3 바이트 → (16 kHz float32 모노, 16 kHz PCM16 wav 바이트)."""
    from math import gcd
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = gcd(int(sr), 16000); x = resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    b = io.BytesIO(); sf.write(b, x, 16000, subtype="PCM_16", format="WAV")
    return x, b.getvalue()


def read_shard(source, key, cands):
    import pyarrow.parquet as pq
    file, _ = key
    f = pq.ParquetFile(Path(source) / file)
    by_group = defaultdict(list)
    for c in cands:
        by_group[c["row_group"]].append(c)
    items = []
    for g, cs in sorted(by_group.items()):
        t = f.read_row_group(g, columns=["id", "audio"])
        ids, audio = t.column("id"), t.column("audio")
        for c in cs:
            if ids[c["row"]].as_py() != c["utt_id"]:
                raise ValueError(f"row mismatch {file} rg{g} row{c['row']}")
            items.append(dict(c, mp3=audio[c["row"]].as_py()["bytes"]))
    out = []
    for c in items:
        c = dict(c); mp3 = c.pop("mp3")
        try:
            x, wav = decode(mp3)
            out.append(dict(c, x=x, wav=wav, duration=len(x) / 16000, error=None))
        except Exception as exc:  # noqa: BLE001
            out.append(dict(c, x=None, wav=None, duration=0.0, error=f"{type(exc).__name__}: {str(exc)[:200]}"))
    return out


def wer(ref, hyp):
    from vapasr.data.textnorm import score_en
    r, h = score_en(ref).split(), score_en(hyp).split()
    return edit_ops(r, h) / max(1, len(r)), len(r)


def words_row(it, items, eng, tar_path, duration):
    """Y.words_row 와 같은 검사(project_items · 80 ms 끝 넘침 보정 · build_candidate), 타깃은 사람 문장의 lexical 형."""
    from semcommit_build_speechlm_candidates import build_candidate
    from vapasr.data.selection_alignment import project_items
    from vapasr.data.selection_alignment_qc import classify_failed
    text = it["target"]
    enc = eng.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    qc = None
    try:
        proj = project_items(text, items, enc, duration)
    except ValueError as exc:
        res = classify_failed(dict(alignment_ok=False, error=f"ValueError: {exc}", duration_s=duration, aligner_items=items, target_text=text), enc)
        if res["state"] != "ACCEPT_CLAMPED_80MS":
            return None, f"{res['state']}:{res['reason']}"
        proj, qc = res["corrected"], res["corrected"]["qc"]
    aligned = dict(key=it["utt_id"], source=SET_NAME, lang=LANG, audio=dict(path=f"{tar_path}::{it['utt_id']}.wav"), target_text=text,
                   duration_s=duration, alignment_ok=True, training_eligible=False, dedup_status="unverified", source_shard=it["shard"],
                   timing_quality=proj.get("timing_quality"), **{k: proj[k] for k in ("words", "tokens")})
    record = dict(key=it["utt_id"], selection_state="KEEP_ASR_SILVER", split="train",                       # build_candidate 의 관문 값(YODAS 와 같다); 출처는 아래 selection
                  source_manifest=dict(split="train", num_channels=1, corpus=CORPUS, original_transcript=it["ref"]))
    row, why = build_candidate(aligned, record, tokenizer=eng.pieces, max_duration_s=it["max_dur"])
    if row is None:
        return None, why
    if qc:
        row["alignment_qc"] = qc
    row.update(id=f"cv4en-{it['utt_id']}", set=SET_NAME, candidate_only=False, training_eligible=True,
               selection="human_sentence_qwen_wer_le_%g" % it["max_wer"], wer_vs_qwen=it["wer"], qwen_text=it["hyp"],
               source_corpus=CORPUS, speaker=it["client_id"][:16], split=it["split"])
    return row, None


def process_shard(key, cands, a, eng, final_dir):
    name = shard_name(*key)
    stage = Path(a.out) / "work" / f"{name}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.{int(time.time())}"
    stage.mkdir(parents=True)
    t0 = time.monotonic()
    items = read_shard(a.source, key, cands)
    for it in items:
        it.update(shard=name, max_wer=a.max_wer, max_dur=a.max_dur)
        if it["x"] is not None and not (a.min_dur <= it["duration"] <= a.max_dur):
            it["state"] = "DROP_DURATION"
    ok = [it for it in items if it["x"] is not None and "state" not in it]
    for b in Y.batches(ok, a.batch, a.batch_sec):
        for it, r in zip(b, Y.run_split(eng.transcribe, b)):
            it["hyp"], it["asr_warning"], it["asr_error"] = r.get("text"), r.get("warning"), r.get("error")
    keep = []
    for it in items:
        if it.get("state"): continue
        if it["x"] is None: it["state"] = "HOLD_AUDIO"
        elif it.get("asr_error") or it.get("hyp") is None: it["state"] = "HOLD_ASR_ERROR"
        elif it.get("asr_warning"): it["state"] = "HOLD_TOKEN_CAP"
        else:
            it["wer"], it["ref_words"] = wer(it["target"], it["hyp"]); it["wer"] = round(it["wer"], 4)
            if it["wer"] <= a.max_wer: it["state"] = "KEEP"; keep.append(it)
            else: it["state"] = "DROP_DISAGREE"
    tar_rel = final_dir / "audio.tar"
    rows, failed, members = {"train": [], "dev": []}, [], []
    for b in Y.batches(keep, a.batch, a.batch_sec):
        for it in b: it["hyp_qwen"] = it["hyp"]; it["hyp"] = it["target"]   # Y.Engines.align 은 it["hyp"] 를 정렬한다 → 사람 문장(lexical)
        for it, r in zip(b, Y.run_split(eng.align, b)):
            it["hyp"] = it.pop("hyp_qwen")
            if r.get("error"):
                failed.append(dict(utt_id=it["utt_id"], reason="aligner_error", error=r["error"])); it["state"] = "FAIL_ALIGN"; continue
            try:
                row, why = words_row(it, r["items"], eng, tar_rel, it["duration"])
            except Exception as exc:  # noqa: BLE001
                row, why = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            if row is None:
                failed.append(dict(utt_id=it["utt_id"], reason=why, duration_s=it["duration"], target_text=it["target"], aligner_items=r["items"]))
                it["state"] = "FAIL_ALIGN"; continue
            rows[it["split"]].append(row); members.append((f"{it['utt_id']}.wav", it["wav"])); it["state"] = "ALIGNED"
    states = Counter(it["state"] for it in items)
    if members:
        Y.write_tar(stage / "audio.tar", members)
    with open(stage / "records.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            Y.jline(f, {k: it.get(k) for k in ("utt_id", "client_id", "split", "duration", "ref", "target", "hyp", "wer", "ref_words",
                                                "state", "asr_warning", "asr_error", "error")})
    for split in ("train", "dev"):
        with open(stage / f"words-{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows[split]: Y.jline(f, row)
    with open(stage / "failed.jsonl", "w", encoding="utf-8") as f:
        for x in failed: Y.jline(f, x)
    hours = defaultdict(float)
    for it in items: hours[it["state"]] += it["duration"] / 3600
    summary = dict(complete=True, shard=name, source=list(key), candidates=len(items), states=dict(states),
                   hours={k: round(v, 4) for k, v in hours.items()}, words={k: len(v) for k, v in rows.items()},
                   failed_reasons=dict(Counter(x["reason"] for x in failed)), max_wer=a.max_wer, min_dur=a.min_dur, max_dur=a.max_dur,
                   asr_model=a.asr_model, aligner_model=a.aligner_model, elapsed_s=round(time.monotonic() - t0, 1), code_sha256=CODE_SHA256)
    (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    final = Path(a.out) / "results" / name
    if final.exists():
        raise FileExistsError(f"{final} already exists; refusing to replace a finished shard")
    stage.rename(final)
    return summary


def cmd_worker(a):
    rank = a.rank if a.rank is not None else int(os.environ.get("SLURM_PROCID", 0))
    world = a.world if a.world is not None else int(os.environ.get("SLURM_NTASKS", 1))
    shards = load_candidates(a.out)
    keys = list(shards)[rank::world]
    todo = [k for k in keys if not (Path(a.out) / "results" / shard_name(*k) / "summary.json").exists()][:a.limit_shards or None]
    print(f"rank {rank}/{world}: {len(keys)} shards, {len(todo)} to do", flush=True)
    if not todo: return
    eng = Y.Engines(a)
    (Path(a.out) / "results").mkdir(parents=True, exist_ok=True)
    for k in todo:
        name = shard_name(*k)
        try:
            s = process_shard(k, shards[k], a, eng, Path(a.out) / "results" / name)
        except Exception as exc:  # noqa: BLE001
            err = Path(a.out) / "errors"; err.mkdir(exist_ok=True)
            (err / f"{name}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.json").write_text(
                json.dumps(dict(shard=name, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()), ensure_ascii=False, indent=1))
            print(f"SHARD_ERROR rank={rank} {name} {type(exc).__name__}: {str(exc)[:200]}", flush=True); continue
        print(f"SHARD_DONE rank={rank} {s['shard']} cand={s['candidates']} states={json.dumps(s['states'])} {s['elapsed_s']} s", flush=True)


def cmd_summarize(a):
    shards = load_candidates(a.out)
    states, hours, words, reasons, done = Counter(), Counter(), Counter(), Counter(), 0
    for k in shards:
        p = Path(a.out) / "results" / shard_name(*k) / "summary.json"
        if not p.exists(): continue
        s = json.loads(p.read_text()); done += 1
        states.update(s["states"]); hours.update(s["hours"]); words.update(s["words"]); reasons.update(s["failed_reasons"])
    out = dict(shards=len(shards), done=done, remaining=len(shards) - done, states=dict(states), hours={k: round(v, 2) for k, v in hours.items()},
               words=dict(words), failed_reasons=dict(reasons.most_common(20)))
    (Path(a.out) / "align-summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_lists(a):
    out = Path(a.out) / "lists"; out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        paths = sorted(str(p) for p in (Path(a.out) / "results").glob(f"*/words-{split}.jsonl") if p.stat().st_size > 0)
        (out / f"cv4-en-words-{split}.list").write_text("".join(p + "\n" for p in paths))
        print(split, len(paths), "files")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prefilter")
    q.add_argument("--source", default=SOURCE)
    q.add_argument("--out", required=True)
    q.add_argument("--decon-manifest", nargs="*", help="평가 manifest(jsonl) — 영어 문장을 학습에서 뺀다")
    q.add_argument("--dev-speakers", type=int, default=200)
    q.set_defaults(fn=cmd_prefilter)
    q = sub.add_parser("worker")
    q.add_argument("--out", required=True)
    q.add_argument("--source", default=SOURCE)
    q.add_argument("--rank", type=int)
    q.add_argument("--world", type=int)
    q.add_argument("--max-wer", type=float, default=0.10)
    q.add_argument("--min-dur", type=float, default=1.0)
    q.add_argument("--max-dur", type=float, default=30.0)
    q.add_argument("--limit-shards", type=int, default=0, help="시험용: 이 수의 shard 만")
    q.add_argument("--asr-model", default="/soundai/Model/Qwen3-ASR-1.7B")
    q.add_argument("--aligner-model", default="/soundai/Model/Qwen3-ForcedAligner-0.6B")
    q.add_argument("--batch", type=int, default=64)
    q.add_argument("--batch-sec", type=float, default=960.0)
    q.set_defaults(fn=cmd_worker)
    q = sub.add_parser("summarize")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_summarize)
    q = sub.add_parser("lists")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_lists)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
