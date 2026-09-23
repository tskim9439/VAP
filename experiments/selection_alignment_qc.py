#!/usr/bin/env python3
"""Full CPU QC for qwen-verbatim-align-v1; input artifacts remain immutable."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest
from vapasr.data.selection_alignment_qc import POLICY, classify_failed, sample_rank, validate_original


def save(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def process(job):
    source, out, spec, fingerprint, tokenizer_path, samples_per_cell = job
    name = spec["name"]
    src, dest = source / "results" / name, out / "results" / name
    if dest.exists():
        result = json.loads((dest / "summary.json").read_text())
        if not result["complete"] or result["fingerprint"] != fingerprint:
            raise ValueError("invalid_resume:" + name)
        for filename, sha in result["outputs"].items():
            if file_digest(dest / filename) != sha:
                raise ValueError("changed_resume_output:" + name)
        return result
    upstream = json.loads((src / "summary.json").read_text())
    if not upstream["complete"] or upstream["fingerprint"] != spec["fingerprint"]:
        raise ValueError("upstream_summary_mismatch:" + name)
    for filename, sha in upstream["outputs"].items():
        if file_digest(src / filename) != sha:
            raise ValueError("upstream_output_changed:" + name)
    local_root = Path(os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "/tmp")))
    local_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="vapasr-align-qc-" + name + "-", dir=local_root))
    counts, sources, langs, candidates = Counter(), defaultdict(Counter), Counter(), {}
    tokenizer = None
    def decide(record, failed):
        nonlocal tokenizer
        if failed:
            # Tokenization is needed only for the <=80 ms repair path.
            preliminary = classify_failed(record)
            if preliminary["state"] == "NEED_ENCODING":
                preliminary = None
            if preliminary is None:
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                        use_fast=True, fix_mistral_regex=True)
                    if not tokenizer.is_fast:
                        raise ValueError("fast_tokenizer_required")
                encoding = tokenizer(record["target_text"], add_special_tokens=False,
                                     return_offsets_mapping=True)
                return classify_failed(record, encoding)
            return preliminary
        return validate_original(record)
    paths = {n: (stage / n).open("w", encoding="utf-8") for n in
             ("decisions.jsonl", "corrected.jsonl", "review.jsonl", "rejected.jsonl")}
    try:
        for filename, failed in (("aligned.jsonl", False), ("failed.jsonl", True)):
            for line in (src / filename).open(encoding="utf-8"):
                record = json.loads(line)
                try:
                    decision = decide(record, failed)
                except Exception as exc:
                    decision = dict(state="REJECT_INVALID", reason=f"qc_exception:{type(exc).__name__}:{str(exc)[:240]}",
                                    max_overshoot_s=None, corrected=None)
                state = decision["state"]
                counts[state] += 1; sources[record["source"]][state] += 1; langs[record["lang"]] += 1
                compact = dict(key=record["key"], source=record["source"], lang=record["lang"],
                               source_shard=name, state=state, reason=decision["reason"],
                               max_overshoot_s=decision["max_overshoot_s"], training_eligible=False)
                paths["decisions.jsonl"].write(json.dumps(compact, ensure_ascii=False) + "\n")
                if state == "ACCEPT_CLAMPED_80MS":
                    paths["corrected.jsonl"].write(json.dumps(decision["corrected"], ensure_ascii=False) + "\n")
                elif state == "REVIEW_80_320MS":
                    paths["review.jsonl"].write(json.dumps(record, ensure_ascii=False) + "\n")
                elif state.startswith("REJECT"):
                    paths["rejected.jsonl"].write(json.dumps(record, ensure_ascii=False) + "\n")
                cell = (record["source"], state)
                rank = sample_rank(record["key"], state)
                current = candidates.get(cell)
                sample = dict(record, qc_decision=compact, sample_rank=rank)
                if current is None:
                    candidates[cell] = [sample]
                else:
                    current.append(sample); current.sort(key=lambda x: x["sample_rank"])
                    del current[samples_per_cell:]
    finally:
        for handle in paths.values(): handle.close()
    with (stage / "sample-candidates.jsonl").open("w", encoding="utf-8") as handle:
        for cell in sorted(candidates):
            for row in candidates[cell]: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = sum(counts.values())
    if rows != upstream["rows"] or counts["ACCEPT_ORIGINAL"] > upstream["ok"]:
        raise ValueError("row_accounting_failed:" + name)
    result = dict(complete=True, name=name, fingerprint=fingerprint, rows=rows,
                  counts=counts, sources=sources, langs=langs, upstream_outputs=upstream["outputs"],
                  outputs={p.name: file_digest(p) for p in stage.glob("*.jsonl")})
    save(stage / "summary.json", result)
    publish = out / "work" / f"{name}-publish-{os.getpid()}"
    shutil.copytree(stage, publish)
    publish.rename(dest)
    shutil.rmtree(stage)
    return result


def export_sample(row, out):
    import soundfile as sf
    from vapasr.data.selection_audio import decode
    x, info = decode(row)
    if info["sha256"] != row["waveform_sha256"]:
        raise ValueError("review_sample_waveform_changed")
    rel = f"audio/{row['key']}.wav"; sf.write(out / rel, x, 16000, subtype="PCM_16")
    return rel


def bundle(rows, out):
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        audio = export_sample(row, out); q = row["qc_decision"]
        items = row.get("aligner_items", [])
        tail = items[-8:]
        table = "".join(f"<tr><td>{html.escape(str(x['text']))}</td><td>{x['start_time']:.3f}</td><td>{x['end_time']:.3f}</td></tr>" for x in tail)
        cards.append(f'''<article data-key="{row['key']}"><h2>{html.escape(row['source'])} · {q['state']}</h2>
<p>길이 {row['duration_s']:.3f}s · 초과 {q.get('max_overshoot_s') or 0:.3f}s</p>
<audio controls preload="none" src="{audio}"></audio><p><b>Qwen 원출력:</b> {html.escape(row['target_text'])}</p>
<table><tr><th>마지막 단어</th><th>시작</th><th>끝</th></tr>{table}</table>
<label>판정 <select><option>unreviewed</option><option>pass</option><option>fail</option><option>uncertain</option></select></label>
<textarea placeholder="끝 단어 잘림·무음·정렬 오류 메모"></textarea></article>''')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Forced alignment QC</title>
<style>body{max-width:1000px;margin:24px auto;font-family:system-ui}article{border:1px solid #aaa;padding:16px;margin:18px 0;border-radius:8px}audio,textarea{width:100%}textarea{height:55px}td,th{padding:3px 12px;text-align:left}</style>
<h1>Qwen 원출력 forced alignment QC 표본</h1><p>끝 단어가 실제 음성에 맞는지, 잘렸는지 확인합니다. 내려받기 전에는 입력이 보존되지 않습니다.</p>
<button onclick="download()">검수 JSON 내려받기</button>''' + "\n".join(cards) + '''<script>
function download(){let rows=[...document.querySelectorAll('article')].map(x=>({key:x.dataset.key,label:x.querySelector('select').value,note:x.querySelector('textarea').value}));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({training_eligible:false,rows},null,2)],{type:'application/json'}));a.download='alignment-qc-review.json';a.click()}</script></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path); ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--samples-per-cell", type=int, default=3)
    ap.add_argument("--max-shards", type=int, default=0, help="0=all; positive values are smoke tests")
    a = ap.parse_args()
    if not 1 <= a.workers <= 32 or not 1 <= a.samples_per_cell <= 20 or a.max_shards < 0: raise ValueError("invalid limits")
    source, out = a.source.resolve(), a.out.resolve()
    if source == out or source in out.parents or out in source.parents: raise ValueError("independent output required")
    summary=json.loads((source/"summary.json").read_text()); config=json.loads((source/"config.json").read_text())
    if not summary["complete"] or summary["fingerprint"] != config["fingerprint"]: raise ValueError("incomplete upstream")
    root=Path(__file__).resolve().parents[1]
    run_config=dict(policy=POLICY,source=str(source),source_summary_sha256=file_digest(source/"summary.json"),
        upstream_fingerprint=config["fingerprint"],tokenizer=config["tokenizer"],tokenizer_files=config["tokenizer_files"],
        code={str(p.relative_to(root)):file_digest(p) for p in (Path(__file__).resolve(),root/"vapasr/data/selection_alignment_qc.py",root/"vapasr/data/selection_alignment.py")},
        samples_per_cell=a.samples_per_cell,max_shards=a.max_shards,
        scope="smoke" if a.max_shards else "full")
    run_config["fingerprint"]=digest(run_config);out.mkdir(parents=True,exist_ok=True)
    lock=(out/"run.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/"config.json").exists():
        if json.loads((out/"config.json").read_text()) != run_config: raise ValueError("QC resume fingerprint changed")
    elif any(p.name!="run.lock" for p in out.iterdir()): raise ValueError("unfingerprinted output")
    save(out/"config.json",run_config)
    for d in ("results","work"): (out/d).mkdir(exist_ok=True)
    specs=[]
    for p in sorted((source/"results").glob("*/summary.json")):
        r=json.loads(p.read_text());specs.append(dict(name=p.parent.name,fingerprint=r["fingerprint"]))
    if a.max_shards: specs=specs[:a.max_shards]
    jobs=[(source,out,s,run_config["fingerprint"],config["tokenizer"],a.samples_per_cell) for s in specs]
    results=[]; started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(process,j) for j in jobs]):
            r=future.result();results.append(r)
            if len(results)%10==0: print(f"PROGRESS {len(results)}/{len(jobs)}",flush=True)
    counts=Counter();sources=defaultdict(Counter);sample_rows=[]
    for r in results:
        counts.update(r["counts"])
        for source_name,c in r["sources"].items():sources[source_name].update(c)
        sample_rows.extend(map(json.loads,(out/"results"/r["name"]/"sample-candidates.jsonl").open()))
    selected={}
    for row in sample_rows:
        cell=(row["source"],row["qc_decision"]["state"]);selected.setdefault(cell,[]).append(row)
    review=[]
    for rows in selected.values():review.extend(sorted(rows,key=lambda x:x["sample_rank"])[:a.samples_per_cell])
    review.sort(key=lambda x:(x["source"],x["qc_decision"]["state"],x["sample_rank"]))
    review_root=out/"human-review";review_root.mkdir(exist_ok=True);bundle(review,review_root)
    accepted_original=[str(source/"results"/r["name"]/"aligned.jsonl") for r in sorted(results,key=lambda x:x["name"])]
    corrected=[str(out/"results"/r["name"]/"corrected.jsonl") for r in sorted(results,key=lambda x:x["name"])]
    with (out/"accepted-inputs.jsonl").open("w") as f:
        for path in accepted_original:f.write(json.dumps(dict(kind="original",path=path))+"\n")
        for path in corrected:f.write(json.dumps(dict(kind="boundary_clamped",path=path))+"\n")
    final=dict(complete=True,policy=POLICY,fingerprint=run_config["fingerprint"],rows=sum(counts.values()),counts=counts,
        sources=sources,automatic_accept=counts["ACCEPT_ORIGINAL"]+counts["ACCEPT_CLAMPED_80MS"],
        deferred_review_rows=counts["REVIEW_80_320MS"],
        rejected_rows=sum(v for k,v in counts.items() if k.startswith("REJECT")),
        acoustic_spot_check_required=True,acoustic_spot_check_sample_n=len(review),
        review_bundle=str(review_root/"index.html"),training_eligible=False,
        accepted_inputs_sha256=file_digest(out/"accepted-inputs.jsonl"),elapsed_s=time.time()-started)
    expected_rows=sum(json.loads((source/"results"/s["name"]/"summary.json").read_text())["rows"] for s in specs)
    if final["rows"] != expected_rows:raise ValueError("global row accounting failed")
    save(out/"summary.json",final);print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
