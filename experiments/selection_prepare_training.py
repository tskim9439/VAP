#!/usr/bin/env python3
"""Build a fail-closed, Qwen-verbatim Phase-1 stream view from an approval."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.approved_training import alignment_density_problem, join_supervision, pack_streams, read_jsonl, SCHEMA, TARGET_POLICY
from vapasr.data.selection import digest, file_digest


def save(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def shard_job(job):
    name, selection, alignment, qc, out, approval_fp, run_fp = job
    dest = out / "parts" / name
    if (dest / "summary.json").exists():
        summary = json.loads((dest / "summary.json").read_text())
        if summary["fingerprint"] != run_fp:
            raise ValueError("resume fingerprint mismatch: " + name)
        return summary
    selected = {r["key"]: r for r in read_jsonl(selection / "results" / name / "keep-asr.jsonl")}
    aligned = list(read_jsonl(alignment / "results" / name / "aligned.jsonl"))
    aligned += list(read_jsonl(qc / "results" / name / "corrected.jsonl"))
    approved = [join_supervision(selected[a["key"]], a, approval_fp) for a in aligned]
    if len(approved) != len({r["key"] for r in approved}):
        raise ValueError("duplicate approved key: " + name)
    rows, excluded = [], []
    for row in approved:
        problem = alignment_density_problem(row)
        if problem:
            row.update(asr_training_eligible=False, training_exclusion=problem); excluded.append(row)
        else: rows.append(row)
    streams = pack_streams(rows, approval_fp)
    stage = Path(tempfile.mkdtemp(prefix="vapasr-training-view-" + name + "-"))
    with (stage / "supervision.jsonl").open("w") as sf, (stage / "excluded.jsonl").open("w") as ef, (stage / "streams.jsonl").open("w") as mf, (stage / "align.jsonl").open("w") as af:
        for row in rows: sf.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in excluded: ef.write(json.dumps(row, ensure_ascii=False) + "\n")
        for manifest, align in streams:
            mf.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            af.write(json.dumps(align, ensure_ascii=False) + "\n")
    counts = Counter(r["lang"] for r in rows)
    stream_counts = Counter(manifest["lang"] for manifest, _ in streams)
    result = dict(complete=True, name=name, fingerprint=run_fp, approved_alignment_rows=len(approved),
                  trainable_rows=len(rows), excluded_alignment_density=len(excluded),
                  streams=len(streams), hours=sum(r["duration_s"] for r in rows) / 3600,
                  languages=counts, stream_languages=stream_counts,
                  outputs={p.name: file_digest(p) for p in stage.glob("*.jsonl")})
    save(stage / "summary.json", result)
    dest.parent.mkdir(parents=True, exist_ok=True)
    publish = dest.parent / f"{name}-publish-{os.getpid()}"
    shutil.copytree(stage, publish)
    publish.rename(dest)
    shutil.rmtree(stage)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--approval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-shards", type=int, default=0)
    a = ap.parse_args()
    approval = json.loads((a.approval / "approval.json").read_text())
    approval_summary = json.loads((a.approval / "summary.json").read_text())
    if not approval_summary.get("training_eligible"):
        raise ValueError("approval is not training eligible")
    qc = Path(approval["qc_root"]); qc_config = json.loads((qc / "config.json").read_text())
    alignment = Path(qc_config["source"]); align_config = json.loads((alignment / "config.json").read_text())
    selection = Path(align_config["source"])
    names = sorted(p.parent.name for p in (alignment / "results").glob("*/summary.json"))
    if a.max_shards: names = names[:a.max_shards]
    config = dict(schema=SCHEMA, target_policy=TARGET_POLICY,
                  approval=str(a.approval.resolve()), approval_fingerprint=approval["fingerprint"],
                  approval_summary_sha256=file_digest(a.approval / "summary.json"),
                  selection=str(selection), alignment=str(alignment), qc=str(qc),
                  tokenizer=align_config["tokenizer"], tokenizer_files=align_config["tokenizer_files"],
                  pack=dict(scope="within_source_shard_and_affinity", target_s=22.0, max_s=25.0,
                            silence_s=[.3, .8]), max_shards=a.max_shards,
                  supervision=dict(asr=True, speaker=False, turn=False),
                  code={"script": file_digest(Path(__file__)),
                        "module": file_digest(Path(__file__).resolve().parents[1] / "vapasr/data/approved_training.py")})
    config["fingerprint"] = digest(config)
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "config.json").exists() and json.loads((a.out / "config.json").read_text()) != config:
        raise ValueError("output config changed")
    save(a.out / "config.json", config)
    jobs = [(n, selection, alignment, qc, a.out, approval["fingerprint"], config["fingerprint"]) for n in names]
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(shard_job, j) for j in jobs]):
            results.append(future.result())
            if len(results) % 20 == 0: print(f"PROGRESS {len(results)}/{len(jobs)}", flush=True)
    results.sort(key=lambda r: r["name"])
    dataset_names = {"English": "approved-en", "Korean": "approved-ko"}
    manifest_dirs = {lang: a.out / "manifests" / name for lang, name in dataset_names.items()}
    align_dirs = {lang: a.out / "align" / name for lang, name in dataset_names.items()}
    for path in (*manifest_dirs.values(), *align_dirs.values()): path.mkdir(parents=True, exist_ok=True)
    supervision = a.out / "supervision-inputs.jsonl"
    manifest_handles = {lang: (path / "streams.jsonl").open("w") for lang, path in manifest_dirs.items()}
    try:
        with supervision.open("w") as sf:
            for r in results:
                part = a.out / "parts" / r["name"]
                ids = {}
                for row in read_jsonl(part / "streams.jsonl"):
                    ids[row["id"]] = row["lang"]
                    manifest_handles[row["lang"]].write(json.dumps(row, ensure_ascii=False) + "\n")
                part_handles = {}
                try:
                    for lang, path in align_dirs.items():
                        (path / "parts").mkdir(exist_ok=True)
                        part_handles[lang] = (path / "parts" / f"{r['name']}.jsonl").open("w")
                    for row in read_jsonl(part / "align.jsonl"):
                        part_handles[ids[row["id"]]].write(json.dumps(row, ensure_ascii=False) + "\n")
                finally:
                    for handle in part_handles.values(): handle.close()
                sf.write(json.dumps({"path": str(part / "supervision.jsonl"), "sha256": r["outputs"]["supervision.jsonl"]}) + "\n")
    finally:
        for handle in manifest_handles.values(): handle.close()
    total = dict(complete=True, training_eligible=True, schema=SCHEMA,
                 fingerprint=config["fingerprint"], approval_fingerprint=approval["fingerprint"],
                 approved_alignment_rows=sum(r["approved_alignment_rows"] for r in results),
                 trainable_rows=sum(r["trainable_rows"] for r in results),
                 excluded_alignment_density=sum(r["excluded_alignment_density"] for r in results),
                 streams=sum(r["streams"] for r in results),
                 hours=sum(r["hours"] for r in results), shards=len(results),
                 languages=dict(sum((Counter(r["languages"]) for r in results), Counter())),
                 datasets={name: dict(lang=lang,
                                      streams=sum(r["stream_languages"].get(lang, 0) for r in results),
                                      manifest_sha256=file_digest(manifest_dirs[lang] / "streams.jsonl"))
                           for lang, name in dataset_names.items()},
                 outputs={"supervision-inputs.jsonl": file_digest(supervision)})
    if not a.max_shards and total["approved_alignment_rows"] != approval_summary["rows"]:
        raise ValueError(f"approval accounting mismatch {total['approved_alignment_rows']} != {approval_summary['rows']}")
    target_policy = dict(schema="qwen-verbatim-training-target-v1", target_policy=TARGET_POLICY,
                         normalization="none", target_origin="qwen3_asr_pseudo_label",
                         approval_fingerprint=approval["fingerprint"], tokenizer_files=align_config["tokenizer_files"],
                         speaker_supervision=False, turn_supervision=False)
    target_policy["fingerprint"] = digest(target_policy)
    for lang, name in dataset_names.items():
        save(align_dirs[lang] / "target-policy.json", target_policy)
        ds_card = dict(schema_version="vapasr-ds-1.1", name=name, corpus="approved-multicorpus",
                       splits={"train": total["datasets"][name]["streams"]}, rows=total["datasets"][name]["streams"],
                       modes={"stream": total["datasets"][name]["streams"]}, langs={lang: total["datasets"][name]["streams"]},
                       target_policy=TARGET_POLICY, approval_fingerprint=approval["fingerprint"])
        al_card = dict(schema_version="vapasr-align-1.1", manifest=name, align_root="align",
                       target_policy=TARGET_POLICY, approval_fingerprint=approval["fingerprint"],
                       tokenizer=dict(json_sha256=None), records=dict(streams=total["datasets"][name]["streams"]))
        save(manifest_dirs[lang] / "dataset.json", ds_card)
        save(align_dirs[lang] / "align.json", al_card)
    save(a.out / "summary.json", total)
    print(json.dumps(total, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
