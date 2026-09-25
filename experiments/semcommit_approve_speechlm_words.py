#!/usr/bin/env python3
"""Candidate speechlm semcommit words → approved words shards + index for slurm/semcommit-main-label-apex-n2.sbatch.

Keeps only rows whose QC decision is `state == QC_PASS_CANDIDATE` (the human-approved source filter), stamps the gold-gate
training-eligibility fingerprint, and splits by duration into the worker's pools (short < 8 s, long ≥ 8 s). Inputs are
verified: the training-eligibility summary must be gate-passed, the QC decision file must hash to the digest it names, and
each candidate words file must hash to its builder stats. Never overwrites: --out-dir must not exist.

  python experiments/semcommit_approve_speechlm_words.py \
      --training-eligibility <gate>/decision/training-eligibility-summary.json \
      --candidates <cand>/part-000123.words.jsonl ... --out-dir <approved> [--shard-rows 2000]
Outputs: <out-dir>/{short,long}/<part>-NNNN.jsonl, <out-dir>/index.tsv (pool<TAB>path), <out-dir>/summary.json.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

PASS = "QC_PASS_CANDIDATE"
SHORT_S = 8.0


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eligibility(path):
    te = json.loads(Path(path).read_text())
    need = dict(complete=True, training_eligible=True, labeling_eligible=True, approval_status="gold_gate_passed",
                approved_filter=f"state == {PASS}")
    bad = {k: te.get(k) for k, v in need.items() if te.get(k) != v}
    if bad or not te.get("approval_fingerprint") or not te.get("decision_file"):
        raise ValueError(f"training eligibility summary is not a gate-passed approval: {bad}")
    return te


def candidate_rows(paths):
    """Yield (path, row) from verified candidate words files (hash = builder stats words_sha256)."""
    for p in map(Path, paths):
        stats = p.with_name(p.name.replace(".words.jsonl", ".stats.json"))
        st = json.loads(stats.read_text())
        if st.get("candidate_only") is not True or st.get("words_sha256") != sha256(p):
            raise ValueError(f"candidate words do not match their builder stats: {p}")
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield p, json.loads(line)


def qc_states(decision_file, expected_sha, keys):
    """One pass over the QC decision file: verify its hash and return {key: state} for the wanted keys."""
    h, states = hashlib.sha256(), {}
    with Path(decision_file).open("rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            k = line[9:73].decode() if line.startswith(b'{"key": "') else json.loads(line)["key"]
            if k in keys:
                states[k] = json.loads(line)["state"]
    if h.hexdigest() != expected_sha:
        raise ValueError(f"QC decision file digest mismatch: {decision_file}")
    return states


def approve(te_path, candidates, out_dir, shard_rows=2000):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    te = load_eligibility(te_path)
    rows = list(candidate_rows(candidates))
    keys = {r["source_key"] for _, r in rows}
    states = qc_states(te["decision_file"], te["decision_file_sha256"], keys)
    counts, by_pool, punct = Counter(), defaultdict(list), Counter()
    for path, r in rows:
        if r.get("set") != "speechlm-partial" or r.get("candidate_only") is not True or r.get("training_eligible") is not False:
            raise ValueError(f"not a candidate row: {r.get('id')}")
        st = states.get(r["source_key"])
        if st != PASS:
            counts[f"drop_{st or 'no_decision'}"] += 1
            continue
        r = dict(r, candidate_only=False, training_eligible=True, approval_fingerprint=te["approval_fingerprint"],
                 approval_status=te["approval_status"])
        pool = "short" if r["duration_s"] < SHORT_S else "long"
        by_pool[(pool, path.name.replace(".words.jsonl", ""))].append(r)
        counts[f"kept_{pool}"] += 1
        ws = r["words"]
        punct["words"] += len(ws); punct["punct_final"] += sum("punct_final" in w["tags"] for w in ws)
        punct["streams"] += 1; punct["stream_end_punct"] += bool(ws) and "punct_final" in ws[-1]["tags"]
    index = []
    for (pool, part), rs in sorted(by_pool.items()):
        (out_dir / pool).mkdir(parents=True, exist_ok=True)
        for k in range(0, len(rs), shard_rows):
            dest = out_dir / pool / f"{part}-{k // shard_rows:04d}.jsonl"
            with dest.open("x", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rs[k:k + shard_rows])
            index.append(f"{pool}\t{dest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.tsv").write_text("\n".join(index) + "\n")
    summary = dict(schema="speechlm-semcommit-approved-words-v1", complete=True, training_eligibility=str(te_path),
                   training_eligibility_sha256=sha256(te_path), approval_fingerprint=te["approval_fingerprint"],
                   candidates={str(p): sha256(p) for p in map(Path, candidates)}, counts=dict(counts), shards=len(index),
                   punct=dict(punct, word_rate=round(punct["punct_final"] / max(1, punct["words"]), 4),
                              stream_end_rate=round(punct["stream_end_punct"] / max(1, punct["streams"]), 4)),
                   meaning="approved labeling input (teacher labels not yet produced)")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(dict(counts=summary["counts"], shards=len(index), punct=summary["punct"]), ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--training-eligibility", type=Path, required=True)
    p.add_argument("--candidates", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shard-rows", type=int, default=2000)
    a = p.parse_args()
    approve(a.training_eligibility, a.candidates, a.out_dir, a.shard_rows)


if __name__ == "__main__":
    main()
