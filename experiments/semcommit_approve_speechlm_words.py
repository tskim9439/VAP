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
Part mode (--qc-split <split dir> --part part-NNNNNN --out-words W): one words file with every approved row of that part,
keys from semcommit_split_qc_pass.py; writes W and W.ok (digest, counts). Used by slurm/semcommit-part-label-worker.sh.
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


def stamp(r, te):
    if r.get("set") != "speechlm-partial" or r.get("candidate_only") is not True or r.get("training_eligible") is not False:
        raise ValueError(f"not a candidate row: {r.get('id')}")
    return dict(r, candidate_only=False, training_eligible=True, approval_fingerprint=te["approval_fingerprint"],
                approval_status=te["approval_status"])


def approve_part(te_path, qc_split, part, candidates, out_words):
    """Part-unit mode: approved rows of one alignment part → one words file (+ <out>.ok with digest and counts).
    Keys come from semcommit_split_qc_pass.py (its summary must name the same decision file digest and fingerprint)."""
    out_words = Path(out_words)
    te = load_eligibility(te_path)
    split = json.loads((Path(qc_split) / "summary.json").read_text())
    if (split.get("decision_file_sha256") != te["decision_file_sha256"] or split.get("approval_fingerprint") != te["approval_fingerprint"]):
        raise ValueError("QC split does not belong to this training eligibility")
    kf = Path(qc_split) / "pass-keys" / f"{part}.keys"
    if kf.exists():
        if sha256(kf) != split["key_files_sha256"].get(part):
            raise ValueError(f"pass-keys digest mismatch: {kf}")
        keys = set(kf.read_text().split())
    else:
        keys = set()                                      # a part with no QC-passed row
    counts, kept = Counter(), []
    for _, r in candidate_rows([candidates]):
        if r["source_key"] not in keys:
            counts["drop_not_qc_pass"] += 1
            continue
        kept.append(stamp(r, te)); counts["kept_short" if r["duration_s"] < SHORT_S else "kept_long"] += 1
    with out_words.open("x", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in kept)
    ok = dict(part=part, words_sha256=sha256(out_words), counts=dict(counts), approval_fingerprint=te["approval_fingerprint"],
              candidates_sha256=sha256(candidates))
    Path(str(out_words) + ".ok").open("x").write(json.dumps(ok))
    print(json.dumps(ok))
    return ok


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
        st = states.get(r["source_key"])
        if st != PASS:
            stamp(r, te); counts[f"drop_{st or 'no_decision'}"] += 1
            continue
        r = stamp(r, te)
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
    p.add_argument("--out-dir", type=Path, help="pooled shards mode (decision-file scan)")
    p.add_argument("--shard-rows", type=int, default=2000)
    p.add_argument("--qc-split", type=Path, help="part mode: semcommit_split_qc_pass.py output dir")
    p.add_argument("--part", help="part mode: alignment part name (part-NNNNNN)")
    p.add_argument("--out-words", type=Path, help="part mode: single approved words file (+ .ok)")
    a = p.parse_args()
    if a.qc_split:
        if not (a.part and a.out_words and len(a.candidates) == 1):
            p.error("part mode needs --part, --out-words and exactly one --candidates file")
        approve_part(a.training_eligibility, a.qc_split, a.part, a.candidates[0], a.out_words)
    elif a.out_dir:
        approve(a.training_eligibility, a.candidates, a.out_dir, a.shard_rows)
    else:
        p.error("give --out-dir (pooled) or --qc-split/--part/--out-words (part mode)")


if __name__ == "__main__":
    main()
