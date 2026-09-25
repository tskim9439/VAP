#!/usr/bin/env python3
"""One pass over the speechlm QC decision file → per-part approved keys + a DB-interleaved processing order.

Part-unit labeling (slurm/semcommit-part-label-apex.sbatch) handles one alignment part at a time, so each part needs its own
approved-key list instead of rescanning the 6.9 GB decision file. The decision file must hash to the digest the gate-passed
training-eligibility summary names. Never overwrites: --out-dir must not exist.

  python experiments/semcommit_split_qc_pass.py --training-eligibility <gate>/decision/training-eligibility-summary.json \
      --out-dir <work>/qc-pass-split
Outputs: pass-keys/<part>.keys (one key per line, state == QC_PASS_CANDIDATE), parts-order.tsv (part<TAB>source<TAB>n_pass,
round-robin over sources so any prefix covers every DB), summary.json (digests, per-part key-file sha256, counts).
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

PASS = "QC_PASS_CANDIDATE"


def interleave(by_source):
    """Round-robin over sources (each source's parts in order) → [(part, source)]."""
    queues = {s: sorted(ps) for s, ps in sorted(by_source.items())}
    out, k = [], 0
    while any(k < len(q) for q in queues.values()):
        out += [(q[k], s) for s, q in queues.items() if k < len(q)]
        k += 1
    return out


def split(te_path, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    te = json.loads(Path(te_path).read_text())
    if te.get("training_eligible") is not True or te.get("approval_status") != "gold_gate_passed":
        raise ValueError("training eligibility summary is not gate-passed")
    keys, source = defaultdict(list), {}
    h = hashlib.sha256()
    with Path(te["decision_file"]).open("rb") as f:
        for line in f:
            h.update(line)
            if not line.strip():
                continue
            r = json.loads(line)
            if r["state"] == PASS:
                keys[r["source_shard"]].append(r["key"]); source[r["source_shard"]] = r["source"]
    if h.hexdigest() != te["decision_file_sha256"]:
        raise ValueError("QC decision file digest mismatch")
    (out_dir / "pass-keys").mkdir(parents=True)
    key_sha = {}
    for part, ks in sorted(keys.items()):
        data = ("\n".join(ks) + "\n").encode()
        (out_dir / "pass-keys" / f"{part}.keys").open("xb").write(data)
        key_sha[part] = hashlib.sha256(data).hexdigest()
    by_source = defaultdict(list)
    for part, s in source.items():
        by_source[s].append(part)
    order = interleave(by_source)
    (out_dir / "parts-order.tsv").open("x").write("".join(f"{p}\t{s}\t{len(keys[p])}\n" for p, s in order))
    summary = dict(schema="speechlm-semcommit-qc-pass-split-v1", complete=True, training_eligibility=str(te_path),
                   decision_file=te["decision_file"], decision_file_sha256=te["decision_file_sha256"],
                   approval_fingerprint=te["approval_fingerprint"], parts=len(order), pass_rows=sum(map(len, keys.values())),
                   per_source={s: dict(parts=len(ps), pass_rows=sum(len(keys[p]) for p in ps)) for s, ps in sorted(by_source.items())},
                   key_files_sha256=key_sha)
    (out_dir / "summary.json").open("x").write(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: summary[k] for k in ("parts", "pass_rows")}))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--training-eligibility", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    split(a.training_eligibility, a.out_dir)


if __name__ == "__main__":
    main()
