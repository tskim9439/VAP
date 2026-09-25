#!/usr/bin/env bash
# Approved speechlm semcommit teacher labeling. Submit only after gold gate and
# speechlm QC/human approval. One model/process/GPU; 2 nodes x 8 H200 GPUs.
#SBATCH --job-name=SA_SFT_FullDuplex
#SBATCH --partition=apex
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=slurmmxch200v5-hpc-52
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail

PROJECT=/soundai/users/tskim/VAPKT
DATA=/soundai/users/tskim/VAPKT-data/data
PY=/soundai/users/tskim/VAPKT-data/conda/envs/semcommit-teacher/bin/python
QWEN=/soundai/Model/Qwen3.8-27B
EXAONE=/soundai/Model/EXAONE-4.0-32B
: "${WORD_INDEX:?tab-separated pool and approved words path required}"
: "${APPROVAL_SUMMARY:?human-approved training summary required}"
: "${GATE_ACCEPTED:?accepted gold gate attestation required}"
: "${THRESHOLDS:?accepted gate thresholds.json required}"
: "${A_KIND:?qwen38 or exaone4 required}"
: "${C_KIND:?qwen38 or exaone4 required}"
export A_KIND C_KIND
OUT=${OUT:-$DATA/semcommit-work/labels/speechlm-approved-${SLURM_JOB_ID}}
cd "$PROJECT"
test -x "$PY" && test -f "$WORD_INDEX" && test -f "$APPROVAL_SUMMARY"
test -f "$GATE_ACCEPTED" && test -f "$THRESHOLDS"
test -d "$QWEN" && test -d "$EXAONE"
# 기본 2 노드 × 8 GPU. 파일럿은 sbatch --nodes=1 --ntasks=8 로 덮어쓴다(노드당 8 작업·작업당 GPU 1 장은 고정).
test "${SLURM_JOB_NUM_NODES:?}" -ge 1 && test "${SLURM_NTASKS:?}" -eq $((SLURM_JOB_NUM_NODES * 8))
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" - "$APPROVAL_SUMMARY" "$GATE_ACCEPTED" "$THRESHOLDS" "$WORD_INDEX" <<'PY'
import hashlib, json, os, pathlib, sys
approval, gate, thresholds, index = map(pathlib.Path, sys.argv[1:])
a, g = json.loads(approval.read_text()), json.loads(gate.read_text())
if not a.get("complete") or a.get("training_eligible") is not True:
    raise SystemExit("speechlm approval is not training eligible")
if g.get("accepted") is not True or g.get("thresholds_sha256") != hashlib.sha256(thresholds.read_bytes()).hexdigest():
    raise SystemExit("gold gate acceptance/threshold digest mismatch")
selected = next((r for r in g.get("results", []) if r.get("kind") == g.get("selected_kind")), None)
report = thresholds.with_name("thresholds.report.json")
if (selected is None or not report.is_file()
        or selected.get("thresholds_report_sha256") != hashlib.sha256(report.read_bytes()).hexdigest()):
    raise SystemExit("selected gate report is missing or altered")
extra = json.loads(report.read_text()).get("extra")
if not isinstance(extra, list) or not extra or not all(isinstance(x, str) for x in extra):
    raise SystemExit("selected candidate sources are invalid")
expected_kind = {"qwen": "qwen38", "exaone": "exaone4"}.get(g.get("selected_kind"))
if not expected_kind or (expected_kind, expected_kind) != (os.environ["A_KIND"], os.environ["C_KIND"]):
    raise SystemExit("A/C model kind does not match accepted gold gate")
if (a.get("gate_acceptance_sha256") != hashlib.sha256(gate.read_bytes()).hexdigest()
        or a.get("source_approval_sha256") != g.get("source_approval_sha256")
        or not a.get("approval_fingerprint")):
    raise SystemExit("training eligibility is not bound to this gold gate and source approval")
rows = [line.split("\t", 1) for line in index.read_text().splitlines() if line.strip()]
if not rows or any(len(r) != 2 or r[0] not in ("short", "long") or not pathlib.Path(r[1]).is_file() for r in rows):
    raise SystemExit("invalid approved word index")
if len({r[1] for r in rows}) != len(rows):
    raise SystemExit("duplicate words path in index")
print(f"PREFLIGHT_OK shards={len(rows)} short={sum(r[0]=='short' for r in rows)} long={sum(r[0]=='long' for r in rows)}")
PY
EXTRA_CANDIDATES=$("$PY" - "$THRESHOLDS" <<'PY'
import json, pathlib, sys
report = pathlib.Path(sys.argv[1]).with_name("thresholds.report.json")
print(",".join(json.loads(report.read_text())["extra"]))
PY
)
mkdir -p "$OUT"

export PROJECT DATA PY QWEN EXAONE OUT WORD_INDEX THRESHOLDS A_KIND C_KIND APPROVAL_SUMMARY EXTRA_CANDIDATES
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task=8 \
  --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
  bash "$PROJECT/slurm/semcommit-main-label-worker.sh"
