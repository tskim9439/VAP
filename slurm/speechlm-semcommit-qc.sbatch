#!/usr/bin/env bash
# Freeze completed alignment shards; exact dedup and alignment QC only.
# Never automatically grants training eligibility.
#SBATCH --job-name=SA_SEM_speechlm_qc
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=slurmmxch200v5-hpc-52
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail

PROJECT=/soundai/users/tskim/VAPKT
DATA=/soundai/users/tskim/VAPKT-data/data
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
SOURCE=$DATA/speechlm-asr-align-v1-full-b128-20260923/results
APPROVED=$DATA/selection/approved-qwen-verbatim-mono-v1.1-20260921
OUT=$DATA/semcommit-work/qc/speechlm-partial-${SLURM_JOB_ID}
cd "$PROJECT"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -x "$PY" && test -d "$SOURCE" && test -f "$APPROVED/supervision-inputs.jsonl"
"$PY" -u experiments/speechlm_semcommit_qc.py \
  --results "$SOURCE" --approved-view "$APPROVED" --out "$OUT" \
  --max-parts "${MAX_PARTS:-0}"
echo "QC_REPORT=$OUT/summary.json"
