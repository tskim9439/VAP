#!/usr/bin/env bash
# CPU-only full QC; no GPU model inference.
#SBATCH --job-name=SA_SFT_FullDuplexStage5-align-qc
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
cd /soundai/users/tskim/VAPKT
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
SOURCE=${SOURCE:-/soundai/users/tskim/VAPKT-data/data/selection/align-qwen-verbatim-v1-20260921}
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/selection/align-qwen-verbatim-v1-20260921-qc-v1}
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -u experiments/selection_alignment_qc.py --source "$SOURCE" --out "$OUT" \
  --workers "${WORKERS:-32}" --samples-per-cell "${SAMPLES_PER_CELL:-3}"
