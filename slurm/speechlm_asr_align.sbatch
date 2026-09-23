#!/usr/bin/env bash
# CPU preparation must finish before submission. Every node uses all 8 GPUs.
#SBATCH --job-name=VAPKT-speechlm-asr-align
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/speechlm-asr-align-v1-smoke-20260922}
cd "$PROJECT"
test -f "$OUT/config.json"
test "${SLURM_NTASKS:?}" -eq "$((SLURM_JOB_NUM_NODES * 8))"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Never override CUDA_VISIBLE_DEVICES: Slurm supplies the allocated device.
set +e
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 --label \
    "$PY" -u experiments/speechlm_asr_align.py worker --out "$OUT"
worker_rc=$?
"$PY" experiments/speechlm_asr_align.py summarize --out "$OUT"
summary_rc=$?
set -e
if (( worker_rc != 0 )); then exit "$worker_rc"; fi
exit "$summary_rc"
