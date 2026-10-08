#!/usr/bin/env bash
# YODAS-Granary German, GPU stage (experiments/yodas_de_align.py): Qwen3-ASR-1.7B transcription, selection at
# WER <= 3 % against the Granary text, Qwen3-ForcedAligner timing, words rows and audio tar shards.
# Every node uses all 8 GPUs (one worker per GPU). Preemption or the time limit requeues the job; finished shards are
# skipped on restart. Submit with slurm/submit-yodas-de-align.sh (inputs via sbatch --export only).
# SELECT_FLAGS=dur_long MAX_DUR=40 runs the long pass (30-40 s rows) into separate <shard>-dur_long folders.
#SBATCH --job-name=SA_EVAL_Fullduplex
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
OUT=${OUT:?OUT 를 --export 로 주세요}
cd "$PROJECT"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES restart=${SLURM_RESTART_COUNT:-0} OUT=$OUT pass=${SELECT_FLAGS:-main} max_dur=${MAX_DUR:-30}"
test -f "$OUT/prefilter/prefilter.parquet"
test "${SLURM_NTASKS:?}" -eq "$((SLURM_JOB_NUM_NODES * 8))"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Never override CUDA_VISIBLE_DEVICES: Slurm supplies the allocated device.
set +e
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 --label \
    "$PY" -u experiments/yodas_de_align.py worker --out "$OUT" ${MAX_WER:+--max-wer "$MAX_WER"} \
    ${SELECT_FLAGS:+--select-flags "$SELECT_FLAGS"} ${MAX_DUR:+--max-dur "$MAX_DUR"}
worker_rc=$?
"$PY" experiments/yodas_de_align.py summarize --out "$OUT" ${SELECT_FLAGS:+--select-flags "$SELECT_FLAGS"}
summary_rc=$?
set -e
if (( worker_rc != 0 )); then exit "$worker_rc"; fi
exit "$summary_rc"
