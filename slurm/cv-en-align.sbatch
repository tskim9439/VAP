#!/usr/bin/env bash
# Common Voice 4 English (CoVoST 2 en_de train), GPU stage (experiments/cv_en_align.py): mp3 → 16 kHz, Qwen3-ASR-1.7B check
# (WER <= 10 % against the human sentence, target = the sentence in asr-tn-v1 lexical form), Qwen3-ForcedAligner timing, words + tar.
# Every node uses all 8 GPUs (one worker per GPU). Preemption or the time limit requeues the job; finished shards are
# skipped on restart. Submit with slurm/submit-cv-en-align.sh (inputs via sbatch --export only).
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
echo "[$(date '+%F %T')] job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES restart=${SLURM_RESTART_COUNT:-0} OUT=$OUT"
test -f "$OUT/prefilter/prefilter.parquet"
test "${SLURM_NTASKS:?}" -eq "$((SLURM_JOB_NUM_NODES * 8))"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Never override CUDA_VISIBLE_DEVICES: Slurm supplies the allocated device.
set +e
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 --label \
    "$PY" -u experiments/cv_en_align.py worker --out "$OUT" ${MAX_WER:+--max-wer "$MAX_WER"}
worker_rc=$?
"$PY" experiments/cv_en_align.py summarize --out "$OUT" && "$PY" experiments/cv_en_align.py lists --out "$OUT"
summary_rc=$?
set -e
if (( worker_rc != 0 )); then exit "$worker_rc"; fi
exit "$summary_rc"
