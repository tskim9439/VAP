#!/usr/bin/env bash
# Qwen 원출력 forced alignment. 제출 전 slurm/selection-align-qwen.md 참조.
# 기본 노드 1개. --nodes=N으로 늘리며, 항상 노드당 GPU 8개 전체 할당.
# task 당 GPU 1, CPU 4 (decode 3 + torch 1).
#SBATCH --job-name=SA_SFT_FullDuplexStage5-qwen-align
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail

PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
SOURCE=${SOURCE:-/soundai/users/tskim/VAPKT-data/data/selection/pair-v0.6-full-20260921}
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/selection/align-qwen-verbatim-v1-20260921}
cd "$PROJECT"
test -x "$PY"
test "${SLURM_JOB_NUM_NODES:?Run with sbatch}" -ge 1
expected_tasks=$((SLURM_JOB_NUM_NODES * 8))
test "${SLURM_NTASKS:?}" -eq "$expected_tasks"
test "${SLURM_CPUS_PER_TASK:?}" -ge 4
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES는 srun이 설정한다. 물리 GPU 번호로 덮어쓰지 않는다.
args=(--source "$SOURCE" --out "$OUT" --batch "${BATCH:-64}"
      --batch-sec "${BATCH_SEC:-480}" --mem-frac "${MEM_FRAC:-0.9}"
      --io-threads 3
      --aligner "${ALIGNER:-/soundai/Model/Qwen3-ForcedAligner-0.6B}"
      --tokenizer "${TOKENIZER:-/soundai/Model/Qwen3-ASR-0.6B}")
if [[ ${SMOKE:-0} == 1 ]]; then
    OUT="${OUT}-smoke"
    args+=(--out "$OUT" --max-shards 8 --max-rows 16)
fi
if [[ ${RESUME:-0} == 1 ]]; then
    test -f "$OUT/config.json" || { echo "RESUME=1이지만 $OUT/config.json 없음" >&2; exit 1; }
    echo "RESUME_EXISTING_CONFIG $OUT/config.json"
else
    "$PY" -u experiments/selection_align_qwen.py prepare "${args[@]}"
fi

# srun의 task별 GPU binding을 사용해 각 task에는 할당 GPU 1개만 보이게 한다.
# One failed worker fails the job; successful atomic shards remain resumable.
set +e
srun --ntasks="$expected_tasks" --ntasks-per-node=8 --cpus-per-task=4 \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 --label \
    "$PY" -u experiments/selection_align_qwen.py worker --out "$OUT"
worker_rc=$?
"$PY" -u experiments/selection_align_qwen.py summarize --out "$OUT"
summary_rc=$?
set -e
if (( worker_rc != 0 )); then exit "$worker_rc"; fi
exit "$summary_rc"
