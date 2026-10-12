#!/usr/bin/env bash
# 한 스트림 실시간성 측정(experiments/stream_rtf.py)을 다른 작업이 없는 노드에서 돈다(2026-10-12).
# mxc 호스트는 CPU 부하가 커서(load 240 / 96 코어) 토큰마다 도는 Python 디코드 루프 시간이 부풀 수 있다 → 노드를 통째로 잡고
# GPU 0 에서 fp32, GPU 1 에서 bf16 을 동시에 잰다(두 프로세스뿐이라 CPU 경합이 없다).
# 제출: bash slurm/submit-stream-rtf.sh [--partition P] (입력은 sbatch --export 로만: CK, OUT).
#SBATCH --job-name=SA_EVAL_Fullduplex
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
CK=${CK:?CK(체크포인트)를 --export 로 주세요}
OUT=${OUT:?OUT(결과 폴더)를 --export 로 주세요}
export CK OUT PY
cd "$PROJECT"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) CK=$CK OUT=$OUT"
test -f "$CK/config.json"; mkdir -p "$OUT"
export PYTHONPATH="$PROJECT" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VAPASR_STAGE_DIR=/dev/shm/tskim-vapasr-infer-eval
srun --ntasks=2 --ntasks-per-node=2 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=1 --gpu-bind=single:1 --label bash -c '
  dt=$([ "$SLURM_PROCID" = 0 ] && echo float32 || echo bfloat16)
  "$PY" experiments/stream_rtf.py --checkpoint "$CK" --encoder /soundai/Model/nemotron-3.5-asr-streaming-0.6b \
    --manifest /soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-fleurs.jsonl \
    --out "$OUT" --per-set 24 --deltas 4 10 --dtype $dt > "$OUT/run-$dt.log" 2>&1'
