#!/usr/bin/env bash
# 한 스트림 실시간성 측정(experiments/stream_rtf.py)을 다른 작업이 없는 노드에서 돈다(2026-10-12).
# mxc 호스트는 CPU 부하가 커서(load 240 / 96 코어) 토큰마다 도는 Python 디코드 루프 시간이 부풀 수 있다 → 노드를 통째로 잡고 세 가지를
# GPU 0·1·2 에서 동시에 잰다(세 프로세스뿐이라 CPU 경합이 없다):
#   task 0 = 평가 경로(hf, 파일 전체 인코딩) fp32 — 기준
#   task 1 = 실시간 경로(graph 디코드 + NeMo 스트리밍 인코더, vapasr/hf/realtime.py) fp32 — 평가와 같은 출력(--parity 로 대조)
#   task 2 = 실시간 경로 bf16 — 속도 참고(출력은 fp32 평가와 다를 수 있다)
# 제출: bash slurm/submit-stream-rtf.sh [--partition P] (입력은 sbatch --export 로만: CK, OUT).
#SBATCH --job-name=SA_EVAL_Fullduplex
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
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
srun --ntasks=3 --ntasks-per-node=3 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=1 --gpu-bind=single:1 --label bash -c '
  case "$SLURM_PROCID" in
    0) opt="--dtype float32"; tag=hf-full-float32 ;;
    1) opt="--dtype float32 --engine graph --encoder-mode stream --parity"; tag=graph-stream-float32 ;;
    2) opt="--dtype bfloat16 --engine graph --encoder-mode stream"; tag=graph-stream-bfloat16 ;;
  esac
  "$PY" experiments/stream_rtf.py --checkpoint "$CK" --encoder /soundai/Model/nemotron-3.5-asr-streaming-0.6b \
    --manifest /soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-fleurs.jsonl \
    --out "$OUT" --per-set 24 --deltas 4 10 $opt > "$OUT/run-$tag.log" 2>&1'
