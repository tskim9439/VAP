#!/usr/bin/env bash
# 제출 예: sbatch --partition=apex --export=ALL,MANIFEST=<manifest>,OUT=<새 결과 폴더> slurm/p2_eval_vibevoice.sbatch
#SBATCH --job-name=SA_SFT_FullDuplexStage5-vibe-eval
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
cd /soundai/users/tskim/VAPKT
EVAL_ROOT=${EVAL_ROOT:-/soundai/users/tskim/VAPKT-data/baselines/vibevoice-1.5b-v1}
PY="$EVAL_ROOT/venv/bin/python"
test -x "$PY" || { echo "먼저 experiments/p2_setup_vibevoice.sh 실행 필요"; exit 1; }
test -f "$EVAL_ROOT/SETUP_READY" || { echo "평가 환경 준비가 완료되지 않음"; exit 1; }
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
"$PY" -u experiments/p2_eval_vibevoice.py --manifest "${MANIFEST:?}" \
  --model "$EVAL_ROOT/model" --vibevoice-source "$EVAL_ROOT/source" \
  --out "${OUT:?}" --speed "${SPEED:-throughput}"
"$PY" experiments/p2_score_baseline.py --manifest "$MANIFEST" --results "$OUT"
echo "EVALUATION_DONE"
