#!/bin/bash
# Phase 2 lane 디코드 평가(GPU 1 장) — 로그인 노드 GPU 가 모두 사용 중일 때의 대안.
#   sbatch --partition=apex --export=ALL,SETS=aihub71631:seen:8,ami:seen:6,nikl2020:heldout:10,chime6:heldout:6,OUT=/soundai/users/tskim/VAPKT-data/eval/p2-D1 slurm/p2_eval_lanes.sbatch
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2eval
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" && mkdir -p "$TMPDIR"; fi
DATA=${DATA:-/soundai/users/tskim/VAPKT-data/data/phase2}
python experiments/p2_eval_lanes.py --model "${MODEL:-/soundai/Model/VAPASR/p2-D1/final}" --data "$DATA" --mono-cache "$DATA/mono" --sets "${SETS:?}" --out "${OUT:?}" ${EXTRA:-}
echo EXIT=0
