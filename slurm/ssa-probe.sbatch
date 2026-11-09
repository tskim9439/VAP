#!/bin/bash
# P1 화자 조건 인코더 단독 검증(2026-11-09, experiments/ssa_probe.py — wiki output-agn-reduction-20261104 12 절) 노드 하나 8 GPU.
#   MODE=train: torchrun 8 프로세스로 학습(OUT 에 step-N.pt 가 있으면 이어 간다 — 선점 requeue 도 같다). ARGS 는 train 에 덧붙인다.
#   MODE=eval : EVALS 의 판(';' 로 구분, 판 = '이름|--model …|--act …|--rc …')마다 8 GPU 에 회의를 나눠(rank r/8) 돌린다. 산출 RES/ssa-probe-<이름>(끝난 회의는 건너뛴다).
#   제출은 slurm/submit-ssa-probe.sh(사용자). 아무것도 지우지 않는다.
#SBATCH --job-name=SA_SFT_FullDuplex
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
: "${MODE:?MODE=train|eval}"
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}                              # NFS 위 conda import 지연 회피(v3-eval-node.sbatch 와 같다)
PY=python
if [ -n "$VAPASR_LOCAL_ENV" ]; then
  bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  if [ -x "$VAPASR_LOCAL_ENV/bin/python" ]; then export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" PY="$VAPASR_LOCAL_ENV/bin/python"; fi
fi
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
echo "[$(date '+%F %T')] ssa-probe $MODE job $SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0}"
if [ "$MODE" = train ]; then
  : "${OUT:?OUT}"
  [ -f "$OUT/DONE" ] && { echo "이미 끝: $OUT"; exit 0; }
  torchrun --standalone --nproc_per_node 8 experiments/ssa_probe.py train --out "$OUT" ${ARGS:-}; rc=$?
  echo "[$(date '+%F %T')] TRAIN_EXIT $rc"; exit $rc
fi
: "${EVALS:?EVALS}"; RES=${RES:-/soundai/users/tskim/VAPKT-data/results}
IFS=';' read -ra V <<< "$EVALS"
for v in "${V[@]}"; do
  name=${v%%|*}; opts=${v#*|}; opts=${opts//|/ }; O=$RES/ssa-probe-$name
  echo "[$(date '+%F %T')] 평가 $name: $opts → $O"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY experiments/ssa_probe.py eval $opts --out "$O" --rank $g --world-size 8 >> "$O.rank$g.log" 2>&1 &
    sleep 5
  done
  wait; echo "[$(date '+%F %T')] 평가 $name 끝 — 회의 $(cat $O/predictions-rank*.jsonl 2>/dev/null | wc -l)"
done
echo "[$(date '+%F %T')] EVAL_DONE"
