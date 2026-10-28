#!/bin/bash
# V3 평가 노드(1 노드 8 GPU, apex — 비선점): 작업 목록 TASKS 의 줄 '<GPU 0–7>|<명령>' 을 GPU 마다 위에서부터 차례로 돈다(GPU 를 목록에서 정해 둔다 — 잠금 없음).
#   명령은 보통 'bash raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh <작업> <run> <대상>'(앞에 MODEL_ROOT=… 를 붙일 수 있다).
#   제출(사용자): sbatch --export=ALL,TASKS=<목록 파일> slurm/v3-eval-node.sbatch
#   작업마다 CUDA_VISIBLE_DEVICES=GPU_IDX=<GPU>, PY = 노드 로컬 env python. 각 작업은 끝났거나 시작된 산출이 있으면 건너뛰고 아무것도 지우지 않는다.
#   진행: logs/train/SA_V3_EVAL-<job>.out(작업 시작·끝·종료 코드).
#   --requeue(2026-10-23): low_p_hpc 처럼 선점되는 파티션에서 다시 큐에 든다 — 작업은 완료 표시로 건너뛰거나 이어 한다(v3_eval_task.sh, avg_checkpoints.py).
#SBATCH --job-name=SA_V3_EVAL
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
: "${TASKS:?작업 목록 파일 TASKS}"
[ -f "$TASKS" ] || { echo "목록 없음: $TASKS"; exit 1; }
echo "[$(date '+%F %T')] V3 평가 노드 job $SLURM_JOB_ID node=$(hostname) 작업 $(grep -c '^[0-7]|' "$TASKS") 개 ← $TASKS"
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}                              # NFS 위 conda import 지연 회피(stage3-train-apex.sbatch 와 같다)
if [ -n "$VAPASR_LOCAL_ENV" ]; then
  bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  if [ -x "$VAPASR_LOCAL_ENV/bin/python" ]; then
    export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" PY="$VAPASR_LOCAL_ENV/bin/python"; mkdir -p "$TMPDIR"
    echo "[$(date '+%T')] 로컬 env 사용 $PY"
  else echo "!! 로컬 env 준비 실패 → NFS env"; fi
fi
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
worker() {
  local g=$1 cmd n=0 rc
  while IFS= read -r cmd; do
    n=$((n + 1)); echo "[$(date '+%F %T')] GPU $g ← $n: $cmd"
    CUDA_VISIBLE_DEVICES=$g GPU_IDX=$g bash -c "$cmd"; rc=$?                                   # 2026-10-28: $(date) 가 먼저 돌아 $? 가 늘 0 이었다
    echo "[$(date '+%F %T')] GPU $g 끝 $n (exit $rc)"
  done < <(grep "^$g|" "$TASKS" | cut -d'|' -f2-)
}
for g in 0 1 2 3 4 5 6 7; do worker $g & sleep 20; done                                         # 20 s 간격(NFS 부하 분산)
wait
echo "[$(date '+%F %T')] V3_EVAL_NODE_DONE"
