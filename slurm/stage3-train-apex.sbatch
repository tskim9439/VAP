#!/bin/bash
# Stage 3 학습(experiments/stage3_train.py) — 노드당 8 GPU torchrun. 제출은 slurm/submit-stage3-train.sh(사용자).
# 입력은 sbatch --export 로만: OUT(산출 폴더), ARGS_EXTRA(stage3_train.py 인자, 쉼표는 %2C), MAX_RESTART(실패 재시도 한도, 기본 3).
# SCRIPT(2026-10-25, 기본 experiments/stage3_train.py): 같은 규약(--nemotron-dir·--out-dir 을 받고 PREEMPT·DONE 을 쓰는)의 다른 학습 스크립트 — 예 experiments/diar_train.py(E1 화자 분리 가지).
# 선점 신호(USR1/TERM) → PREEMPT 표시 → 학습이 체크포인트를 저장하고 끝나면 requeue. 아무것도 지우지 않는다(PREEMPT 표시 파일만 시작 때 다시 만든다).
# 감시자(watchdog): 실패(랑데부 시간 초과 등) 뒤 srun 이 한 노드 작업을 못 끝내고 멈추면 WD_GRACE(300 s) 뒤 requeue(한도 넘으면 취소).
#SBATCH --job-name=SA_SFT_FullDuplex
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
: "${OUT:?output directory}" "${ARGS_EXTRA:?stage3_train.py arguments}"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID nodes=$SLURM_NNODES ($SLURM_JOB_NODELIST) restart=${SLURM_RESTART_COUNT:-0}/${MAX_RESTART:-3} OUT=$OUT"
[ -f "$OUT/DONE" ] && { echo "이미 완료(DONE) — 종료"; exit 0; }
pre_start() { echo "[$(date '+%T')] 신호 수신(학습 시작 전) → requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }   # 91580: env 복사 중(33 s) 선점이면 아래 on_preempt 가 아직 없어 requeue 줄이 없었다
trap pre_start USR1 TERM
mkdir -p "$OUT" /soundai/users/tskim/VAPKT/logs/train
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}          # NFS 위 conda import 지연 회피(semcommit-train-apex.sbatch 와 같다)
if [ -n "$VAPASR_LOCAL_ENV" ]; then
  if srun --export=ALL --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 /bin/bash scripts/stage-env-local.sh 2>&1 \
       | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /" \
     && srun --export=ALL --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 /usr/bin/test -x "$VAPASR_LOCAL_ENV/bin/python"; then
    export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp"; mkdir -p "$TMPDIR"
    echo "[$(date '+%T')] 로컬 env 사용 $VAPASR_LOCAL_ENV"
  else echo "!! 로컬 env 준비 실패 → NFS env 로 진행"; fi
fi
export OMP_NUM_THREADS=8 NCCL_DEBUG=${NCCL_DEBUG:-WARN} TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23} NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7} PYTHONFAULTHANDLER=1
export VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_TAR_INDEX_ROOT=${VAPASR_TAR_INDEX_ROOT:-/soundai/users/tskim/VAPKT-data/data/speechlm-asr-tar-index-v1}

ARGS="--nemotron-dir $MXC_NEMOTRON_DIR --out-dir $OUT ${ARGS_EXTRA//%2C/,}"
ARGS=$(echo $ARGS)
rm -f "$OUT/PREEMPT"
on_preempt() { echo "[$(date '+%T')] 신호 수신 → PREEMPT 표시, 저장 대기"; touch "$OUT/PREEMPT"; wait "$PID"; echo "requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_preempt USR1 TERM
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); PORT=$((20000 + SLURM_JOB_ID % 10000))
MASTER_IP=$(ip -4 -o addr show "$NCCL_SOCKET_IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1); [ -n "$MASTER_IP" ] || MASTER_IP=$MASTER
echo "[$(date '+%T')] torchrun × $SLURM_NNODES 노드 × 8 GPU (rdzv $MASTER_IP:$PORT): $ARGS"
srun --export=ALL --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  /bin/bash -c "torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:$PORT --rdzv_id=$SLURM_JOB_ID ${SCRIPT:-experiments/stage3_train.py} $ARGS" &
PID=$!
watchdog() {  # 랑데부 실패 등으로 srun 이 작업을 끝내라고 했는데도 한 노드 작업이 안 끝나면(91256: hpc-102, 1 시간 넘게 RUNNING) 유예 뒤 스스로 정리
  local err=/soundai/users/tskim/VAPKT/logs/train/$SLURM_JOB_NAME-$SLURM_JOB_ID.err; local off; off=$(stat -c %s "$err" 2>/dev/null || echo 0)
  while kill -0 "$PID" 2>/dev/null; do
    sleep "${WD_POLL:-60}"
    tail -c +$((off + 1)) "$err" 2>/dev/null | grep -qE "RendezvousTimeoutError|Terminating StepId" || continue
    sleep "${WD_GRACE:-300}"; kill -0 "$PID" 2>/dev/null || return 0
    if [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_RESTART:-3}" ]; then echo "[$(date '+%T')] 실패 뒤 srun 이 ${WD_GRACE:-300} s 넘게 안 끝남 → requeue (${SLURM_RESTART_COUNT:-0}/${MAX_RESTART:-3})"; scontrol requeue "$SLURM_JOB_ID"
    else echo "[$(date '+%T')] 실패 뒤 srun 이 안 끝남, restart 한도 초과 → 취소"; scancel "$SLURM_JOB_ID"; fi
    return 0
  done
}
watchdog & WD=$!
wait "$PID"; RC=$?; kill "$WD" 2>/dev/null
if [ "$RC" -eq 0 ] && [ -f "$OUT/DONE" ]; then echo "[$(date '+%T')] 학습 완료 → DONE (parity: $OUT/parity.json)"
elif [ "$RC" -eq 0 ]; then echo "[$(date '+%T')] 미완료 종료(선점 저장 등) → requeue"; scontrol requeue "$SLURM_JOB_ID"
elif [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_RESTART:-3}" ]; then echo "[$(date '+%T')] 실패 rc=$RC → requeue (${SLURM_RESTART_COUNT:-0}/${MAX_RESTART:-3})"; scontrol requeue "$SLURM_JOB_ID"
else echo "[$(date '+%T')] 실패 rc=$RC (restart 한도 초과)"; exit "$RC"; fi
