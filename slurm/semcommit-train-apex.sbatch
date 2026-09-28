#!/bin/bash
# Semantic-commit 학습(experiments/semcommit_train.py) — 라벨링 중간 스냅숏(semcommit_collect_done.py)으로 다중 노드 DDP.
# 제출은 slurm/submit-semcommit-train-apex.sh(입력은 sbatch --export 로만 전달). 선점·시간 제한 시 PREEMPT 로 저장 뒤 requeue,
# 같은 OUT 에서 checkpoint-N 으로 이어서(지문이 다르면 거부). 완료 시 OUT/final + 재로드 parity(parity.json) + DONE.
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
#SBATCH --exclude=slurmmxch200v5-hpc-52
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
: "${OUT:?output directory}" "${SNAP:?snapshot directory with main/short-{words,labels}.list}" "${INIT:?init HF checkpoint}" "${ARGS_EXTRA=}"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID nodes=$SLURM_NNODES ($SLURM_JOB_NODELIST) restart=${SLURM_RESTART_COUNT:-0} OUT=$OUT SNAP=$SNAP INIT=$INIT"
[ -f "$OUT/DONE" ] && { echo "이미 완료(DONE) — 종료"; exit 0; }
mkdir -p "$OUT" /soundai/users/tskim/VAPKT/logs/train
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}          # NFS 위 conda import 지연 회피(s3_train_hf.sbatch 와 같다)
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
# speechlm KO 오디오는 databricks 의 tar 안에 있다 — 파이프라인이 만든 tar 인덱스를 읽기만 한다(없으면 tar 전체 스캔 뒤 읽기 전용 경로에
# 인덱스 쓰기를 시도해 실패). voxpopuli 처럼 자체 <tar>.index.json 이 있는 tar 는 그것을 쓴다(vapasr/data/archive.py).
export VAPASR_TAR_INDEX_ROOT=${VAPASR_TAR_INDEX_ROOT:-/soundai/users/tskim/VAPKT-data/data/speechlm-asr-tar-index-v1}

ARGS="--init $INIT --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $OUT \
 --train-words @$SNAP/main-words.list --train-labels @$SNAP/main-labels.list \
 --short-train-words @$SNAP/short-words.list --short-train-labels @$SNAP/short-labels.list \
 $ARGS_EXTRA"
ARGS=$(echo $ARGS)                                                      # 한 줄로(srun bash -c 안에서 줄바꿈이 명령을 끊지 않게)
rm -f "$OUT/PREEMPT"
on_preempt() { echo "[$(date '+%T')] 신호 수신 → PREEMPT 표시, 저장 대기"; touch "$OUT/PREEMPT"; wait "$PID"; echo "requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_preempt USR1 TERM
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); PORT=$((20000 + SLURM_JOB_ID % 10000))
MASTER_IP=$(ip -4 -o addr show "$NCCL_SOCKET_IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1); [ -n "$MASTER_IP" ] || MASTER_IP=$MASTER
echo "[$(date '+%T')] torchrun × $SLURM_NNODES 노드 × 8 GPU (rdzv $MASTER_IP:$PORT): $ARGS"
srun --export=ALL --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  /bin/bash -c "torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:$PORT --rdzv_id=$SLURM_JOB_ID experiments/semcommit_train.py $ARGS" &
PID=$!; wait "$PID"; RC=$?
if [ "$RC" -eq 0 ] && [ -f "$OUT/DONE" ]; then echo "[$(date '+%T')] 학습 완료 → DONE (parity: $OUT/parity.json)"
elif [ "$RC" -eq 0 ]; then echo "[$(date '+%T')] 미완료 종료(선점 저장 등) → requeue"; scontrol requeue "$SLURM_JOB_ID"
elif [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_RESTART:-3}" ]; then echo "[$(date '+%T')] 실패 rc=$RC → requeue (${SLURM_RESTART_COUNT:-0}/${MAX_RESTART:-3})"; scontrol requeue "$SLURM_JOB_ID"
else echo "[$(date '+%T')] 실패 rc=$RC (restart 한도 초과)"; exit "$RC"; fi
