#!/bin/bash
# Stage 2 학습 — 다중 노드(기본 8 노드 × 8 GPU = 64 rank) DDP, thinker full FT + 온라인 Nemotron 특징, 1,930 h. 선점·시간 제한 시 저장 후 requeue.
#   제출:  sbatch --dependency=afterany:65787 --export=ALL,RUN=C,TRAIN=librispeech-960:kspon-full,EPOCHS=15,EVAL_EVERY=1000,SAVE_EVERY=500 slurm/s2_train.sbatch
#          (TRAIN 은 콜론 구분. LR/WARMUP/BS_EN/BS_KO 로 조정: 64 GPU 면 유효 배치가 8 GPU 의 8 배(EN 768 / KO 3072)라 LR 상향을 검토)
#   산출물: /soundai/Model/VAPASR/s1-<RUN>/  (ckpt-last.pt 재개용 · ckpt-<step>.pt · results.json · eval/ · DONE)
#   로그:   /soundai/users/tskim/VAPKT/logs/train/SA_SFT_FullDuplexStage5-<jobid>.out|.err
#SBATCH --job-name=SA_SFT_FullDuplexStage5
#SBATCH --partition=apex
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
RUN=${RUN:-C}
ROOT=/soundai/Model/VAPASR; OUT=$ROOT/s1-$RUN; mkdir -p "$OUT" /soundai/users/tskim/VAPKT/logs/train
echo "[$(date '+%F %T')] job $SLURM_JOB_ID RUN=$RUN nodes=$SLURM_NNODES ($SLURM_JOB_NODELIST) restart=${SLURM_RESTART_COUNT:-0}"
[ -f "$OUT/DONE" ] && { echo "이미 완료(DONE) — 종료"; exit 0; }

cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export MXC_CKPT_EXP_DIR=$ROOT
export MXC_DATA_MANIFEST_DIR=${MAN_IN:-$MXC_DATA_MANIFEST_DIR} VAPASR_ALIGN_ROOT=${ALIGN_ROOT:-${MAN_IN:-$MXC_DATA_MANIFEST_DIR}/align-asr-tn-v1}
export OMP_NUM_THREADS=8 NCCL_DEBUG=${NCCL_DEBUG:-WARN} TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
# 다중 노드 NCCL: 부트스트랩 소켓을 이더넷(eth0)에 고정. 기본값으로 두면 IPoIB(172.16.x.x, ib0–ib7) 주소를 골라 노드 간 "No route to host" 로 죽는다(job 65931).
# 데이터 경로는 IB verbs(NCCL_IB_HCA 자동). IB 자체가 문제면 NCCL_IB_DISABLE=1 로 이더넷 폴백.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-23} NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7} PYTHONFAULTHANDLER=1   # IB 링크 흔들림(status=12 retry exceeded, job 65963) 에 대한 여유. retry_cnt 는 3 비트라 최대 7. 반복되면 NCCL_IB_DISABLE=1
case "$(which python)" in "$MXC_CONDA_DIR"/envs/vapasr/*) ;; *) echo "!! conda env 아님: $(which python)"; exit 1;; esac
echo "manifest ← $MXC_DATA_MANIFEST_DIR | 정렬 ← $VAPASR_ALIGN_ROOT | conda $MXC_CONDA_DIR"

ADAPTER=$ROOT/uslm/s1-adapter-distill/adapter.pt
[ -f "$ADAPTER" ] || { echo "!! 증류 adapter 없음: $ADAPTER (slurm/s1_mono.sbatch 가 생성)"; exit 1; }

# ── run 설정 (rank 당 배치 EN/KO; 64 rank 면 유효 배치 EN 768 / KO 3072)
TRAIN=${TRAIN:-librispeech-960,kspon-full}; TRAIN=${TRAIN//:/,}; EPOCHS=${EPOCHS:-15}; FEATURES=${FEATURES:-online}
EVAL_EVERY=${EVAL_EVERY:-1000}; SAVE_EVERY=${SAVE_EVERY:-500}; BS_EN=${BS_EN:-12}; BS_KO=${BS_KO:-48}; LR=${LR:-4e-5}; LR_ADAPTER=${LR_ADAPTER:-1e-3}; WARMUP=${WARMUP:-200}
# 주의: srun bash -c "…" 안에 들어가므로 COMMON 에 줄바꿈이 있으면 두 번째 줄이 별개 명령이 되어 인자가 잘린다(job 65955: --full-ft·--lr 등 누락 → LoRA 로 실행됨)
COMMON="--tag $RUN --out-dir $OUT --train $TRAIN --features $FEATURES --epochs $EPOCHS --bs-en $BS_EN --bs-ko $BS_KO --init-adapter $ADAPTER --next-weight 0.3 --next-weight-ko 0.15 --eval-every $EVAL_EVERY --eval-bias 0 --sentinel-stream 10 --sentinel-utt 100 --save-every $SAVE_EVERY --warmup $WARMUP --resume auto"
case "$RUN" in
  A*) EXTRA="--lora-r 16 --lr ${LR_LORA:-4e-4} --lr-adapter $LR_ADAPTER" ;;
  *)  EXTRA="--full-ft --lr $LR --lr-adapter $LR_ADAPTER" ;;
esac
INIT_CKPT=${INIT_CKPT:-}; [ -n "$INIT_CKPT" ] && { [ -f "$INIT_CKPT" ] || { echo "!! INIT_CKPT 없음: $INIT_CKPT"; exit 1; }; EXTRA="$EXTRA --ckpt $INIT_CKPT"; }   # 이전 run 의 ckpt 로 가중치 초기화(옵티마이저는 새로)

# ── 선점·시간 제한: PREEMPT 파일(공유 FS) → 모든 rank 가 다음 step 경계에서 감지(all_reduce) → rank 0 저장 → 종료 → requeue
rm -f "$OUT/PREEMPT"
on_preempt() { echo "[$(date '+%T')] 신호 수신 → PREEMPT 표시, 저장 대기"; touch "$OUT/PREEMPT"; wait "$PID"; echo "requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_preempt USR1 TERM

ARGS="$COMMON $EXTRA"; case "$ARGS" in *$'\n'*) echo "!! 인자에 줄바꿈"; exit 1;; esac
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); PORT=$((20000 + SLURM_JOB_ID % 10000))
# rendezvous 주소는 호스트명이 아니라 첫 노드(= 이 배치 스크립트가 도는 노드)의 eth0 IPv4 로 — 호스트명이 IPoIB 로 풀리면 keep-alive/TCPStore 가 막힌다
MASTER_IP=$(ip -4 -o addr show "${NCCL_SOCKET_IFNAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1); [ -n "$MASTER_IP" ] || MASTER_IP=$MASTER
echo "[$(date '+%T')] torchrun × $SLURM_NNODES 노드 (rdzv $MASTER=$MASTER_IP:$PORT): $COMMON $EXTRA"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c "torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=\$SLURM_NODEID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_IP:$PORT --rdzv_id=$SLURM_JOB_ID experiments/s1_train_mono.py $ARGS" &
PID=$!; wait "$PID"; RC=$?
if [ "$RC" -eq 0 ] && grep -q '"steps_per_epoch"' "$OUT/results.json" 2>/dev/null && python - "$OUT" <<'PY'
import json, sys, os, torch
o = sys.argv[1]; ck = torch.load(os.path.join(o, "ckpt-last.pt"), map_location="cpu"); sys.exit(0 if ck["step"] >= ck["args"]["steps"] else 1)
PY
then touch "$OUT/DONE"; echo "[$(date '+%T')] 학습 완료 → DONE"
elif [ "$RC" -eq 0 ]; then echo "[$(date '+%T')] 미완료 종료(선점 저장 등) → requeue"; scontrol requeue "$SLURM_JOB_ID"
elif [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_RESTART:-3}" ]; then echo "[$(date '+%T')] 학습 실패 rc=$RC → 인프라 오류 가능성, requeue (restart ${SLURM_RESTART_COUNT:-0}/${MAX_RESTART:-3}, ckpt-last 에서 재개)"; scontrol requeue "$SLURM_JOB_ID"
else echo "[$(date '+%T')] 학습 실패 rc=$RC (restart 한도 초과)"; exit "$RC"; fi
