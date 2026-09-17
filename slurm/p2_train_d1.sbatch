#!/bin/bash
# Phase 2 D1 실학습 — 노드 1 개 × GPU 8 장(torchrun DDP; 사용자 지시 2026-09-16 "노드당 8 GPU"). 정본 output-phase2-lane-plan §9·§13.
#   제출:  sbatch --partition=apex --export=ALL,RUN=D1,MAX_TOKENS=28000,MIX=sqrt,EPOCHS=4 slurm/p2_train_d1.sbatch
#   D1b:   sbatch --partition=apex --export=ALL,RUN=D1b,MAX_TOKENS=<probe>,MIX=sqrt,EPOCHS=5,LR=1e-4,WARMUP=150,TRAIN_ENCODER=1,LR_ENCODER=1e-5,DELAY_ONSET=2,TAG_WEIGHT=2,START_MODE=mixed slurm/p2_train_d1.sbatch
#          (인코더 해동 학습 — 사용자 지시 2026-09-17; 창 시작 mixed·δ_onset 2·태그 가중 2 — output-phase2-d1-lane-eval §4; 학습 종료 시 final 재로드 parity 검사가 기본으로 돈다)
#          옵션: EPOCHS(기본 3) 또는 STEPS(>0 이면 step 수 고정), CORPORA, MAX_TOKENS(rank 당 동적 배치 예산; 0 이면 BS_EN/BS_KO 고정 8/8), MAX_BS(48), MIX(equal|sqrt|prop),
#                LR(6e-5), LR_ADAPTER(1e-3), WARMUP(150), SAVE_EVERY(200), WORKERS(rank 당 데이터 워커 6), GPUS(8), TRAIN_ENCODER(1 이면 인코더 해동, LR_ENCODER 기본 1e-5),
#                DELAY_ONSET(0), TAG_WEIGHT(1), START_MODE(grid|mixed), RANDOM_FRAC(0.5), EXTRA(추가 인자)
#          동적 배치 예산은 experiments/p2_mem_probe.py 실측(GPU 1 장 기준 28 k → 106 GB 예약, 74 %)으로 정한다. mono 캐시는 scripts/p2-mono-local.sh 로 미리 만든다.
#   산출물: /soundai/Model/VAPASR/p2-<RUN>/{checkpoint-N,final,DONE,run.json,results.json,tb}
#   재개: 같은 명령을 다시 내면 마지막 완성 checkpoint 에서 이어간다(--resume auto). 선점(USR1/TERM) → PREEMPT 파일 → 모든 rank 가 다음 step 에서 저장 후 종료 → requeue.
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2d1
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
RUN=${RUN:?RUN=D1 등}; ROOT=/soundai/Model/VAPASR; R=$ROOT/p2-$RUN; DATA=/soundai/users/tskim/VAPKT-data/data/phase2; INIT=${INIT:-$ROOT/hf-E2/final}; GPUS=${GPUS:-8}
mkdir -p "$R" "$DATA/mono" /soundai/users/tskim/VAPKT/logs/train
echo "[$(date '+%F %T')] p2 D1 job $SLURM_JOB_ID RUN=$RUN node=$(hostname) gpus=$GPUS init=$INIT restart=${SLURM_RESTART_COUNT:-0}"
[ -f "$R/DONE" ] && { echo "이미 완료(DONE) — 종료"; exit 0; }
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" && mkdir -p "$TMPDIR"; fi
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=${NCCL_DEBUG:-WARN} PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}      # 단일 노드라 GPU 간은 NVLink 지만 부트스트랩 소켓은 eth0 고정(s3_train_hf.sbatch 와 동일)
ARGS="--init $INIT --data $DATA --corpora ${CORPORA:-aihub71631,aihub134-1,aihub134-2,otoSpeech,ami,notsofar,icsi} --mono-cache $DATA/mono --out-dir $R --max-steps ${STEPS:-0} --epochs ${EPOCHS:-3} --max-tokens ${MAX_TOKENS:-0} --max-bs ${MAX_BS:-48} --mix ${MIX:-equal} --bs-en ${BS_EN:-8} --bs-ko ${BS_KO:-8} --lr ${LR:-6e-5} --lr-adapter ${LR_ADAPTER:-1e-3} --lr-encoder ${LR_ENCODER:-1e-5} --warmup ${WARMUP:-150} --log-every ${LOG_EVERY:-20} --save-every ${SAVE_EVERY:-200} --num-workers ${WORKERS:-6} --delay-onset ${DELAY_ONSET:-0} --tag-weight ${TAG_WEIGHT:-1} --start-mode ${START_MODE:-grid} --random-frac ${RANDOM_FRAC:-0.5} ${TRAIN_ENCODER:+--train-encoder} ${EXTRA:-}"
# ── 선점·시간 제한: PREEMPT 파일 → 모든 rank 가 다음 step 경계에서 감지(gloo all_reduce) → 저장 → 종료 → requeue
rm -f "$R/PREEMPT"
on_preempt() { echo "[$(date '+%T')] 신호 수신 → PREEMPT 표시, 저장 대기"; touch "$R/PREEMPT"; wait "$PID"; echo "requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_preempt USR1 TERM
PORT=$((20000 + SLURM_JOB_ID % 10000))
echo "[$(date '+%T')] torchrun × $GPUS GPU (port $PORT): $ARGS"
torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" --master_port="$PORT" experiments/p2_train_hf.py $ARGS &
PID=$!; wait "$PID"; RC=$?
if [ "$RC" -eq 0 ] && [ -f "$R/DONE" ]; then echo "[$(date '+%F %T')] p2 D1 완료 → DONE"; echo EXIT=0
elif [ "$RC" -eq 0 ]; then echo "[$(date '+%T')] 미완료 종료(선점 저장 등) → requeue"; scontrol requeue "$SLURM_JOB_ID"
elif [ "${SLURM_RESTART_COUNT:-0}" -lt "${MAX_RESTART:-3}" ]; then echo "[$(date '+%T')] 학습 실패 rc=$RC → requeue (restart ${SLURM_RESTART_COUNT:-0}/${MAX_RESTART:-3}, 마지막 checkpoint 에서 재개)"; scontrol requeue "$SLURM_JOB_ID"
else echo "[$(date '+%T')] 학습 실패 rc=$RC (restart 한도 초과)"; exit "$RC"; fi
