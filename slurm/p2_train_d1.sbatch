#!/bin/bash
# Phase 2 D1 실학습 — GPU 1 장 (사용자 지시 2026-09-16: 최대 GPU 1 장). 정본 output-phase2-lane-plan §9·§13.
#   제출:  sbatch --partition=apex --export=ALL,RUN=D1 slurm/p2_train_d1.sbatch
#          옵션: STEPS(기본 0 = 1 epoch), CORPORA, MAX_TOKENS(동적 길이 배치 예산; 0 이면 BS_EN/BS_KO 고정 8/8), MAX_BS(48), MIX(equal|sqrt|prop), LR(6e-5), EXTRA="--train-encoder" 등
#          동적 배치 예산은 experiments/p2_mem_probe.py 실측(GPU 메모리 80 %)으로 정한다. mono 캐시는 scripts/p2-mono-local.sh 로 미리 만든다.
#   산출물: /soundai/Model/VAPASR/p2-<RUN>/{checkpoint-N,final,run.json,results.json,tb}, mono 캐시 /soundai/users/tskim/VAPKT-data/data/phase2/mono
#   재개: 같은 명령을 다시 내면 마지막 완성 checkpoint 에서 이어간다(--resume auto). 선점 시 PreemptCallback 이 저장 후 종료 → requeue.
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2d1
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" && mkdir -p "$TMPDIR"; fi
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
RUN=${RUN:?RUN=D1 등}; ROOT=/soundai/Model/VAPASR; R=$ROOT/p2-$RUN; DATA=/soundai/users/tskim/VAPKT-data/data/phase2; INIT=${INIT:-$ROOT/hf-E2/final}
mkdir -p "$R" "$DATA/mono"; echo "[$(date '+%F %T')] p2 D1 job $SLURM_JOB_ID RUN=$RUN node=$(hostname) init=$INIT"
python experiments/p2_train_hf.py --init "$INIT" --data "$DATA" --corpora "${CORPORA:-aihub71631,aihub134-1,aihub134-2,otoSpeech,ami,notsofar,icsi}" --mono-cache "$DATA/mono" --out-dir "$R" \
  --max-steps "${STEPS:-0}" --epochs "${EPOCHS:-1}" --max-tokens "${MAX_TOKENS:-0}" --max-bs "${MAX_BS:-48}" --mix "${MIX:-equal}" --bs-en "${BS_EN:-8}" --bs-ko "${BS_KO:-8}" --lr "${LR:-6e-5}" --warmup "${WARMUP:-500}" --log-every 50 --save-every "${SAVE_EVERY:-500}" --num-workers "${WORKERS:-6}" ${EXTRA:-}
echo "[$(date '+%F %T')] p2 D1 done"; echo EXIT=0
