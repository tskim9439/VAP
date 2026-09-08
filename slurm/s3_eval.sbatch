#!/bin/bash
# HF 모델 오프라인 평가(단일 노드 8 GPU 분산 디코드). 전체 dev 셋 또는 큰 표본으로 최종 수치를 낸다.
#   제출:  sbatch --partition=hpc --export=ALL,CKPT=/soundai/Model/VAPASR/hf-C2/final,OUT=/soundai/Model/VAPASR/hf-C2,TAG=fulldev,STREAMS=100000,UTTS=100000,DELAY=2,BIAS=0 slurm/s3_eval.sbatch
#          STREAMS/UTTS: 셋당 스트림/발화 상한(100000 = 전체). SEED 표본 seed(기본 7). 결과 OUT/eval/<TAG>-<step>.json (+ TensorBoard)
#   로그:   /soundai/users/tskim/VAPKT/logs/train/SA_SFT_FullDuplexStage5-eval-<jobid>.out|.err
#SBATCH --job-name=SA_SFT_FullDuplexStage5-eval
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
CKPT=${CKPT:?HF 모델 디렉토리}; OUT=${OUT:-$(dirname "$CKPT")}; TAG=${TAG:-fulldev}; STREAMS=${STREAMS:-100000}; UTTS=${UTTS:-100000}; DELAY=${DELAY:-2}; BIAS=${BIAS:-0}; SEED=${SEED:-7}
echo "[$(date '+%F %T')] eval job $SLURM_JOB_ID node=$SLURM_JOB_NODELIST ckpt=$CKPT tag=$TAG streams=$STREAMS utts=$UTTS delay=$DELAY bias=$BIAS"
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export MXC_DATA_MANIFEST_DIR=${MAN_IN:-$MXC_DATA_MANIFEST_DIR} VAPASR_ALIGN_ROOT=${ALIGN_ROOT:-${MAN_IN:-$MXC_DATA_MANIFEST_DIR}/align-asr-tn-v1}
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 PYTHONFAULTHANDLER=1
case "$(which python)" in "$MXC_CONDA_DIR"/envs/vapasr/*) ;; *) echo "!! conda env 아님"; exit 1;; esac
torchrun --nproc_per_node=8 --master_port=$((20000 + SLURM_JOB_ID % 10000)) experiments/s3_train_hf.py --eval-only --init "$CKPT" --out-dir "$OUT" --sentinel-stream "$STREAMS" --sentinel-utt "$UTTS" --eval-seed "$SEED" --eval-tag "$TAG" --eval-delay "$DELAY" --eval-bias "$BIAS" --num-workers 0
RC=$?; echo "[$(date '+%T')] eval rc=$RC"; [ "$RC" -eq 0 ] || { [ "${SLURM_RESTART_COUNT:-0}" -lt 3 ] && scontrol requeue "$SLURM_JOB_ID"; }; exit $RC
