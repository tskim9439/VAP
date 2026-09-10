#!/bin/bash
# 평가 묶음(1 노드 8 GPU): <RUN> 의 select(14000·final, δ2/δ4, 71631-dev 포함) + C2 재평가 + 타이밍 스윕(experiments/s3_d2_eval.sh) 뒤에
#   test 평가(experiments/s3_test_eval.sh: test-clean·test-other·kspon-eval 전체 + 71631-dev 1,000, δ2/4) 를 <RUN>/final 과 C2/final 에 대해 실행.
#   제출:  sbatch --partition=hpc --export=ALL,RUN=E2 slurm/s3_eval_suite.sbatch      (로그인 노드 GPU 가 다른 작업으로 찼을 때 대안)
#   결과:  /soundai/Model/VAPASR/hf-<RUN>/eval/{select-*,select-d4-*,sweep-*,test-d*}.json, hf-C2/eval/{selectx-*,test-d*}.json
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
RUN=${RUN:?RUN=E2 등}; ROOT=/soundai/Model/VAPASR; R=$ROOT/hf-$RUN
echo "[$(date '+%F %T')] eval suite job $SLURM_JOB_ID RUN=$RUN node=$(hostname)"
[ -f "$R/DONE" ] || { echo "!! $R/DONE 없음 — 학습이 끝난 뒤 제출"; exit 1; }
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" VAPASR_STAGE_DIR="$(dirname "$VAPASR_LOCAL_ENV")/vapasr-stage" && mkdir -p "$TMPDIR" "$VAPASR_STAGE_DIR"; fi
export MXC_CKPT_EXP_DIR=$ROOT MXC_DATA_MANIFEST_DIR=${MAN_IN:-$MXC_DATA_MANIFEST_DIR} VAPASR_ALIGN_ROOT=${ALIGN_ROOT:-${MAN_IN:-$MXC_DATA_MANIFEST_DIR}/align-asr-tn-v1}
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True GPUS=0,1,2,3,4,5,6,7 R
echo "manifest ← $MXC_DATA_MANIFEST_DIR | python $(which python)"
bash experiments/s3_d2_eval.sh
[ "${SKIP_TEST:-0}" = 1 ] || { bash experiments/s3_test_eval.sh "$R/final"; [ -f "$ROOT/hf-C2/eval/test-d4-0.json" ] || bash experiments/s3_test_eval.sh "$ROOT/hf-C2/final"; }
echo "[$(date '+%F %T')] eval suite done"
