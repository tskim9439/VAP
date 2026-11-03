#!/usr/bin/env bash
# 2026-11-03 E14 학습 전 점검(서버 호스트, 제출 없음): ① slurm/submit-q17-e14.sh --dry-run 이 만든 stage3_train 인자 그대로 stage3_train.py --dry-run(CPU — 셋마다 시퀀스 불변식:
# X 칸 배치·다시 읽기·학습 rng 경로의 놓침 흉내), ② 같은 인자로 GPU 하나에서 2 step 학습 스모크(전문가 확률 활동·놓침 마스크 forward·역전파). 산출 tmp/e14-check(새 폴더).
#   bash e14_dry.sh [GPU(비우면 ①만)] [CHECK_ITEMS(기본 32)]. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e14-check; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VAPASR_LOCAL_CACHE=/tmp/sa_tskim VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
A=$(bash slurm/submit-q17-e14.sh --dry-run --partition low_p_hpc 2>&1 | grep -o "ARGS_EXTRA=.*" | head -1 | sed -e 's/^ARGS_EXTRA=//' -e 's# slurm/stage3-train-apex.sbatch$##')
[ -n "$A" ] || { echo "인자를 못 읽었다"; exit 1; }
echo "[$(date '+%F %T')] dry-run 시작" > $T/dry.log
CUDA_VISIBLE_DEVICES= $P experiments/stage3_train.py --dry-run --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $T/dry ${A//%2C/,} --check-items ${2:-32} >> $T/dry.log 2>&1
echo "[$(date '+%F %T')] DRY_EXIT $?" >> $T/dry.log
[ -n "${1:-}" ] || exit 0
echo "[$(date '+%F %T')] 스모크 시작(GPU $1)" > $T/smoke.log
CUDA_VISIBLE_DEVICES=$1 $P experiments/stage3_train.py --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $T/smoke ${A//%2C/,} --max-steps 2 --save-every 1000 --log-every 1 --no-check-save >> $T/smoke.log 2>&1
echo "[$(date '+%F %T')] SMOKE_EXIT $?" >> $T/smoke.log
