#!/usr/bin/env bash
# 2026-11-02 E13 학습 전 데이터 점검(서버 호스트 CPU, GPU·제출 없음): slurm/submit-q17-e13.sh --dry-run 이 만든 stage3_train 인자 그대로 stage3_train.py --dry-run 을 돈다
# — 실제 데이터로 셋마다 시퀀스 불변식(게이트 배치·번역 칸·학습 rng 경로)을 검사하고 예를 찍는다. 산출 tmp/e13-check/dry(새 폴더). 아무것도 지우지 않는다.
#   bash e13_dry.sh [CHECK_ITEMS(기본 32)]
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e13-check; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
export VAPASR_LOCAL_CACHE=/tmp/sa_tskim VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
A=$(bash slurm/submit-q17-e13.sh --dry-run --partition low_p_hpc 2>&1 | grep -o "ARGS_EXTRA=.*" | head -1 | sed -e 's/^ARGS_EXTRA=//' -e 's# slurm/stage3-train-apex.sbatch$##')
[ -n "$A" ] || { echo "인자를 못 읽었다"; exit 1; }
echo "[$(date '+%F %T')] dry-run 시작" > $T/dry.log
$P experiments/stage3_train.py --dry-run --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $T/dry ${A//%2C/,} --check-items ${1:-32} >> $T/dry.log 2>&1
echo "[$(date '+%F %T')] DRY_EXIT $?" >> $T/dry.log
