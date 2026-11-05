#!/usr/bin/env bash
# 2026-11-06 E15b 학습 전 점검(서버 호스트, 제출 없음): ① 시작 대칭(E13 final 에 0 초기화 슬롯 조건 인코더 — 회의 120 s 디코드가 같아야 한다)
#   ② slurm/submit-q17-e15b.sh --dry-run 이 만든 stage3_train 인자 그대로 GPU 하나에서 2 step 학습(새 합성 셋 없이 — --sim-weight 0, 셋 빌드 중) → final 저장·재로드 parity
#   ③ 그 체크포인트로 회의 120 s 디코드. 산출 VAPKT-data/tmp/e15b-check(새 폴더). bash e15b_smoke.sh <GPU> [층(기본 16)]. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e15b-check; P=$D/conda/envs/vapasr/bin/python; L=${2:-16}; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VAPASR_LOCAL_CACHE=/tmp/sa_tskim VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
[ -e $T/smoke ] && { echo "이미 있다: $T/smoke(새 폴더에만 쓴다)"; exit 1; }
X=raw/sources/experiments/2026-11-06-e15b; AB=$D/data/ami-bench-v1
echo "[$(date '+%F %T')] ① 시작 대칭(GPU $1, 층 $L)" > $T/smoke.log
CUDA_VISIBLE_DEVICES=$1 $P $X/e15b_parity.py --mode init --checkpoint /soundai/Model/VAPASR/stage3-q17-e13/final --encoder $MXC_NEMOTRON_DIR --data $AB --layer $L >> $T/smoke.log 2>&1
echo "[$(date '+%F %T')] PARITY_EXIT $?" >> $T/smoke.log
A=$(bash slurm/submit-q17-e15b.sh --dry-run --partition hpc --sim-weight 0 --layer $L 2>&1 | grep -o "ARGS_EXTRA=.*" | head -1 | sed -e 's/^ARGS_EXTRA=//' -e 's# slurm/stage3-train-apex.sbatch$##')
[ -n "$A" ] || { echo "인자를 못 읽었다" >> $T/smoke.log; exit 1; }
echo "$A" > $T/args.txt
echo "[$(date '+%F %T')] ② 학습 스모크" >> $T/smoke.log
CUDA_VISIBLE_DEVICES=$1 $P experiments/stage3_train.py --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $T/smoke ${A//%2C/,} --max-steps 2 --save-every 1000 --log-every 1 >> $T/smoke.log 2>&1
echo "[$(date '+%F %T')] SMOKE_EXIT $?" >> $T/smoke.log
echo "[$(date '+%F %T')] ③ 체크포인트 디코드" >> $T/smoke.log
CUDA_VISIBLE_DEVICES=$1 $P $X/e15b_parity.py --mode ckpt --checkpoint $T/smoke/final --encoder $MXC_NEMOTRON_DIR --data $AB >> $T/smoke.log 2>&1
echo "[$(date '+%F %T')] DECODE_EXIT $?" >> $T/smoke.log
echo "[$(date '+%F %T')] ALL_DONE" >> $T/smoke.log
