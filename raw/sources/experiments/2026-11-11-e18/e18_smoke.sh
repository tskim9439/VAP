#!/usr/bin/env bash
# 2026-11-11 E18 학습 전 점검(서버 호스트, 제출 없음): slurm/submit-q17-e18.sh --dry-run 이 만든 stage3_train 인자 그대로 GPU 하나에서 2 step → final 저장·재로드 parity.
#   인코더 전체를 풀었을 때 도는지, GPU 메모리가 얼마인지 본다. bash e18_smoke.sh <GPU> [배치 토큰 상한(기본 제출 값 그대로)]. 새 폴더에만 쓴다. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e18-check/$(date +%m%d-%H%M%S); P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VAPASR_LOCAL_CACHE=/tmp/sa_tskim VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
A=$(bash slurm/submit-q17-e18.sh --dry-run --partition low_p_hpc 2>&1 | grep -o "ARGS_EXTRA=.*" | head -1 | sed -e 's/^ARGS_EXTRA=//' -e 's# slurm/stage3-train-apex.sbatch$##')
[ -n "$A" ] || { echo "인자를 못 읽었다"; exit 1; }
echo "$A" > $T/args.txt; X=""; [ -n "${2:-}" ] && X="--batch-max-tokens $2"
echo "[$(date '+%F %T')] 스모크 시작(GPU $1 ${X})" > $T/smoke.log
CUDA_VISIBLE_DEVICES=$1 $P experiments/stage3_train.py --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $T/smoke ${A//%2C/,} --max-steps 2 --save-every 1000 --log-every 1 $X >> $T/smoke.log 2>&1 &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F, -v p=$PID '$1 == p {print $2 + 0}'); [ -n "$m" ] && [ "$m" -gt "$PEAK" ] && PEAK=$m
  sleep 5
done
wait $PID; echo "[$(date '+%F %T')] SMOKE_EXIT $? · GPU 메모리 최대 ${PEAK} MiB" >> $T/smoke.log
echo "[$(date '+%F %T')] ALL_DONE $T" >> $T/smoke.log
