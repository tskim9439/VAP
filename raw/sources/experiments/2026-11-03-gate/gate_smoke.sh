#!/usr/bin/env bash
# 2026-11-03 E13 게이트 진단 스모크(호스트 GPU 하나): --gate-thresh·--gate-lags 경로가 끝까지 도는지 — AMI SDM 창 8 개(기준 · gt0.3 · gl0)와 회의 1 개(gt0.3, --verify).
#   bash gate_smoke.sh <GPU>. 산출 tmp/e13-gate/smoke-*(새 폴더). 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e13-gate; P=$D/conda/envs/vapasr/bin/python; CK=/soundai/Model/VAPASR/stage3-q17-e13/final; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=$1
W="--checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $D/data/ami-bench-v1 --corpora ami-sdm --deltas 10 --backchannel 1.0,3 --max-speakers 4 --declare 4"
for v in "base:" "gt03:--gate-thresh 0.3" "gl0:--gate-lags 0"; do
  n=${v%%:*}; o=${v#*:}
  $P experiments/cst_stage3_eval.py run $W --limit-windows 8 --batch-size 8 --window 20,90 --window-draws 1 $o --out $T/smoke-win-$n > $T/smoke-win-$n.log 2>&1
  $P experiments/cst_stage3_eval.py summarize --out $T/smoke-win-$n >> $T/smoke-win-$n.log 2>&1
  echo "[$(date +%T)] win $n exit $?" >> $T/smoke-done.txt
done
$P experiments/cst_stage3_eval.py run $W --limit-windows 1 --window full --context-s 30 --verify --gate-thresh 0.3 --out $T/smoke-full-gt03 > $T/smoke-full-gt03.log 2>&1
$P experiments/cst_stage3_eval.py summarize --out $T/smoke-full-gt03 >> $T/smoke-full-gt03.log 2>&1
echo "GATE_SMOKE_DONE $(date +%T)" >> $T/smoke-done.txt
