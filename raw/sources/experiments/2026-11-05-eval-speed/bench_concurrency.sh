#!/usr/bin/env bash
# 2026-11-05 평가 속도(사용자 "평가는 왜 5시간이나 걸리는 거지? 더 빠르게 평가를 수행 못하나?" → "3번과 4번을 진행해줘"): 회의 통째 디코드를 한 GPU 에서
# 여러 프로세스로 동시에 돌리면 빨라지는지 잰다(서버 호스트, 제출 없음). E13 final · ami-ihmmix 가장 짧은 회의 하나 · δ10 · 문맥 30 s(평가와 같은 인자).
#   P=1 하나를 돈 뒤 P=N(기본 4) 개를 같은 GPU 에서 동시에 돌린다. 행 로그의 "디코드 X s" 를 견준다. 산출 VAPKT-data/tmp/evalspeed(새 폴더).
#   bash bench_concurrency.sh <GPU> [N]. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/evalspeed; P=$D/conda/envs/vapasr/bin/python; N=${2:-4}; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
run() { CUDA_VISIBLE_DEVICES=$1 $P experiments/cst_stage3_eval.py run --checkpoint /soundai/Model/VAPASR/stage3-q17-e13/final --encoder $MXC_NEMOTRON_DIR --data $D/data/ami-bench-v1 \
  --corpora ami-ihmmix --limit-windows 1 --deltas 10 --backchannel 1.0,3 --max-speakers 4 --window full --context-s 30 --declare 4 --rank 0 --world-size 1 --out $2 > $2.log 2>&1; }
L=$T/bench.log; echo "[$(date '+%F %T')] P=1 시작(GPU $1)" >> $L
[ -e $T/p1 ] || run $1 $T/p1
echo "[$(date '+%F %T')] P=1 끝 — $(grep -aE '디코드 [0-9.]+ s' $T/p1.log | tail -1 | cut -c1-200)" >> $L
echo "[$(date '+%F %T')] P=$N 시작" >> $L
for i in $(seq 1 $N); do [ -e $T/p$N-$i ] || run $1 $T/p$N-$i & sleep 2; done; wait
for i in $(seq 1 $N); do echo "[$(date '+%F %T')] P=$N #$i — $(grep -aE '디코드 [0-9.]+ s' $T/p$N-$i.log | tail -1 | cut -c1-200)" >> $L; done
echo "[$(date '+%F %T')] ALL_DONE" >> $L
