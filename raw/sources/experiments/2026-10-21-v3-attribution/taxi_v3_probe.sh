# V3 진단(mxc 호스트): bash taxi_v3_probe.sh <run> <대상> <bias…> — 세션 통째(창 90 s)·δ10 TAXI 를 bias 마다 디코드(<EOT> 기록) → cst_stage4_taxi score.
# 산출: results/taxi-v3probe-<run>-<태그>-b<bias>/(hyp·asr 는 mxc 에만). 빈 메모리 MIN_FREE_MB(기본 40000) 이상 GPU 를 기다린다. 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
R=/soundai/users/tskim/VAPKT-data/results; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
export CODE PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; X=raw/sources/experiments/2026-10-21-v3-attribution
RUN=$1; t=$2; shift 2; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=/soundai/Model/VAPASR/stage3-$RUN/$t
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
  [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-40000} ] && break; sleep 300
done
export CUDA_VISIBLE_DEVICES=$(echo $g | cut -d' ' -f1); echo "[$(date '+%F %T')] $RUN/$t → GPU $CUDA_VISIBLE_DEVICES"
for b in "$@"; do
  O=$R/taxi-v3probe-$RUN-$tag-b$b; mkdir -p $O
  $P $X/taxi_v3_probe.py --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O --deltas 10 --max-window 90 --tag-bias $b > $O/run.log 2>&1
  $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O > $O/summary.log 2>&1
  echo "[$(date '+%F %T')] b$b 끝 — $(grep -h PROBE $O/run.log | cut -c1-200) · $(grep -h 받아쓰기 $O/summary.log | grep -o "'split': {[^}]*}")"
done
echo "TAXI_V3_PROBE_DONE"
