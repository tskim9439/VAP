# V2 TAXI 하락 원인 가르기(mxc 호스트): bash taxi_v2_probe.sh [<run> [<대상>]] — 기본 v2-p1 checkpoint-2000.
# taxi_v2_probe.py 를 탭 켬(tap)·끔(notap)으로 차례로(δ10, 창 40 s = 기본 TAXI 평가와 같은 창) → cst_stage4_taxi score. 빈 메모리 MIN_FREE_MB(기본 40000) 이상 GPU 를 기다린다.
# 산출: results/taxi-probe-<run>-<태그>-{tap,notap}/(hyp·asr 는 mxc 에만, probe.json·summary.log). 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
R=/soundai/users/tskim/VAPKT-data/results; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
export CODE PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; X=raw/sources/experiments/2026-10-20-v2-multispeaker
RUN=${1:-v2-p1}; t=${2:-checkpoint-2000}; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=/soundai/Model/VAPASR/stage3-$RUN/$t
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
  [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-40000} ] && break; sleep 300
done
export CUDA_VISIBLE_DEVICES=$(echo $g | cut -d' ' -f1); echo "[$(date '+%F %T')] $RUN/$t → GPU $CUDA_VISIBLE_DEVICES"
for v in tap notap; do
  O=$R/taxi-probe-$RUN-$tag-$v; mkdir -p $O; NT=""; [ $v = notap ] && NT=--no-tap
  $P $X/taxi_v2_probe.py --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O --deltas 10 $NT > $O/run.log 2>&1
  $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O > $O/summary.log 2>&1
  echo "[$(date '+%F %T')] $v 끝 — $(grep -h PROBE $O/run.log | cut -c1-300) · $(grep -h 받아쓰기 $O/summary.log | cut -c1-200)"
done
echo "TAXI_PROBE_DONE"
