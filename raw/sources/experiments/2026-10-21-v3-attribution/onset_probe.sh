# V3 온셋 분류(mxc 호스트 GPU): bash onset_probe.sh <run> <대상> [<run> <대상> …] — TAXI 세션 통째 δ10, bias 없음(고정 설정)으로 디코드하며
# taxi_v3_probe.py --classify-onsets 로 교대마다 OK·RESUME·REASSERT·SILENT·ABSORBED(앞 턴 상태·<EOT> 유무·새 화자 언어별) 집계 → results/taxi-v3onset-<run>-<태그>/onset-classes-d10.json.
# 값은 고르지 않는다(진단). 빈 메모리 MIN_FREE_MB(기본 40000) GPU 를 기다린다. 있는 산출 폴더는 건너뛴다. 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export CODE PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
X=raw/sources/experiments/2026-10-21-v3-attribution
while [ $# -ge 2 ]; do
  RUN=$1; t=$2; shift 2; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; O=$R/taxi-v3onset-$RUN-$tag
  [ -e $O ] && { echo "이미 있다: $O"; continue; }
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
    [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-40000} ] && break; sleep 120
  done
  export CUDA_VISIBLE_DEVICES=$(echo $g | cut -d' ' -f1); mkdir -p $O; echo "[$(date '+%F %T')] $RUN/$t → GPU $CUDA_VISIBLE_DEVICES"
  $P $X/taxi_v3_probe.py --checkpoint ${MODEL_ROOT:-/soundai/Model/VAPASR}/stage3-$RUN/$t --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O --deltas 10 --max-window 90 --classify-onsets > $O/run.log 2>&1
  echo "[$(date '+%F %T')] $RUN/$t 끝 — $(grep -h ONSET $O/run.log | cut -c1-300)"
done
echo "ONSET_PROBE_DONE"
