# V3 평가 큐(mxc 호스트): bash v3_eval_queue.sh <run> <대상> [<대상> …] — 대상 = checkpoint-N(trainer_state.json 이 생기면) | final(DONE 이 생기면).
# 기존 평가 큐(stage4_eval_queue.sh — 대화 번역·합성·TAXI 창 40 s·Stage 3 회귀·단일 화자·CallHome)에 더하는 V3 단계. 대상마다 차례로:
#   1. 비-TAXI 전화 dev(xl-tel-dev 160)와 짝 비교 셋(xl-eval-v1 160) 디코드·턴 결정 지표(v3_dev_decode.sh, bias 없음) — 체크포인트 선택은 여기서만
#   2. TAXI 세션 통째(--max-window 90, δ4·10, bias 없음 = 고정 설정)·score --turn-metrics → results/stage4-eval-<run>-<태그>/taxi90 — 보고 전용(값을 고르지 않는다)
# 빈 메모리 MIN_FREE_MB(기본 40000) GPU 를 기다린다. 아무것도 지우지 않는다. MODEL_ROOT(기본 /soundai/Model/VAPASR)로 stage3-<run> 의 위치를 바꾼다.
set -u
RUN=$1; shift
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; OUT=${MODEL_ROOT:-/soundai/Model/VAPASR}/stage3-$RUN
X=raw/sources/experiments/2026-10-21-v3-attribution; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
for t in "$@"; do
  echo "[$(date '+%F %T')] $RUN/$t 대기"
  if [ "$t" = final ] && [ ! -f $OUT/DONE ]; then until [ -f $OUT/DONE ]; do sleep 120; done; fi
  if [ "$t" != final ]; then until [ -f $OUT/$t/trainer_state.json ]; do sleep 120; done; sleep 60; fi
  tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag
  bash $X/v3_dev_decode.sh $RUN $t xl-tel-dev sessions; bash $X/v3_dev_decode.sh $RUN $t xl-eval-v1 sessions
  O=$R/stage4-eval-$RUN-$tag; mkdir -p $O
  if [ ! -e $O/taxi90 ]; then
    while true; do
      g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
      [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-40000} ] && break; sleep 120
    done
    export CUDA_VISIBLE_DEVICES=$(echo $g | cut -d' ' -f1); echo "[$(date '+%F %T')] $RUN/$t TAXI 통째 → GPU $CUDA_VISIBLE_DEVICES"
    $P experiments/cst_stage4_taxi.py run --checkpoint $OUT/$t --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi90 --deltas 4,10 --max-window 90 --batch-size 8 > $O/taxi90.log 2>&1
    $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi90 --turn-metrics > $O/taxi90-summary.log 2>&1
    echo "[$(date '+%F %T')] $RUN/$t TAXI 통째 끝 — $(grep -h "δ10 받아쓰기" $O/taxi90-summary.log | grep -o "'split': {[^}]*}")"
  else echo "[$(date '+%F %T')] $O/taxi90 이미 있다 — 건너뜀"; fi
  echo "[$(date '+%F %T')] $RUN/$t V3 평가 끝"
done
echo "V3_QUEUE_DONE"
