# V3 dev 디코드(mxc 호스트 GPU): bash v3_dev_decode.sh <run> <대상> <셋: xl-tel-dev|xl-eval-v1> <sessions 파일 이름> [<switch-bias 'eot=…,res=…,wi=…'>]
# cst_stage4_taxi run(번역 prefix, 세션 통째 --max-window 90, δ10) → score --turn-metrics. 산출 results/v3dev-<run>-<태그>-<셋>-<sessions 이름>[-sb<정책>]. 빈 메모리 MIN_FREE_MB(기본 40000) GPU 를 기다린다.
# TAXI 는 쓰지 않는다. 있는 산출 폴더는 건드리지 않고 멈춘다. 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; DEV=/soundai/users/tskim/VAPKT-data/data/v3-dev
RUN=$1; t=$2; SET=$3; SF=$4; SB=${5:-}; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=/soundai/Model/VAPASR/stage3-$RUN/$t
SESS=$DEV/$SET/$SF.jsonl; [ -f $SESS ] || { echo "없음: $SESS"; exit 2; }
O=$R/v3dev-$RUN-$tag-$SET-$SF${SB:+-sb$(echo $SB | tr -d '=,.')}; [ -e $O ] && { echo "이미 있다: $O"; exit 1; }
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
  [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-40000} ] && break; sleep 120
done
export CUDA_VISIBLE_DEVICES=$(echo $g | cut -d' ' -f1); mkdir -p $O; echo "[$(date '+%F %T')] $RUN/$t $SET/$SF ${SB:-bias 없음} → GPU $CUDA_VISIBLE_DEVICES"
$P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $SESS --out $O --deltas 10 --max-window 90 --batch-size 8 ${SB:+--switch-bias $SB} > $O/run.log 2>&1
$P experiments/cst_stage4_taxi.py score --sessions $SESS --out $O --turn-metrics > $O/summary.log 2>&1
echo "[$(date '+%F %T')] 끝 — $(grep -o "'split': {[^}]*}" $O/summary.log | head -1) · $(grep -h -o 'toggle recall[^·]*' $O/summary.log | head -1)"
