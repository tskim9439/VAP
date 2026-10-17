# 화자 헤드 안내 디코딩 평가: bash guide_eval.sh <GPU> <체크포인트> <출력 이름> <corpora> <deltas> <guide 'on,margin[,span]'> [hold] — 아무것도 지우지 않는다.
set -u
GPU=$1; CK=$2; NAME=$3; CORP=$4; DL=$5; G=$6; HOLD=${7:-}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2; R=/soundai/users/tskim/VAPKT-data/results; O=$R/stage3-eval-$NAME; mkdir -p $O
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
echo "[$(date '+%F %T')] $NAME 시작 (GPU $GPU, guide $G ${HOLD:+hold})"
$P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $D --split $D/splits/v2.json --out $O --corpora $CORP --deltas $DL --limit-windows 200 --batch-size 32 --backchannel 1.0,3 --spk-guide $G ${HOLD:+--spk-guide-hold} ${NOINS:+--spk-guide-no-insert} > $O/run.log 2>&1
$P experiments/cst_stage3_eval.py summarize --out $O > $O/summary.log 2>&1
echo "[$(date '+%F %T')] $NAME 끝"
