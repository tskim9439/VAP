# Stage 2 단일 화자 퇴행 축약 평가: bash stage2_regress.sh <체크포인트> <태그> <GPU>. 세트마다 고른 간격 100 발화, δ4·10, 직렬/배치 대조.
# 산출 /soundai/users/tskim/VAPKT-data/results/stage3-regress-<태그>/<세트> — 지우지 않는다.
set -u
CK=$1; TAG=$2; GPU=$3
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
EV=/soundai/users/tskim/VAPKT-data/data/evaluation; A=$EV/ast-fleurs-covost2-v1; O=/soundai/users/tskim/VAPKT-data/results/stage3-regress-$TAG
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
for x in "fleurs $A/manifest-max30s-fleurs.jsonl" "vpaa-kspon $EV/single-turn-vpaa-kspon-v1/manifest.jsonl" "yodas-de $EV/single-turn-yodas-de-dev-v1/manifest.jsonl" "libri-other $EV/single-turn-librispeech-v1/manifest-test-other.jsonl"; do
  set -- $x; mkdir -p $O/$1
  $P experiments/eval_single_turn_asr.py run --manifest $2 --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --out $O/$1 --deltas 4 10 --limit 100 --batch-size 64 --verify > $O/$1/run.log 2>&1
  $P experiments/eval_single_turn_asr.py summarize --out $O/$1 > $O/$1/summary.log 2>&1
  echo "[$(date +%T)] $1 done rc=$?"
done
echo "EXIT"
