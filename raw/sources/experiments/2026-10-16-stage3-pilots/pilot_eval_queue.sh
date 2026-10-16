# 파일럿 평가 큐: bash pilot_eval_queue.sh <GPU> <run> [<run> …] — run 마다 /soundai/Model/VAPASR/stage3-<run>/DONE 을 기다렸다가
#   (1) Stage 3 대화 평가(held-out 창 200 개 × 7 코퍼스, δ4·10, 직렬/배치 대조) → results/stage3-eval-<run>-20261016
#   (2) Stage 2 단일 화자 축약 평가(stage2_regress.sh) → results/stage3-regress-<run>
# 아무것도 지우지 않는다.
set -u
GPU=$1; shift
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2; R=/soundai/users/tskim/VAPKT-data/results
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
for run in "$@"; do
  CK=/soundai/Model/VAPASR/stage3-$run
  until [ -f $CK/DONE ]; do sleep 120; done
  O=$R/stage3-eval-$run-20261016; mkdir -p $O
  echo "[$(date '+%F %T')] $run 대화 평가 시작 (GPU $GPU)"
  $P experiments/cst_stage3_eval.py run --checkpoint $CK/final --encoder $MXC_NEMOTRON_DIR --data $D --split $D/splits/v2.json --out $O --deltas 4,10 --limit-windows 200 --batch-size 32 --verify > $O/run.log 2>&1
  $P experiments/cst_stage3_eval.py summarize --out $O > $O/summary.log 2>&1
  echo "[$(date '+%F %T')] $run 대화 평가 끝 rc=$? · 단일 화자 평가 시작"
  bash $R/stage3-regress/stage2_regress.sh $CK/final $run $GPU > $R/stage3-regress/run-$run.log 2>&1
  echo "[$(date '+%F %T')] $run 단일 화자 평가 끝"
done
echo "QUEUE DONE"
