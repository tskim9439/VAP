# 본 학습 체크포인트 평가 큐: bash ckpt_eval_queue.sh <GPU> <run> <대상> [<대상> …] — 대상 = checkpoint-N(trainer_state.json 이 생기면) | final(DONE 이 생기면).
#   대화 평가(held-out 창 200 개 × 7 코퍼스, δ4·10, 맞장구 규칙 참조 1.0,3, 직렬/배치 대조) → results/stage3-eval-<run>-<대상>
#   화자 귀속 오류 유형 분해(cst_attr_errors.py, CPU) → results/stage3-attr-errors-<run>-<대상>
#   단일 화자 축약 평가(stage2_regress.sh) → results/stage3-regress-<run>-<대상>. 아무것도 지우지 않는다.
set -u
GPU=$1; RUN=$2; shift 2
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2; R=/soundai/users/tskim/VAPKT-data/results; OUT=/soundai/Model/VAPASR/stage3-$RUN
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPU VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
for t in "$@"; do
  if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 120; done; else until [ -f $OUT/$t/trainer_state.json ]; do sleep 120; done; sleep 60; fi
  tag=${t#checkpoint-}; tag=${tag/final/final}; [ "$t" != final ] && tag=ck$tag
  O=$R/stage3-eval-$RUN-$tag; mkdir -p $O
  echo "[$(date '+%F %T')] $RUN/$t 대화 평가 시작 (GPU $GPU)"
  $P experiments/cst_stage3_eval.py run --checkpoint $OUT/$t --encoder $MXC_NEMOTRON_DIR --data $D --split $D/splits/v2.json --out $O --deltas 4,10 --limit-windows 200 --batch-size 32 --verify --backchannel 1.0,3 > $O/run.log 2>&1
  $P experiments/cst_stage3_eval.py summarize --out $O > $O/summary.log 2>&1
  A=$R/stage3-attr-errors-$RUN-$tag; mkdir -p $A
  $P experiments/cst_attr_errors.py --pred $O --data $D --split $D/splits/v2.json --out $A --deltas 4,10 --workers 16 > $A/run.log 2>&1
  echo "[$(date '+%F %T')] $RUN/$t 대화 평가·오류 분해 끝 · 단일 화자 평가 시작"
  bash $R/stage3-regress/stage2_regress.sh $OUT/$t $RUN-$tag $GPU > $R/stage3-regress/run-$RUN-$tag.log 2>&1
  echo "[$(date '+%F %T')] $RUN/$t 단일 화자 평가 끝"
done
echo "QUEUE DONE"
