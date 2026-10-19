# Stage 4 체크포인트 평가 큐(mxc 호스트): bash stage4_eval_queue.sh <run> <대상> [<대상> …] — 대상 = checkpoint-N(trainer_state.json 이 생기면) | final(DONE 이 생기면).
# 대상마다 그때 가장 한가한 호스트 GPU(빈 메모리 최대, 같으면 사용률 낮은 쪽 — 다른 사용자와 같이 쓴다)를 골라 차례로 잰다:
#   1. 같은 언어 held-out 대화 번역(cst_stage4_eval, 7 코퍼스 × 창 100 × δ4·10 × 목표 언어) → results/stage4-eval-<run>-<태그>/dlg
#   2. 영↔독 합성 평가 셋(sim-dialogue-xl-eval-v1, 창 200 × δ4·10) → …/xl
#   3. TAXI 실제 독↔영 전화 대화(cst_stage4_taxi, δ4·10) → …/taxi
#   4. Stage 3 회귀: 대화 받아쓰기·귀속·<EOT>(cst_stage3_eval, 창 200 × δ4·10, 맞장구 1.0,3) → results/stage3-eval-<run>-<태그>, 오류 분해 → stage3-attr-errors-…
#   5. 단일 화자 축약 평가(stage2_regress.sh) → results/stage3-regress-<run>-<태그>
# 산출물 폴더 이름의 run 은 학습 run 이름(stage3-<run> 폴더). 아무것도 지우지 않는다.
set -u
RUN=$1; shift
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; R=/soundai/users/tskim/VAPKT-data/results; OUT=/soundai/Model/VAPASR/stage3-$RUN
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; XE=$D/sim-dialogue-xl-eval-v1; TAXI=$D/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
for t in "$@"; do
  echo "[$(date '+%F %T')] $RUN/$t 대기"
  if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 120; done; else until [ -f $OUT/$t/trainer_state.json ]; do sleep 120; done; sleep 60; fi
  tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=$OUT/$t; O=$R/stage4-eval-$RUN-$tag; mkdir -p $O
  GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1 | cut -d' ' -f1)
  export CUDA_VISIBLE_DEVICES=$GPU; echo "[$(date '+%F %T')] $RUN/$t → GPU $GPU ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU))"
  $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 --limit-windows 100 \
    --batch-size 32 --backchannel 1.0,3 --out $O/dlg > $O/dlg.log 2>&1; $P experiments/cst_stage4_eval.py summarize --out $O/dlg > $O/dlg-summary.log 2>&1
  echo "[$(date '+%F %T')] 1 같은 언어 대화 번역 끝"
  $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $XE --trans $XE/targets-sim_en-de --corpora sim_en-de --deltas 4,10 --limit-windows 200 \
    --batch-size 32 --out $O/xl > $O/xl.log 2>&1; $P experiments/cst_stage4_eval.py summarize --out $O/xl > $O/xl-summary.log 2>&1
  echo "[$(date '+%F %T')] 2 영↔독 합성 끝"
  $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi --deltas 4,10 > $O/taxi.log 2>&1
  $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi > $O/taxi-summary.log 2>&1
  echo "[$(date '+%F %T')] 3 TAXI 끝"
  E3=$R/stage3-eval-$RUN-$tag; mkdir -p $E3
  $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E3 --deltas 4,10 --limit-windows 200 --batch-size 32 \
    --verify --backchannel 1.0,3 > $E3/run.log 2>&1; $P experiments/cst_stage3_eval.py summarize --out $E3 > $E3/summary.log 2>&1
  A=$R/stage3-attr-errors-$RUN-$tag; mkdir -p $A
  $P experiments/cst_attr_errors.py --pred $E3 --data $P2 --split $P2/splits/v2.json --out $A --deltas 4,10 --workers 16 > $A/run.log 2>&1
  echo "[$(date '+%F %T')] 4 Stage 3 회귀 끝"
  bash $R/stage3-regress/stage2_regress.sh $CK $RUN-$tag $GPU > $R/stage3-regress/run-$RUN-$tag.log 2>&1
  echo "[$(date '+%F %T')] $RUN/$t 5 단일 화자 끝 — 대상 끝"
done
echo "QUEUE DONE"
