# Stage 4 체크포인트 평가 큐(mxc 호스트): bash stage4_eval_queue.sh <run> <대상> [<대상> …] — 대상 = checkpoint-N(trainer_state.json 이 생기면) | final(DONE 이 생기면).
# 대상마다 그때 가장 한가한 호스트 GPU(빈 메모리 최대, 같으면 사용률 낮은 쪽 — 다른 사용자와 같이 쓴다; 빈 메모리 MIN_FREE_MB 이상이 될 때까지 기다린다)를 골라 차례로 잰다:
#   1. 같은 언어 held-out 대화 번역(cst_stage4_eval, 7 코퍼스 × 창 100 × δ4·10 × 목표 언어) → results/stage4-eval-<run>-<태그>/dlg
#   2. 영↔독 합성 평가 셋(sim-dialogue-xl-eval-v1, 창 200 × δ4·10) → …/xl
#   3. TAXI 실제 독↔영 전화 대화(cst_stage4_taxi, δ4·10) → …/taxi
#   4. Stage 3 회귀: 대화 받아쓰기·귀속·<EOT>(cst_stage3_eval, 창 200 × δ4·10, 맞장구 1.0,3) → results/stage3-eval-<run>-<태그>, 오류 분해 → stage3-attr-errors-…
#   5. 단일 화자 축약 평가(stage2_regress.sh) → results/stage3-regress-<run>-<태그>
#   6. CallHome 독일어 held-out 발화(전화, single-turn-callhome-de-v1 이 있으면) → results/stage3-regress-<run>-<태그>/callhome-de
# 산출물 폴더 이름의 run 은 학습 run 이름(stage3-<run> 폴더). 아무것도 지우지 않는다.
# GPU 는 단계마다 새로 고른다(2026-10-21 V3 평가에서 한 GPU 로 4 시간을 돌다 다른 사용자 작업과 겹쳐 OOM 5 건). 1·2·4 단계는 CUDA OOM 이면 GPU 를 다시 골라
#   같은 --out 에서 한 번 이어 하고(예측을 이어 쓴다), 3 단계(TAXI)는 처음부터 한 번 다시 한다.
# TRANS_LEAD_SPACE=1: 번역 참조를 행마다 앞 공백으로(--trans-lead-space 로 학습한 run, st4-m1 부터). 1·2 단계에만 쓴다(TAXI 참조는 사람 번역).
set -u
RUN=$1; shift
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; R=/soundai/users/tskim/VAPKT-data/results; OUT=/soundai/Model/VAPASR/stage3-$RUN
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; XE=$D/sim-dialogue-xl-eval-v1; TAXI=$D/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
LS=""; [ "${TRANS_LEAD_SPACE:-0}" = 1 ] && LS=--trans-lead-space
pick() {                                                                                        # 빈 메모리 MIN_FREE_MB(기본 40000) 이상 GPU 를 기다린다(2026-10-20 m2 final 이 다른 작업과 나눠 쓰다 OOM)
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
    [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-40000} ] && break; sleep 300
  done
  GPU=$(echo $g | cut -d' ' -f1); export CUDA_VISIBLE_DEVICES=$GPU
  echo "[$(date '+%F %T')] $RUN/$t 단계 $1 → GPU $GPU ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU))"
}
oom() { grep -q OutOfMemoryError "$1" 2>/dev/null; }
for t in "$@"; do
  echo "[$(date '+%F %T')] $RUN/$t 대기"
  if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 120; done; else until [ -f $OUT/$t/trainer_state.json ]; do sleep 120; done; sleep 60; fi
  tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=$OUT/$t; O=$R/stage4-eval-$RUN-$tag; mkdir -p $O
  s1() { $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 --limit-windows 100 $LS \
    --batch-size 32 --backchannel 1.0,3 --out $O/dlg; }
  s2() { $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $XE --trans $XE/targets-sim_en-de --corpora sim_en-de --deltas 4,10 --limit-windows 200 $LS \
    --batch-size 32 --out $O/xl; }
  s3() { $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi --deltas 4,10; }
  s4() { $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E3 --deltas 4,10 --limit-windows 200 --batch-size 32 \
    --verify --backchannel 1.0,3; }
  pick 1; s1 > $O/dlg.log 2>&1; oom $O/dlg.log && { pick "1 OOM 재시도"; s1 >> $O/dlg.log 2>&1; }
  $P experiments/cst_stage4_eval.py summarize --out $O/dlg > $O/dlg-summary.log 2>&1
  echo "[$(date '+%F %T')] 1 같은 언어 대화 번역 끝"
  pick 2; s2 > $O/xl.log 2>&1; oom $O/xl.log && { pick "2 OOM 재시도"; s2 >> $O/xl.log 2>&1; }
  $P experiments/cst_stage4_eval.py summarize --out $O/xl > $O/xl-summary.log 2>&1
  echo "[$(date '+%F %T')] 2 영↔독 합성 끝"
  pick 3; s3 > $O/taxi.log 2>&1; oom $O/taxi.log && { pick "3 OOM 재시도"; s3 >> $O/taxi.log 2>&1; }
  $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi > $O/taxi-summary.log 2>&1
  echo "[$(date '+%F %T')] 3 TAXI 끝"
  E3=$R/stage3-eval-$RUN-$tag; mkdir -p $E3
  pick 4; s4 > $E3/run.log 2>&1; oom $E3/run.log && { pick "4 OOM 재시도"; s4 >> $E3/run.log 2>&1; }
  $P experiments/cst_stage3_eval.py summarize --out $E3 > $E3/summary.log 2>&1
  A=$R/stage3-attr-errors-$RUN-$tag; mkdir -p $A
  $P experiments/cst_attr_errors.py --pred $E3 --data $P2 --split $P2/splits/v2.json --out $A --deltas 4,10 --workers 16 > $A/run.log 2>&1
  echo "[$(date '+%F %T')] 4 Stage 3 회귀 끝"
  pick 5; bash $R/stage3-regress/stage2_regress.sh $CK $RUN-$tag $GPU > $R/stage3-regress/run-$RUN-$tag.log 2>&1
  CH=/soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-callhome-de-v1/manifest.jsonl                # 전화 독일어 held-out(있으면) — 6 단계
  if [ -f $CH ]; then O6=$R/stage3-regress-$RUN-$tag/callhome-de; mkdir -p $O6; pick 6
    $P experiments/eval_single_turn_asr.py run --manifest $CH --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --out $O6 --deltas 4 10 --batch-size 64 > $O6/run.log 2>&1
    $P experiments/eval_single_turn_asr.py summarize --out $O6 > $O6/summary.log 2>&1; echo "[$(date '+%F %T')] $RUN/$t 6 CallHome 독일어 끝"; fi
  echo "[$(date '+%F %T')] $RUN/$t 5 단일 화자 끝 — 대상 끝"
done
echo "QUEUE DONE"
