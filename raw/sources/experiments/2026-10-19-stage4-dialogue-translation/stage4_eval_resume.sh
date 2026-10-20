# Stage 4 평가 큐에서 OOM 으로 끊긴 단계 이어 돌리기(mxc 호스트): bash stage4_eval_resume.sh <run> <대상> [<대상> …] — 대상 = checkpoint-N | final.
# 2026-10-20 st4-m2a·m2b final: 호스트 GPU 를 다른 작업과 나눠 쓰다 1(대화 번역)·2(영↔독 합성)·4(Stage 3 회귀) 단계가 CUDA OOM 으로 중간에 끝났다(TAXI 는 완료).
# 평가 큐(stage4_eval_queue.sh)가 하나도 안 남을 때까지 기다린 뒤, 대상마다 빈 메모리 MIN_FREE_MB 이상인 GPU 가 날 때까지 기다려
# 같은 --out 에 이어 쓴다(평가 스크립트가 끝난 창을 건너뛴다 — 재개 설정 지문은 batch 크기를 넣지 않는다). batch 16(큐는 32). 아무것도 지우지 않는다.
set -u
RUN=$1; shift
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; R=/soundai/users/tskim/VAPKT-data/results; OUT=/soundai/Model/VAPASR/stage3-$RUN
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; XE=$D/sim-dialogue-xl-eval-v1
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
LS=""; [ "${TRANS_LEAD_SPACE:-0}" = 1 ] && LS=--trans-lead-space
MIN_FREE_MB=${MIN_FREE_MB:-40000}
echo "[$(date '+%F %T')] $RUN 평가 큐 끝 대기"
while pgrep -f stage4_eval_queue.sh > /dev/null; do sleep 120; done
pick_gpu() {                                                                                   # 빈 메모리 최대(같으면 사용률 낮은 쪽), MIN_FREE_MB 미만이면 기다린다
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
    [ "$(echo $g | cut -d' ' -f2)" -ge $MIN_FREE_MB ] && { echo $g | cut -d' ' -f1; return; }; sleep 300
  done
}
for t in "$@"; do
  tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=$OUT/$t; O=$R/stage4-eval-$RUN-$tag
  export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "[$(date '+%F %T')] $RUN/$t 1 → GPU $CUDA_VISIBLE_DEVICES"
  $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 --limit-windows 100 $LS \
    --batch-size 16 --backchannel 1.0,3 --out $O/dlg > $O/dlg-resume.log 2>&1; $P experiments/cst_stage4_eval.py summarize --out $O/dlg > $O/dlg-summary.log 2>&1
  echo "[$(date '+%F %T')] 1 같은 언어 대화 번역 끝($(grep -c . $O/dlg/predictions-rank0.jsonl) 행)"
  export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "[$(date '+%F %T')] $RUN/$t 2 → GPU $CUDA_VISIBLE_DEVICES"
  $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $XE --trans $XE/targets-sim_en-de --corpora sim_en-de --deltas 4,10 --limit-windows 200 $LS \
    --batch-size 16 --out $O/xl > $O/xl-resume.log 2>&1; $P experiments/cst_stage4_eval.py summarize --out $O/xl > $O/xl-summary.log 2>&1
  echo "[$(date '+%F %T')] 2 영↔독 합성 끝($(grep -c . $O/xl/predictions-rank0.jsonl) 행)"
  export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "[$(date '+%F %T')] $RUN/$t 4 → GPU $CUDA_VISIBLE_DEVICES"
  E3=$R/stage3-eval-$RUN-$tag
  $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E3 --deltas 4,10 --limit-windows 200 --batch-size 16 \
    --verify --backchannel 1.0,3 > $E3/run-resume.log 2>&1; $P experiments/cst_stage3_eval.py summarize --out $E3 > $E3/summary.log 2>&1
  A=$R/stage3-attr-errors-$RUN-$tag
  $P experiments/cst_attr_errors.py --pred $E3 --data $P2 --split $P2/splits/v2.json --out $A --deltas 4,10 --workers 16 > $A/run-resume.log 2>&1
  echo "[$(date '+%F %T')] 4 Stage 3 회귀 끝($(grep -c . $E3/predictions-rank0.jsonl) 행) — $RUN/$t 이어 돌리기 끝"
done
