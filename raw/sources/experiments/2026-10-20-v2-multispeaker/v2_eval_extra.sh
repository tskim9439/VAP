# V2 추가 평가(mxc 호스트): bash v2_eval_extra.sh <run> <대상> [<대상> …] — 대상 = checkpoint-N | final. 기존 평가 큐(stage4_eval_queue.sh)에 더하는 셋.
# V1(2 슬롯) 체크포인트도 같은 창으로 잰다(모델 슬롯 수로 디코드·채점 — cst_stage3_eval). 대상마다 빈 메모리 MIN_FREE_MB(기본 40000) 이상 GPU 를 기다린다.
#   7. TAXI 세션 통째(--max-window 90, 최대 72 s 세션을 나누지 않는다)                 → results/stage4-eval-<run>-<태그>/taxi90
#   8. 긴 창 held-out 대화(2 인 7 코퍼스, 60–90 s 이어 덮기 draw 0, 코퍼스마다 100)    → results/stage3-eval-long-<run>-<태그> + 오류 분해
#   9. 회의 held-out(split v4 — AMI 공식 eval·dev + ES2014, NOTSOFAR 공식 dev·eval, ICSI 6; ≤ 4 화자, 20–90 s, 4 슬롯 선언, 코퍼스마다 150) → results/stage3-eval-meet-<run>-<태그> + 오류 분해
#  10. 3–4 인 합성 held-out(sim-dialogue-multi-eval-v1 한·영·독, 같은 창 규칙)         → results/stage3-eval-msim-<run>-<태그> + 오류 분해
#  11. 회의 번역(stage4-dlg-units-meet-v1 타깃이 있으면, 코퍼스마다 100)              → results/stage4-eval-<run>-<태그>/meet-tr
#  12. AMI 공식 dev 18·eval 16 회의 창 전부(ami-bench-v1: 공식 Mix-Headset = IHM-mix, Array1-01 = SDM) → results/stage3-eval-ami-<run>-<태그> + ami-report.txt(공식 분할별)
# TRANS_LEAD_SPACE=1: 번역 참조 행 앞 공백(학습 --trans-lead-space). 아무것도 지우지 않는다.
set -u
RUN=$1; shift
CODE=${CODE:-/soundai/users/tskim/VAPKT}; cd $CODE; set -a; source /soundai/users/tskim/VAPKT/.env; set +a      # CODE: 코드 사본(본 checkout 반영 전에는 VAPKT-dev)
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; R=/soundai/users/tskim/VAPKT-data/results; OUT=/soundai/Model/VAPASR/stage3-$RUN
TAXI=$D/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl; MS=$D/sim-dialogue-multi-eval-v1; MT=$D/stage4-dlg-units-meet-v1/targets-fix-gen-v3-bf16; AB=$D/ami-bench-v1
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; MIN_FREE_MB=${MIN_FREE_MB:-40000}
LS=""; [ "${TRANS_LEAD_SPACE:-0}" = 1 ] && LS=--trans-lead-space
MULTI="--max-speakers 4 --window 20,90 --window-draws 1 --declare 4"
pick_gpu() {
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
    [ "$(echo $g | cut -d' ' -f2)" -ge $MIN_FREE_MB ] && { echo $g | cut -d' ' -f1; return; }; sleep 300
  done
}
s3() {   # $1 = 셋 이름, $2 = 출력 접미, 나머지 = cst_stage3_eval 인자
  local name=$1 sfx=$2; shift 2; local E=$R/stage3-eval-$sfx-$RUN-$tag A=$R/stage3-attr-errors-$sfx-$RUN-$tag; mkdir -p $E $A
  export CUDA_VISIBLE_DEVICES=$(pick_gpu)
  $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --out $E --deltas 4,10 --batch-size 16 --backchannel 1.0,3 "$@" > $E/run.log 2>&1
  $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
  local data split; data=$(echo "$@" | grep -o -- "--data [^ ]*" | cut -d' ' -f2); split=$(echo "$@" | grep -o -- "--split [^ ]*")
  $P experiments/cst_attr_errors.py --pred $E --data $data $split --out $A --deltas 4,10 --workers 16 > $A/run.log 2>&1
  echo "[$(date '+%F %T')] $name 끝($(grep -c . $E/predictions-rank0.jsonl) 행)"
}
for t in "$@"; do
  echo "[$(date '+%F %T')] $RUN/$t 대기"
  if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 120; done; else until [ -f $OUT/$t/trainer_state.json ]; do sleep 120; done; sleep 60; fi
  tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; CK=$OUT/$t; O=$R/stage4-eval-$RUN-$tag; mkdir -p $O
  export CUDA_VISIBLE_DEVICES=$(pick_gpu); echo "[$(date '+%F %T')] $RUN/$t 7 → GPU $CUDA_VISIBLE_DEVICES"
  $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi90 --deltas 4,10 --max-window 90 --batch-size 8 > $O/taxi90.log 2>&1
  $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi90 > $O/taxi90-summary.log 2>&1
  echo "[$(date '+%F %T')] 7 TAXI 통째 끝"
  s3 "8 긴 창" long --data $P2 --split $P2/splits/v2.json --corpora aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc --window 60,90 --window-draws 1 --limit-windows 100
  s3 "9 회의" meet --data $P2 --split $P2/splits/v4.json --corpora ami,icsi,notsofar --limit-windows 150 $MULTI
  [ -f $MS/sim_ko.refined.dialogues.jsonl ] && s3 "10 3–4 인 합성" msim --data $MS --corpora sim_ko,sim_en,sim_de --limit-windows 150 $MULTI
  if [ -f $AB/manifest.json ]; then s3 "12 AMI 공식 dev·eval(IHM-mix·SDM)" ami --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 $MULTI
    $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $R/stage3-eval-ami-$RUN-$tag > $R/stage3-eval-ami-$RUN-$tag/ami-report.txt 2>&1; fi
  if [ -d $MT/pack ]; then export CUDA_VISIBLE_DEVICES=$(pick_gpu)
    $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v4.json --trans $MT --corpora ami,icsi,notsofar --deltas 4,10 \
      --limit-windows 100 --batch-size 16 --backchannel 1.0,3 $LS $MULTI --out $O/meet-tr > $O/meet-tr.log 2>&1
    $P experiments/cst_stage4_eval.py summarize --out $O/meet-tr > $O/meet-tr-summary.log 2>&1; echo "[$(date '+%F %T')] 11 회의 번역 끝"; fi
  echo "[$(date '+%F %T')] $RUN/$t V2 추가 평가 끝"
done
echo "V2_EXTRA_DONE"
