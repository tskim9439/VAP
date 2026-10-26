#!/usr/bin/env bash
# E1 화자 분리 가지 학습 제출(2026-10-25, 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해" — wiki output-diarization-integration-20261025):
# slurm/stage3-train-apex.sbatch 를 SCRIPT=experiments/diar_train.py 로 낸다(선점 저장·requeue·감시 그대로). 제출은 사용자(스케줄러 셸).
#   인코더: 1.7B 회의 + 4 슬롯 final(stage3-q17-v3-p1-tel-xlw-mt4/final)의 인코더를 동결(fp32)해 홀수 층 + 마지막 층을 쓴다.
#   가지: Transformer 12 층 · d 512 · 8 head(약 38M), 화자 4(도착 순서), 손실 = 0.5·도착 순서 BCE + 0.5·PIT.
#   데이터: 대화 10 셋(회의 AMI·ICSI·NOTSOFAR 포함, split v4 — AMI 공식 dev·eval·NOTSOFAR dev·eval 제외) + 합성 4 셋(2 인·3–4 인·교차 언어·전화),
#           창 20–90 s draw 4, 잔향(실제 RIR) 0.5 · 잡음 0.4 · 이득 ±6 dB. 가중 = --weights(기본은 아래 W).
#   산출 /soundai/Model/VAPASR/diar-<run>(checkpoint-N · final · DONE). 평가는 experiments/diar_eval.py(AMI 창 DER — 호스트 GPU 로 돈다).
# 쓰기: bash slurm/submit-diar-train.sh [--run q17-diar-e1] [--init <체크포인트>] [--nodes 1] [--partition apex] [--max-steps 20000] [--batch 8]
#        [--weights '패턴:가중,…'] [--after-job JOBID] [--exclude N,…|none] [--extra "diar_train.py 인자"] [--dry-run]. 아무것도 지우지 않는다.
#   --extra(2026-10-26): 예 E1c 인과 가지 "--lookahead 0 --init-branch /soundai/Model/VAPASR/diar-q17-diar-e1/final/diar.pt --lr 1e-4 --warmup 500".
set -euo pipefail
cd "$(dirname "$0")/.."
run=q17-diar-e1; init=/soundai/Model/VAPASR/stage3-q17-v3-p1-tel-xlw-mt4/final; nodes=1; part=apex; steps=20000; batch=8; after=""; dry=0; extra=""; corpora=""; sims="" 
exclude=hpc-50,hpc-51,hpc-101,hpc-102
W="ami:3,icsi:2,notsofar:0.5,sim-dialogue-v3-multi:3,sim-dialogue-v2:1,sim-dialogue-xl-v1:0.5,sim-dialogue-xl-tel-v1:0.5"   # 폴더 이름 = 그 폴더 셋 전부
while [[ $# -gt 0 ]]; do
  case $1 in
    --run) run=$2; shift 2 ;;  --init) init=$2; shift 2 ;;  --nodes) nodes=$2; shift 2 ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;
    --batch) batch=$2; shift 2 ;;  --weights) W=$2; shift 2 ;;  --after-job) after=$2; shift 2 ;;  --exclude) exclude=$2; shift 2 ;;  --dry-run) dry=1; shift ;;
    --extra) extra=$2; shift 2 ;;  --corpora) corpora=$2; shift 2 ;;  --sim-data) sims=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $exclude == none ]] && exclude=""
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
[[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; exit 2; }
D=/soundai/users/tskim/VAPKT-data/data; OUT=/soundai/Model/VAPASR/diar-$run
ARGS="--init $init --sim-data ${sims:-$D/sim-dialogue-v2,$D/sim-dialogue-v3-multi,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1} --weights $W --noise-bank $D/noise-bank-v1"
ARGS="$ARGS --max-steps $steps --batch $batch --save-every 2000 --num-workers 4${corpora:+ --corpora $corpora}${extra:+ $extra}"
EXC="--exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4"; [[ -n $exclude ]] && EXC="$EXC,$(echo "$exclude" | sed 's/\(^\|,\)hpc-/\1slurmmxch200v5-hpc-/g')"
export MAX_RESTART=${MAX_RESTART:-$([[ $part == apex ]] && echo 5 || echo 20)}
cmd=(sbatch --job-name=SA_DIAR --partition="$part" --nodes="$nodes" --time=24:00:00 $EXC ${after:+--dependency=afterok:$after}
     "--export=ALL,OUT=$OUT,SCRIPT=experiments/diar_train.py,MAX_RESTART=$MAX_RESTART,ARGS_EXTRA=${ARGS//,/%2C}" slurm/stage3-train-apex.sbatch)
echo "[$(date '+%F %T')] E1 화자 분리 가지 diar-$run ← 인코더 $init · $nodes 노드 · $part · $steps step · MAX_RESTART $MAX_RESTART${after:+ · job $after 성공 뒤}" >&2
if (( dry )); then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
[[ -e $OUT/DONE ]] && { echo "이미 끝났다: $OUT/DONE" >&2; exit 1; }
"${cmd[@]}"
