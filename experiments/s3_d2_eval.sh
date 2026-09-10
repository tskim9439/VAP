#!/usr/bin/env bash
# D2 최종 평가 파이프라인(컨테이너, 단일 GPU): hf-D2/DONE 을 기다린 뒤
#   ① select(50/50/300 + 71631-dev 300 발화, seed 7, δ=2): eval-ckpts/checkpoint-14000·16000 + final           → eval/select-<step>.json
#   ② 같은 표본으로 δ=4: final                                                                      → eval/select-d4-<step>.json
#   ③ C2 final 도 같은 표본(71631-dev 포함) δ=2/4 로 재평가(비교 기준)                                 → hf-C2/eval/selectx-d{2,4}-21960.json
#   ④ 타이밍 스윕(final, sentinel 10/10/100 seed 7): δ∈{2,3,4}, δ=2 bias∈{1,2}                        → eval/sweep-d<δ>-b<b>-<step>.json
#   실행: docker exec -d -e CUDA_VISIBLE_DEVICES=4 sa_tskim_fd bash -c "cd …/VAPKT; … source scripts/activate-env.sh; bash experiments/s3_d2_eval.sh > …/d2eval.log 2>&1"
set -uo pipefail
R=${R:-/soundai/Model/VAPASR/hf-D2}; C2=${C2:-/soundai/Model/VAPASR/hf-C2}; S=$R/eval-ckpts; GPU=${GPUS:-${CUDA_VISIBLE_DEVICES:-4}}; EXTRA="--dev-extra ah71-dev=aihub71631-dev:dev:utt --num-workers 0"
while [ ! -f $R/DONE ]; do sleep 60; done; echo "[$(date '+%F %T')] DONE 확인"
# 단일 GPU 로는 select 한 번에 70 분(스트리밍 디코드 700 항목) → GPUS(쉼표 목록) 만큼 torchrun 으로 나눠 돈다
NP=$(echo "$GPU" | tr "," "\n" | wc -l); PORT=$((29500 + RANDOM % 500))
run() { local init=$1 out=$2 tag=$3 d=$4 b=$5 ns=$6 nu=$7; local step=0; [[ "$init" =~ checkpoint-([0-9]+)$ ]] && step=${BASH_REMATCH[1]}
  [ -f "$out/eval/$tag-$step.json" ] && { echo "[$(date '+%F %T')] $tag-$step 있음 → 건너뜀"; return; }                       # 재실행 시 이미 있는 결과는 건너뛴다
  echo "[$(date '+%F %T')] $tag ← $init (δ=$d bias=$b, GPU $GPU)"; CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=$NP --master_port=$PORT experiments/s3_train_hf.py --eval-only --init "$init" --out-dir "$out" --sentinel-stream $ns --sentinel-utt $nu --eval-seed 7 --eval-tag "$tag" --eval-delay $d --eval-bias $b $EXTRA 2>&1 | grep -E "_score|Traceback|Error" | cut -c1-700; PORT=$((PORT + 1)); }
for n in 14000 16000; do [ -d $S/checkpoint-$n ] && [ ! -f $R/eval/select-$n.json ] && run $S/checkpoint-$n $R select 2 0 50 300; done
run $R/final $R select 2 0 50 300
run $R/final $R select-d4 4 0 50 300
run $C2/final $C2 selectx-d2 2 0 50 300
run $C2/final $C2 selectx-d4 4 0 50 300
for cfg in "2 0" "3 0" "4 0" "2 1" "2 2"; do set -- $cfg; run $R/final $R "sweep-d$1-b$2" $1 $2 10 100; done
echo "[$(date '+%F %T')] d2 eval done"; echo EXIT=0
