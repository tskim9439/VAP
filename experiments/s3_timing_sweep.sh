#!/usr/bin/env bash
# 최종 모델 타이밍 분석: δ(방출 지연 조건) ∈ {2,3,4} 와 next_bias ∈ {0,1,2}(δ=2) 를 sentinel 표본(10/10/100, seed 7)으로 평가 → eval/sweep-d<δ>-b<bias>-<step>.json
#   scripts/sync-mxc.sh bg sweep "experiments/s3_timing_sweep.sh /soundai/Model/VAPASR/hf-C2/eval-ckpts/checkpoint-21960 5 <wait-log>"
set -uo pipefail
CK=${1:?ckpt dir}; GPU=${2:-5}; WAIT=${3:-}; R=$(dirname $(dirname $CK))
[ -n "$WAIT" ] && while ! grep -q "EXIT=" "$WAIT" 2>/dev/null; do sleep 20; done
for cfg in "2 0" "3 0" "4 0" "2 1" "2 2"; do
  set -- $cfg; d=$1; b=$2; tag="sweep-d$d-b$b"
  echo "[$(date '+%F %T')] $tag"
  CUDA_VISIBLE_DEVICES=$GPU python experiments/s3_train_hf.py --eval-only --init "$CK" --out-dir "$R" --sentinel-stream 10 --sentinel-utt 100 --eval-seed 7 --eval-tag "$tag" --eval-delay "$d" --eval-bias "$b" --num-workers 0 2>&1 | grep -E "_score|Traceback|Error" | cut -c1-500
done
echo "[$(date '+%T')] sweep done"
