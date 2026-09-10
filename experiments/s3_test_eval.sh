#!/usr/bin/env bash
# 최종 보고용 test 평가(선정된 final 하나): test-clean · test-other · kspon-eval 전체, ah71-dev(71631 은 test 분할이 없어 held-out dev) 1,000 발화 를 δ=2/4 로 평가.
#   dev 는 학습 중 sentinel·select(checkpoint 선정)에 쓰므로 최종 수치는 test 로 낸다(C2·D2 보고서까지는 dev 로 냈음 — 2026-09-10 부터 교정).
#   실행(컨테이너): GPUS=0,1,3,4 bash experiments/s3_test_eval.sh /soundai/Model/VAPASR/hf-E2/final   → <run>/eval/test-d{2,4}-0.json
set -uo pipefail
CK=${1:?final 디렉토리}; R=$(dirname "$CK"); GPU=${GPUS:-0,1,3,4}; NP=$(echo "$GPU" | tr "," "\n" | wc -l); PORT=$((29500 + RANDOM % 500))
SETS="test-clean=librispeech-test:test-clean:stream,test-other=librispeech-test:test-other:stream,kspon-eval-clean=kspon-eval:eval_clean:utt,kspon-eval-other=kspon-eval:eval_other:utt,ah71-dev=aihub71631-dev:dev:utt:1000"
for d in 2 4; do
  echo "[$(date '+%F %T')] test-d$d ← $CK (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=$NP --master_port=$PORT experiments/s3_train_hf.py --eval-only --init "$CK" --out-dir "$R" --sentinel-stream 100000 --sentinel-utt 100000 --eval-seed 7 --eval-tag "test-d$d" --eval-delay $d --eval-bias 0 --dev-extra "$SETS" --num-workers 0 2>&1 | grep -E "_score|Traceback|Error" | cut -c1-900; PORT=$((PORT + 1))
done
echo "[$(date '+%F %T')] test eval done"; echo EXIT=0
