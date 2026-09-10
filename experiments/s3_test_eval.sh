#!/usr/bin/env bash
# 최종 보고용 test 평가(선정된 final 하나): test-clean · test-other · kspon-eval 전체, ah71-dev(71631 은 test 분할이 없어 held-out dev) 1,000 발화 를 δ=2/4 로 평가.
#   dev 는 학습 중 sentinel·select(checkpoint 선정)에 쓰므로 최종 수치는 test 로 낸다(C2·D2 보고서까지는 dev 로 냈음 — 2026-09-10 부터 교정).
#   실행(컨테이너): GPUS=0,1,3,4 bash experiments/s3_test_eval.sh /soundai/Model/VAPASR/hf-E2/final   → <run>/eval/test-d{2,4}-0.json
set -uo pipefail
CK=${1:?final 디렉토리}; R=$(dirname "$CK"); GPU=${GPUS:-0,1,3,4}; NP=$(echo "$GPU" | tr "," "\n" | wc -l); PORT=$((29500 + RANDOM % 500))
# 전체 test 는 모델·δ 당 오디오 ≈17 h(스트리밍 디코드는 실시간 속도로 순차) → 로그인 노드 GPU 2–3 장으로 6–8 h. CAP_STREAM/CAP_UTT 로 seed 7 표본 상한을 두면 태그에 s<N>u<M> 이 붙는다(전체는 SLURM slurm/s3_eval_suite.sbatch)
CS=${CAP_STREAM:-100000}; CU=${CAP_UTT:-100000}; SUF=""; [ "$CS" != 100000 -o "$CU" != 100000 ] && SUF="-s${CS}u${CU}"
SETS="test-clean=librispeech-test:test-clean:stream,test-other=librispeech-test:test-other:stream,kspon-eval-clean=kspon-eval:eval_clean:utt,kspon-eval-other=kspon-eval:eval_other:utt,ah71-dev=aihub71631-dev:dev:utt:1000"
for d in 2 4; do
  TAG="test${SUF}-d$d"; [ -f "$R/eval/$TAG-0.json" ] && { echo "[$(date '+%F %T')] $TAG-0 있음 → 건너뜀"; continue; }
  echo "[$(date '+%F %T')] $TAG ← $CK (GPU $GPU, cap stream $CS / utt $CU)"
  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=$NP --master_port=$PORT experiments/s3_train_hf.py --eval-only --init "$CK" --out-dir "$R" --sentinel-stream $CS --sentinel-utt $CU --eval-seed 7 --eval-tag "$TAG" --eval-delay $d --eval-bias 0 --dev-extra "$SETS" --num-workers 0 2>&1 | grep -E "_score|Traceback|Error" | cut -c1-900; PORT=$((PORT + 1))
done
echo "[$(date '+%F %T')] test eval done"; echo EXIT=0
