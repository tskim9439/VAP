#!/usr/bin/env bash
# Stage 1 세 언어(EN·KO·DE) + 지연 δ4–10 학습(2026-10-08) — q17-s1m-r3(EN·KO, 인코더 [56,3])에서 1 epoch 이어서.
#   독일어: YODAS-Granary asr-tn-de-v1.1.0(2,125 h), δ4–10 + [56,3] → 미리 듣기 320–1,040 ms(<DELAY_9/10> 은 학습 스크립트가 붙인다).
# 경로 변수는 이 스크립트 안에서만 쓰인다(제출 셸에 남지 않는다). 추가 인자는 submit-semcommit-train-apex.sh 에 그대로 넘어가고
# 같은 옵션은 뒤의 값이 이긴다. 예:
#   bash slurm/submit-q17-s1m3-d10.sh --dry-run          # sbatch 없이 명령만 출력
#   bash slurm/submit-q17-s1m3-d10.sh                    # 제출(apex 2 노드)
#   bash slurm/submit-q17-s1m3-d10.sh --after <JOBID>    # 그 job 이 끝난 뒤 시작
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
INIT=/soundai/Model/VAPASR/semcommit-q17-s1m-r3/final
exec bash slurm/submit-semcommit-train-apex.sh --run q17-s1m3-d10 --snap "$SNAP" --init "$INIT" \
  --asr-only --offline-frac 0.3 --asr-list "$ASR" --de-list "$DE" \
  --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$NB" --varlen \
  --delays 4,5,6,7,8,9,10 --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs 1 --save-every 1000 \
  --nodes 2 --extra "--encoder-context-sampling fixed" "$@"
