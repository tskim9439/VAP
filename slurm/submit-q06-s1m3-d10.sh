#!/usr/bin/env bash
# Stage 1 세 언어(EN·KO·DE) + 지연 δ4–10, Qwen3-ASR-0.6B thinker(실시간성) — q06-s0(adapter 워밍업)에서 3 epoch(2026-10-08).
#   q17-s1m-r3 레시피(인코더 학습·[56,3] 고정·SpecAugment·속도·잡음·final 30 %)와 같고, 독일어(YODAS-Granary asr-tn-de-v1.1.0, 2,125 h)와
#   δ4–10(미리 듣기 320–1,040 ms, <DELAY_9/10> 은 학습 스크립트가 붙인다)을 처음부터 함께 배운다.
# 경로 변수는 이 스크립트 안에서만 쓰인다(제출 셸에 남지 않는다). 추가 인자는 submit-semcommit-train-apex.sh 에 그대로 넘어가고
# 같은 옵션은 뒤의 값이 이긴다. 예:
#   bash slurm/submit-q06-s1m3-d10.sh --dry-run          # sbatch 없이 명령만 출력
#   bash slurm/submit-q06-s1m3-d10.sh                    # 제출(apex 2 노드)
#   bash slurm/submit-q06-s1m3-d10.sh --after <JOBID>    # 그 job 이 끝난 뒤 시작
# 데이터 로더 worker 는 GPU 당 4 개(--num-workers 4): job 82738 이 8 개로 돌다 노드 메모리 부족(oom_kill)으로 세 번 죽었다 —
#   독일어 1.3M 항목이 더해져 worker 마다 복사되는 데이터셋이 커졌다. 같은 --run 으로 다시 내면 마지막 checkpoint 에서 이어서 한다.
#   주의: --extra 를 다시 주면 위 --extra 가 통째로 바뀐다(인코더 [56,3] 고정 옵션이 빠짐).
set -euo pipefail
cd "$(dirname "$0")/.."
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
INIT=/soundai/Model/VAPASR/semcommit-q06-s0/final
exec bash slurm/submit-semcommit-train-apex.sh --run q06-s1m3-d10 --snap "$SNAP" --init "$INIT" \
  --asr-only --offline-frac 0.3 --asr-list "$ASR" --de-list "$DE" \
  --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$NB" --varlen \
  --delays 4,5,6,7,8,9,10 --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs 3 --save-every 1000 \
  --nodes 2 --extra "--encoder-right-context 3 --encoder-context-sampling fixed --num-workers 4" "$@"
