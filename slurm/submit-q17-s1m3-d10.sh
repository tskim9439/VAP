#!/usr/bin/env bash
# Stage 1 세 언어(EN·KO·DE) + 지연 δ4–10 학습(2026-10-08) — q17-s1m-r3(EN·KO, 인코더 [56,3])에서 1 epoch 이어서.
#   독일어: YODAS-Granary asr-tn-de-v1.1.0(2,125 h), δ4–10 + [56,3] → 미리 듣기 320–1,040 ms(<DELAY_9/10> 은 학습 스크립트가 붙인다).
# 경로 변수는 이 스크립트 안에서만 쓰인다(제출 셸에 남지 않는다). 추가 인자는 submit-semcommit-train-apex.sh 에 그대로 넘어가고
# 같은 옵션은 뒤의 값이 이긴다. 예:
#   bash slurm/submit-q17-s1m3-d10.sh --dry-run          # sbatch 없이 명령만 출력
#   bash slurm/submit-q17-s1m3-d10.sh                    # 제출(apex 2 노드)
#   bash slurm/submit-q17-s1m3-d10.sh --after <JOBID>    # 그 job 이 끝난 뒤 시작
# 2026-10-22 검토(1.7B 체인 slurm/submit-q17-chain.sh --s1 resume 이 이 스크립트를 쓴다 — 기본 체인은 0.6B 와 같은 submit-q17-s1m3-d10-s0.sh):
#   - 데이터 로더 worker 를 GPU 당 4 개로(--num-workers 4). 10-08 job 82730 이 8 개로 돌다 노드 메모리 부족(oom_kill)으로 세 번 죽었다 — 0.6B 82738 과 같은 원인.
#     worker 수는 지문에 없어 checkpoint-1000 에서 그대로 이어 한다.
#   - --encoder-right-context 3 은 일부러 넣지 않는다: init(q17-s1m-r3) config 가 이미 [56,3] 이라 값은 같고(checkpoint-1000 config 도 [56,3]),
#     옵션을 주면 지문에 encoder_right_context 키가 생겨 checkpoint-1000 재개가 거부된다.
#   - 1.7B 체크포인트는 두 조각(model-0000N-of-00002.safetensors)으로 저장된다. 재개 탐색(vapasr.hf.stage.last_complete_checkpoint)이 2026-10-22 부터 조각 저장도 찾는다.
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
  --nodes 2 --extra "--encoder-context-sampling fixed --num-workers 4" "$@"
