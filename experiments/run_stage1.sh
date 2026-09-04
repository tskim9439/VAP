#!/usr/bin/env bash
# Stage 1 encoder probing 실행 매트릭스. 인자: 인코더들. 각 인코더를 (a) 고유율, (b) 공통 12.5 Hz 로 학습·평가.
#   bash experiments/run_stage1.sh cpc nemotron-c0 ...
set -uo pipefail; cd "$(dirname "${BASH_SOURCE[0]}")/.."
for enc in "$@"; do
  echo "=== [$(date +%H:%M:%S)] $enc native ==="; python experiments/train_probe.py --encoder "$enc" --epochs 4
  case "$enc" in nemotron-*) echo "($enc 고유율이 12.5 Hz → common 조건 생략)";; *) echo "=== [$(date +%H:%M:%S)] $enc @12.5 Hz ==="; python experiments/train_probe.py --encoder "$enc" --frame-hz 12.5 --epochs 4 --tag common;; esac
done
echo "=== [$(date +%H:%M:%S)] 완료 ==="
