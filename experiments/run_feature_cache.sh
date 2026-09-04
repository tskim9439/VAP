#!/usr/bin/env bash
# Stage 1 특징 캐시: 인코더 × 코퍼스. 인자로 인코더 이름들. 재개 가능 (index.jsonl 기준 skip).
#   bash experiments/run_feature_cache.sh cpc nemotron-c0
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
M="${DATA_MANIFEST_DIR:-/data3/tskim/manifests}"
for enc in "$@"; do
  echo "=== [$(date +%H:%M:%S)] $enc ==="
  python experiments/extract_features.py --encoder "$enc" --manifest "$M/otoSpeech"     --audio-root /data3/tskim/corpora/turnbench/otoSpeech16k
  python experiments/extract_features.py --encoder "$enc" --manifest "$M/aihub-ts01-5"  --audio-root /data3/tskim/corpora/aihub/adult-ts01-5-wav --limit-hours 50
  python experiments/extract_features.py --encoder "$enc" --manifest "$M/aihub-vs02"    --audio-root /data3/tskim/corpora/aihub/adult-vs02-wav
  python experiments/extract_features.py --encoder "$enc" --manifest "$M/turnbench-dev" --audio-root /data3/tskim/cache/huggingface/hub/datasets--mundo-ai--turn-benchmark-dev/snapshots --include-flagged   # scorer 는 dev 38개 전부 요구
done
echo "=== [$(date +%H:%M:%S)] 완료 ==="; du -sh ${DATA_FEATURE_CACHE_DIR:-/data3/tskim/features}/* 2>/dev/null
