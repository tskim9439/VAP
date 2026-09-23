#!/usr/bin/env bash
# Calibration only: three teachers, no full-corpus inference or training approval.
set -euo pipefail
project=/soundai/users/tskim/VAPKT
review=/soundai/users/tskim/VAPKT-data/data/selection/review-v0.2-20260919
selection_python=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
cd "$project"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/soundai/users/tskim/VAPKT-data/cache/huggingface-whisper
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
exec 9>/tmp/vapkt-data-selection-review-v02.lock
flock -n 9 || { echo 'Review run already active'; exit 1; }
ready=0
for ((i=0; i<180; i++)); do
  if /usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["complete"])' "$review/audit.json" 2>/dev/null; then
    ready=1; break
  fi
  sleep 10
done
[[ "$ready" == 1 ]] || { echo 'Audit not complete; no teacher started'; exit 1; }
for gpu in 1 2 3; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "$used" -lt 1000 ]] || { echo "GPU $gpu busy; no teacher started"; exit 1; }
done
"$selection_python" -u experiments/selection_review_bundle.py --input "$review/review.jsonl" \
  --out "$review/listening" --export --workers 32 > "$review/listening.log" 2>&1 &
export_pid=$!
"$selection_python" -u experiments/selection_teachers.py --input "$review/review.jsonl" \
  --output "$review/qwen.jsonl" --teacher qwen --model /soundai/Model/Qwen3-ASR-0.6B \
  --gpu 1 --batch 128 > "$review/qwen.log" 2>&1 &
qwen_pid=$!
"$selection_python" -u experiments/selection_teachers.py --input "$review/review.jsonl" \
  --output "$review/whisper-v2.jsonl" --teacher whisper \
  --model /soundai/users/tskim/VAPKT-data/baselines/whisper-large-v2 \
  --gpu 2 --batch 128 > "$review/whisper-v2.log" 2>&1 &
v2_pid=$!
"$selection_python" -u experiments/selection_teachers.py --input "$review/review.jsonl" \
  --output "$review/whisper-v3.jsonl" --teacher whisper \
  --model /soundai/users/tskim/VAPKT-data/baselines/whisper-large-v3 \
  --gpu 3 --batch 256 > "$review/whisper-v3.log" 2>&1 &
v3_pid=$!
echo "START export=$export_pid qwen=$qwen_pid v2=$v2_pid v3=$v3_pid"
failed=0
for child in "$export_pid" "$qwen_pid" "$v2_pid" "$v3_pid"; do
  wait "$child" || failed=1
done
[[ "$failed" == 0 ]] || { echo 'Worker failure: inspect logs, no approval'; exit 1; }
for version in v2 v3; do
  "$selection_python" experiments/selection_report.py --input "$review/review.jsonl" \
    --qwen "$review/qwen.jsonl" --whisper "$review/whisper-$version.jsonl" \
    --out "$review/results-$version"
done
"$selection_python" experiments/selection_review_bundle.py --input "$review/review.jsonl" \
  --out "$review/listening" --teacher "Qwen=$review/qwen.jsonl" \
  --teacher "Whisper-v2=$review/whisper-v2.jsonl" --teacher "Whisper-v3=$review/whisper-v3.jsonl"
echo 'COMPLETE: human listening required; full screening and training NOT started'
