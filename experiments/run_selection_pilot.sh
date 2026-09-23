#!/usr/bin/env bash
# Dedicated two-GPU pilot; no scheduler/training jobs are modified.
set -euo pipefail
project=/soundai/users/tskim/VAPKT
selection=/soundai/users/tskim/VAPKT-data/data/selection/v0.1-20260918
selection_python=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
cd "$project"
export OMP_NUM_THREADS=4
export HF_HOME=/soundai/users/tskim/VAPKT-data/cache/huggingface-whisper
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
# A process lock prevents accidental duplicate submission of this pilot.
exec 9>/tmp/vapkt-data-selection-v01-pilot.lock
flock -n 9 || { echo "Another selection pilot owns the lock"; exit 1; }
ready=0
for ((attempt=0; attempt<180; attempt++)); do
  if /usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("complete",False))' "$selection/census.json" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 20
done
[[ "$ready" == 1 ]] || { echo "Census did not complete; no GPU work started"; exit 1; }
for gpu in 2 3; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "$used" -lt 1000 ]] || { echo "GPU $gpu busy (${used} MiB); refusing to start"; exit 1; }
done
echo "Starting Qwen on GPU 2; Whisper v2 on GPU 3; batch 128; cap 85%"
"$selection_python" -u experiments/selection_teachers.py \
  --input "$selection/pilot.jsonl" --output "$selection/qwen.jsonl" \
  --teacher qwen --model /soundai/Model/Qwen3-ASR-0.6B \
  --gpu 2 --batch 128 --memory-fraction .85 > "$selection/qwen.log" 2>&1 &
qwen_pid=$!
"$selection_python" -u experiments/selection_teachers.py \
  --input "$selection/pilot.jsonl" --output "$selection/whisper-v2.jsonl" \
  --teacher whisper --model /soundai/users/tskim/VAPKT-data/baselines/whisper-large-v2 \
  --gpu 3 --batch 128 --memory-fraction .85 > "$selection/whisper-v2.log" 2>&1 &
whisper_pid=$!
echo "qwen_pid=$qwen_pid whisper_pid=$whisper_pid"
qwen_rc=0; whisper_rc=0
wait "$qwen_pid" || qwen_rc=$?
wait "$whisper_pid" || whisper_rc=$?
[[ "$qwen_rc" == 0 && "$whisper_rc" == 0 ]] || {
  echo "Teacher failed: qwen=$qwen_rc whisper=$whisper_rc"; exit 1;
}
"$selection_python" -u experiments/selection_report.py \
  --input "$selection/pilot.jsonl" --qwen "$selection/qwen.jsonl" \
  --whisper "$selection/whisper-v2.jsonl" --out "$selection/pilot-results"
echo "Pilot complete; human review required before training promotion."
