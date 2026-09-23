#!/usr/bin/env bash
# Stage order: repaired ASR validation -> channel census. Never starts training.
set -euo pipefail
project=/soundai/users/tskim/VAPKT
selection_root=/soundai/users/tskim/VAPKT-data/data/selection
repair="$selection_root/repair-v0.3-20260919"
selection_python=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
cd "$project"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/soundai/users/tskim/VAPKT-data/cache/huggingface-whisper
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
exec 9>/tmp/vapkt-selection-repair-v03.lock
flock -n 9 || exit 1
/usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["original_manifest_unchanged"] and d["repaired_rows"]>0' "$repair/summary.json"
for gpu in 1 2; do
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  [[ "$used" -lt 1000 ]] || { echo "GPU $gpu busy"; exit 1; }
done
"$selection_python" -u experiments/selection_teachers.py --input "$repair/repaired.jsonl" \
  --output "$repair/qwen.jsonl" --teacher qwen --model /soundai/Model/Qwen3-ASR-0.6B \
  --gpu 1 --batch 128 > "$repair/qwen.log" 2>&1 &
qwen_pid=$!
"$selection_python" -u experiments/selection_teachers.py --input "$repair/repaired.jsonl" \
  --output "$repair/whisper-v3.jsonl" --teacher whisper \
  --model /soundai/users/tskim/VAPKT-data/baselines/whisper-large-v3 \
  --gpu 2 --batch 256 > "$repair/whisper-v3.log" 2>&1 &
whisper_pid=$!
echo "REPAIR_VALIDATION qwen=$qwen_pid whisper_v3=$whisper_pid"
failed=0
wait "$qwen_pid" || failed=1
wait "$whisper_pid" || failed=1
[[ "$failed" == 0 ]] || { echo 'Teacher process failed'; exit 1; }
"$selection_python" experiments/selection_report.py --input "$repair/repaired.jsonl" \
  --qwen "$repair/qwen.jsonl" --whisper "$repair/whisper-v3.jsonl" --out "$repair/results-v3"
echo 'CHANNEL_AUDIT starting after repair ASR validation'
"$selection_python" -u experiments/selection_channel_audit.py \
  --selection "$selection_root/v0.1-20260918" --qc "$selection_root/v0.1-20260918/audio-census-w32" \
  --out "$selection_root/channels-v0.3-20260919" --workers 32
echo 'COMPLETE technical checks. Full screening/training NOT started.'
