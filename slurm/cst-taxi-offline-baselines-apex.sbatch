#!/usr/bin/env bash
# CST-Bench TAXI (German<->English, 86 sessions, 640 turns): offline baselines with oracle turn segmentation.
#   GPU0: Qwen3-ASR-1.7B (vapasr env), then gold transcript / ASR transcript -> Qwen3.8-27B MT (teacher env:
#         Qwen3.8 needs transformers 5.8, qwen_asr does not run on it)
#   GPU1: Whisper-large-v3 task=translate (German->English only, vapasr env)
# Then `cstbench eval` on taxi-natural for each hypothesis file.
#SBATCH --job-name=SA_EVAL_Fullduplex
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --exclude=slurmmxch200v5-hpc-52
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail

BENCH=/soundai/users/tskim/VAPKT/cst-bench
PY=/soundai/users/tskim/VAPKT-data/conda/envs/semcommit-teacher/bin/python
PYA=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1
SES=$TAXI/taxi-natural/sessions.jsonl
OUT=/soundai/users/tskim/VAPKT-data/results/cst-taxi-baselines/offline-oracle-${SLURM_JOB_ID}
test -x "$PY" && test -x "$PYA" && test -f "$SES"
test -d /soundai/Model/Qwen3.8-27B && test -d /soundai/Model/Qwen3-ASR-1.7B
test -d /soundai/users/tskim/VAPKT-data/baselines/whisper-large-v3
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$BENCH"
mkdir -p "$OUT/llm" "$OUT/whisper"

( CUDA_VISIBLE_DEVICES=0 "$PYA" baselines/offline_oracle.py --sessions "$SES" --out-dir "$OUT/llm" \
    --systems asr --device cuda:0 --batch 16 --asr-model /soundai/Model/Qwen3-ASR-1.7B &&
  CUDA_VISIBLE_DEVICES=0 "$PY" baselines/offline_oracle.py --sessions "$SES" --out-dir "$OUT/llm" \
    --systems gold-mt,cascade --device cuda:0 --batch 16 --llm-model /soundai/Model/Qwen3.8-27B ) > "$OUT/llm.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 "$PYA" baselines/offline_oracle.py --sessions "$SES" --out-dir "$OUT/whisper" \
  --systems whisper --device cuda:0 --batch 16 \
  --whisper-model /soundai/users/tskim/VAPKT-data/baselines/whisper-large-v3 > "$OUT/whisper.log" 2>&1 &
p2=$!
rc=0
wait $p1 || rc=1
wait $p2 || rc=1
tail -n 5 "$OUT/llm.log" "$OUT/whisper.log"

for h in "$OUT"/llm/hyp-*.jsonl "$OUT"/whisper/hyp-*.jsonl; do
  [[ -f "$h" ]] || continue
  name=$(basename "$h" .jsonl)
  for clock in t elapsed; do
    PYTHONPATH=src "$PY" -m cstbench.cli eval --sessions "$SES" --hyp "$h" --time-key "$clock" \
      --out "$OUT/report-${name#hyp-}-$clock.json" --segments-out "$OUT/segments-${name#hyp-}-$clock.jsonl" > /dev/null
  done
done
echo "REPORTS=$OUT"
exit $rc
