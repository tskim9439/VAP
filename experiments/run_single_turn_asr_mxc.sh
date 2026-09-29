#!/usr/bin/env bash
# Two GPUs at most, per-process environment only. No login-shell/env changes.
# usage: run_single_turn_asr_mxc.sh <checkpoint> <out> [gpus=0,2] [batch=128] [limit=0] [deltas=2,4] [manifest]
#   manifest: 7th argument or MANIFEST env (default: the v1 LibriSpeech+Kspon manifest). English evaluation uses VoxPopuli-Cleaned-AA from
#   2026-09-29: build .../single-turn-vpaa-kspon-v1/manifest.jsonl with `eval_single_turn_asr.py prepare --voxpopuli-aa-root ...` first.
#   PY env overrides the interpreter; when the node-local env cannot import numpy (seen 2026-09-29: PyCapsule datetime error) the conda env is used.
set -euo pipefail
PROJECT=/soundai/users/tskim/VAPKT
DATA=/soundai/users/tskim/VAPKT-data
PY_CONDA=$DATA/conda/envs/vapasr/bin/python
PY=${PY:-/tmp/sa_tskim-vapasr-env-local/bin/python}
if ! "$PY" -c 'import numpy' >/dev/null 2>&1; then
  echo "python env unusable ($PY) -> $PY_CONDA" >&2
  PY=$PY_CONDA
fi
CHECKPOINT=${1:-/soundai/Model/VAPASR/hf-approved-qwen-v1-e10-n2/checkpoint-35000}
OUT=${2:-${DATA}/results/single-turn-35000-d2-d4-v1}
GPU_LIST=${3:-0,2}
BATCH=${4:-128}
LIMIT=${5:-0}
DELAY_LIST=${6:-2,4}
IFS=',' read -r -a DELTAS <<< "$DELAY_LIST"
for delta in "${DELTAS[@]}"; do
  [[ "$delta" =~ ^[1-8]$ ]] || { echo 'Delta must be an integer from 1 to 8.' >&2; exit 2; }
done
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} < 1 || ${#GPUS[@]} > 2 )); then
  echo 'This launcher allows only one or two GPUs.' >&2
  exit 2
fi
if (( ${#GPUS[@]} == 2 )) && [[ ${GPUS[0]} == "${GPUS[1]}" ]]; then
  echo 'GPU IDs must be distinct.' >&2
  exit 2
fi
mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock"
flock -n 9 || { echo 'An evaluation is already running in this directory.' >&2; exit 2; }
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VAPASR_STAGE_DIR=/dev/shm/tskim-vapasr-infer-35000
ENC=/soundai/Model/nemotron-3.5-asr-streaming-0.6b
MANIFEST=${7:-${MANIFEST:-$DATA/data/evaluation/single-turn-v1/manifest.jsonl}}
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 2; }
echo "manifest=$MANIFEST python=$PY"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup INT TERM
for rank in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$rank]} "$PY" experiments/eval_single_turn_asr.py run \
    --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --encoder "$ENC" --out "$OUT" \
    --deltas "${DELTAS[@]}" --tail-s 1.0 --max-flush 8 --batch-size "$BATCH" \
    --rank "$rank" --world-size "${#GPUS[@]}" --limit "$LIMIT" --verify \
    > "$OUT/rank${rank}.log" 2>&1 &
  pids+=("$!")
  echo "rank=$rank gpu=${GPUS[$rank]} pid=${pids[$rank]}"
done
code=0
for p in "${pids[@]}"; do wait "$p" || code=1; done
if (( code != 0 )); then echo 'Worker failed; check rank logs.' >&2; exit "$code"; fi
"$PY" experiments/eval_single_turn_asr.py summarize --out "$OUT" > "$OUT/summary.log" 2>&1
