#!/usr/bin/env bash
# gold-set semcommit eval for one model over sets × δ on one GPU (sequential). usage: run_gold_eval.sh <gpu> <model> <outroot> <set>...
set -u
gpu=$1; model=$2; out=$3; shift 3
cd /soundai/users/tskim/VAPKT
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
G=/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1
L=/soundai/users/tskim/VAPKT-data/data/semcommit-work/gate/speechlm-all19-g1-punctcoord-v035-20260926
P=/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VAPASR_LOCAL_CACHE=/soundai/users/tskim/VAPKT-data/runs/semcommit/eval/.cache
mkdir -p "$out" "$VAPASR_LOCAL_CACHE"
for s in "$@"; do
  for d in 2 3 4 6 8; do
    [ -f "$out/$s-d$d.json" ] && { echo "skip $s d$d"; continue; }
    echo "[$(date '+%T')] $s δ=$d gpu=$gpu"
    $PY experiments/semcommit_eval.py --model "$model" --encoder /soundai/Model/nemotron-3.5-asr-streaming-0.6b \
      --words "$G/words-$s.jsonl" --labels "$L/labels-$s.jsonl" --delay $d --sem-bias -2 -1 0 1 2 --gpu "$gpu" --batch-size 32 \
      --path-remap "/data4/tskim/DBs/KsponSpeech/extracted/=$P/KsponSpeech/" "/data5/LibriSpeech/=$P/LibriSpeech/" "/data4/tskim/semcommit/data/gold/wav/=$G/wav/" \
      --out "$out/$s-d$d.json" > "$out/$s-d$d.log" 2>&1 || echo "FAIL $s d$d (see $out/$s-d$d.log)"
  done
done
echo "[$(date '+%T')] all done gpu=$gpu"
