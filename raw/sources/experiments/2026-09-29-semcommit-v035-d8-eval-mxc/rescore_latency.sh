#!/usr/bin/env bash
# 골드 평가 스트림(88220f7 디코드)을 5103556 채점(창 없는 commit·display·after_text 지연)으로 다시 채점 — GPU 없음, 원 보고서·스트림은 건드리지 않고 latency/ 에 새로 쓴다.
# usage: rescore_latency.sh <set>
set -u
s=$1; R=/soundai/users/tskim/VAPKT-data/runs/semcommit/eval/v035-snap0929-d8
cd /soundai/users/tskim/VAPKT
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
G=/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1
L=/soundai/users/tskim/VAPKT-data/data/semcommit-work/gate/speechlm-all19-g1-punctcoord-v035-20260926
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
mkdir -p "$R/latency"
for d in 2 3 4 6 8; do
  [ -f "$R/latency/$s-d$d.json" ] && { echo "skip $s d$d"; continue; }
  echo "[$(date '+%T')] $s δ=$d"
  $PY experiments/semcommit_eval.py --score-only --words "$G/words-$s.jsonl" --labels "$L/labels-$s.jsonl" \
    --out "$R/latency/$s-d$d.json" --streams-out "$R/$s-d$d.streams.jsonl" > "$R/latency/$s-d$d.log" 2>&1 || echo "FAIL $s d$d (see $R/latency/$s-d$d.log)"
done
echo "[$(date '+%T')] rescore done $s"
