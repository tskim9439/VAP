#!/usr/bin/env bash
# semcommit v0.3.5 part-unit labeling directly on the mxc host GPUs (no SLURM), next to (or instead of) the SLURM job.
# Same worker as slurm/semcommit-part-label-apex.sbatch; part locks keep it from colliding with a running job, and it walks
# parts-order.tsv from the end (ORDER=reverse) so the two meet as late as possible.
#   bash slurm/semcommit-direct-label.sh 1,3,4        # one background worker per listed GPU (check nvidia-smi first)
# Logs: $OUT/direct-logs/<run id>-g<gpu>.log. Stop: kill the worker PIDs printed at launch (parts resume later).
set -euo pipefail
cd "$(dirname "$0")/.."
gpus=${1:?comma-separated GPU indices, e.g. 1,3,4}

D=/soundai/users/tskim/VAPKT-data/data
W=$D/semcommit-work
G=$W/gate/speechlm-all19-g1-punctcoord-v035-20260926
export PROJECT=$PWD
export PY=/soundai/users/tskim/VAPKT-data/conda/envs/semcommit-teacher/bin/python
export QWEN=/soundai/Model/Qwen3.8-27B EXAONE=/soundai/Model/EXAONE-4.0-32B QWEN_ASR=/soundai/Model/Qwen3-ASR-0.6B
export RESULTS=$D/speechlm-asr-align-v1-full-b128-20260923/results
export QC_SPLIT=$W/qc-pass-split-v035-20260926
export APPROVAL_SUMMARY=$G/decision/training-eligibility-summary.json
export GATE_ACCEPTED=$G/decision/gate-accepted.json
export THRESHOLDS=$G/thresholds.json
export OUT=$W/labels/speechlm-all19-v035
export A_KIND=qwen38 C_KIND=qwen38 ORDER=reverse
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# the same preflight as the SLURM job (gate acceptance, thresholds, QC split bound to the approval)
mkdir -p "$OUT/parts" "$OUT/direct-logs"
awk '/^"\$PY" - "\$APPROVAL_SUMMARY"/{f=1;next} /^PY$/{if(f){exit}} f' slurm/semcommit-part-label-apex.sbatch \
  | "$PY" - "$APPROVAL_SUMMARY" "$GATE_ACCEPTED" "$THRESHOLDS" "$QC_SPLIT/summary.json"
export EXTRA_CANDIDATES=$("$PY" -c "import json,pathlib,sys; print(','.join(json.loads(pathlib.Path(sys.argv[1]).with_name('thresholds.report.json').read_text())['extra']))" "$THRESHOLDS")

IFS=, read -ra list <<< "$gpus"
export RUN_ID="direct$(date +%Y%m%d%H%M%S)" NWORKERS=${#list[@]}
k=0
for g in "${list[@]}"; do
  log=$OUT/direct-logs/$RUN_ID-g$g.log
  CUDA_VISIBLE_DEVICES=$g WORKER_ID=$k setsid nohup bash slurm/semcommit-part-label-worker.sh > "$log" 2>&1 < /dev/null &
  echo "worker $k gpu $g pid $! log $log"
  k=$((k + 1))
done
