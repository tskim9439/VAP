#!/usr/bin/env bash
# Only this shell/Slurm job receives these settings; no profile/global exports.
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
DATA=${DATA:-/soundai/users/tskim/VAPKT-data/data}
mode=${1:-smoke}
nodes=${2:-1}
[[ $mode == smoke || $mode == full ]] || { echo 'mode must be smoke or full' >&2; exit 2; }
[[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo 'nodes must be a positive integer' >&2; exit 2; }
OUT=${OUT:-$DATA/speechlm-asr-align-v1-$mode-20260922}
"$PY" - "$OUT" "$DATA/speechlm-asr-index-audit-v1" <<'PY'
import json, sys
from pathlib import Path
root, audit = map(Path, sys.argv[1:])
config = json.loads((root / 'config.json').read_text())
index = json.loads((audit / 'index-summary.json').read_text())
inputs = json.loads((audit / 'index-input.json').read_text())
check = json.loads((root / 'cpu-check.json').read_text())
assert index['complete'] and index['ready'], 'tar pre-index not ready'
assert inputs['source_summary_sha256'] == config['source_summary_sha256'], 'different index source'
assert inputs['tar_index'] == config['tar_index'], 'different index directory'
assert not check['counts'].get('error'), 'CPU audio checks failed'
assert check['counts'].get('ok', 0) > 0, 'No passing CPU audio samples'
print(f"Submitting {config['scope']}: {config['rows']} rows / {len(config['jobs'])} shards")
PY
cd "$PROJECT"
sbatch --partition=hpc --nodes="$nodes" --ntasks="$((nodes * 8))" \
    --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=8 \
    --export="ALL,OUT=$OUT,PY=$PY,PROJECT=$PROJECT" slurm/speechlm_asr_align.sbatch
