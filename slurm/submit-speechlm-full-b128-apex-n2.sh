#!/usr/bin/env bash
# One-command full ASR/alignment submission. No global environment changes.
set -euo pipefail

cd /soundai/users/tskim/VAPKT
mkdir -p logs/train

sbatch \
  --partition=apex \
  --nodes=2 \
  --ntasks=16 \
  --ntasks-per-node=8 \
  --gres=gpu:8 \
  --cpus-per-task=8 \
  --time=24:00:00 \
  --job-name=SA_EVAL_FullduplexEval \
  --open-mode=append \
  --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out \
  --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

export PROJECT=/soundai/users/tskim/VAPKT
export PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export OUT=/soundai/users/tskim/VAPKT-data/data/speechlm-asr-align-v1-full-b128-20260923
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

cd "$PROJECT"
# Keep two independently submitted jobs from preparing/writing the same root.
exec 9>"${OUT}.job.lock"
flock -n 9 || { echo "Another job is using $OUT" >&2; exit 2; }

if [[ ! -f "$OUT/config.json" ]]; then
    "$PY" -u experiments/speechlm_asr_align.py prepare \
        --out "$OUT" \
        --per-catalog 0 \
        --shard-rows 4096 \
        --batch 128 \
        --batch-sec 960 \
        --io-workers 7
fi

# Refuse to silently resume a different recipe.
"$PY" - "$OUT/config.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    config = json.load(handle)
expected = dict(scope="full", batch=128, batch_sec=960, io_workers=7)
for name, value in expected.items():
    if config.get(name) != value:
        raise SystemExit(f"Recipe mismatch: {name}={config.get(name)!r}, expected {value!r}")
PY

"$PY" -u experiments/speechlm_asr_align.py check \
    --out "$OUT" --check-rows 2

# Executed as bash: the embedded #SBATCH defaults do not override this job.
bash slurm/speechlm_asr_align.sbatch
SBATCH
