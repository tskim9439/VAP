#!/usr/bin/env bash
# All environment changes are confined to this child shell.
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
DATA=${DATA:-/soundai/users/tskim/VAPKT-data/data}
SOURCE=$DATA/speechlm-asr-single-speaker-v1-20260922
INDEX=$DATA/speechlm-asr-tar-index-v1
OUT=${OUT:-$DATA/speechlm-asr-align-v1-smoke-20260922}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$PROJECT"
mode=${1:-smoke}
case "$mode" in
    index)
        "$PY" -u experiments/speechlm_asr_align.py index --source "$SOURCE" \
            --out "$DATA/speechlm-asr-index-audit-v1" --tar-index "$INDEX" --index-workers "${INDEX_WORKERS:-16}"
        ;;
    smoke|full)
        per_catalog=32
        if [[ $mode == full ]]; then
            per_catalog=0
            OUT=${OUT_FULL:-$DATA/speechlm-asr-align-v1-full-20260922}
        fi
        "$PY" -u experiments/speechlm_asr_align.py prepare --source "$SOURCE" --out "$OUT" \
            --tar-index "$INDEX" --per-catalog "$per_catalog" --batch 64 --batch-sec 480 --io-workers 7
        "$PY" -u experiments/speechlm_asr_align.py check --out "$OUT" --check-rows 2
        ;;
    *) echo "Usage: bash $0 {index|smoke|full}" >&2; exit 2 ;;
esac
