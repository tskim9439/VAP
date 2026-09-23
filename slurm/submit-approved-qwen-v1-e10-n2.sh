#!/usr/bin/env bash
# approved Qwen-verbatim v1.1: apex 2 nodes x 8 GPUs, encoder unfrozen, 10 epochs.
# This wrapper does not modify the caller's environment and does not propagate
# the caller's full environment into the Slurm job.
set -euo pipefail

readonly PROJECT="/soundai/users/tskim/VAPKT"
readonly VIEW="/soundai/users/tskim/VAPKT-data/data/selection/approved-qwen-verbatim-mono-v1.1-20260921"
readonly RUN_NAME="approved-qwen-v1-e10-n2"

cd "$PROJECT"

exec sbatch \
  --partition=apex \
  --nodes=2 \
  --export="VAPASR_STRICT_SCHEMA=1,RUN=${RUN_NAME},TRAIN=approved-en:approved-ko,EPOCHS=10,BS_EN=8,BS_KO=8,EVAL_EVERY=1000,SAVE_EVERY=500,LANGUAGE_SCHEDULE=proportional,TRAIN_MAN_ROOT=${VIEW}/manifests,TRAIN_ALIGN_ROOT=${VIEW}/align,TRAIN_ENCODER=1,LR_ENCODER=1e-5" \
  slurm/s3_train_hf.sbatch
