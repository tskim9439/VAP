#!/usr/bin/env bash
# semcommit v0.2 r1: E2 → +<SEM_END>/<TURN_END>, LibriSpeech train-clean-100 (400 streams, 3.3 h) + KsponSpeech_01 (1977 streams, 3.9 h), rack4 GPU 0.
# Encoder frozen (saved into every checkpoint), low LR so ASR stays near E2. Resumable: rerun (resume=auto, fingerprint-checked).
set -uo pipefail
cd /home/tskim/VAP
export HF_HOME=/data3/tskim/cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=/data4/tskim/semcommit/data/v0; L=/data4/tskim/semcommit/labels/v0.2
NEMO=/data3/tskim/cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/1c8deaecc64b91f034d73e08dd8b64625eb3395d
/opt/conda/envs/vapasr/bin/python experiments/semcommit_train.py --init /data4/tskim/VAPASR/exports/hf-E2-final --nemotron-dir $NEMO \
  --train-words $D/words-ls-train.jsonl --train-labels $L/labels-ls-train.jsonl --train-words $D/words-ks-train.jsonl --train-labels $L/labels-ks-train.jsonl \
  --out-dir /data4/tskim/semcommit/runs/v0.2-r1 --epochs 3 --lr 3e-5 --lr-adapter 1e-4 --warmup 50 --bs-en 6 --bs-ko 16 \
  --save-every 300 --save-total-limit 0 --log-every 10 --num-workers 6 --gpu 0 "$@"
rc=$?; echo "[$(date "+%F %T")] TRAIN_EXIT rc=$rc"
