#!/usr/bin/env bash
# G3 스모크(2026-11-01): E11 final + G2 학생(활동·인코더 같은 graft.pt) — AMI SDM 창 8 개, 회의 통째 1 개(--verify). 산출 tmp/e11-check/g3-smoke-*. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python; G=/soundai/Model/VAPASR/graft-q17-graft-g2-lora/final/graft.pt; CK=/soundai/Model/VAPASR/stage3-q17-e11/final
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=$1
$P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $D/data/ami-bench-v1 --corpora ami-sdm --limit-windows 8 --deltas 10 --batch-size 8 \
  --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --encoder-state $G --graft-expert $G --out $T/g3-smoke-win > $T/g3-smoke-win.log 2>&1
$P experiments/cst_stage3_eval.py summarize --out $T/g3-smoke-win >> $T/g3-smoke-win.log 2>&1
$P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $D/data/ami-bench-v1 --corpora ami-sdm --limit-windows 1 --deltas 10 \
  --backchannel 1.0,3 --max-speakers 4 --window full --context-s 30 --declare 4 --verify --encoder-state $G --graft-expert $G --out $T/g3-smoke-full > $T/g3-smoke-full.log 2>&1
$P experiments/cst_stage3_eval.py summarize --out $T/g3-smoke-full >> $T/g3-smoke-full.log 2>&1
echo "G3_SMOKE_DONE $(date +%T)" >> $T/g3-smoke-done.txt
