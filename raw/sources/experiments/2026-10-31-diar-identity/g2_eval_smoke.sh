#!/usr/bin/env bash
# G2 평가 쪽 스모크(2026-10-31): 스모크 graft.pt(학생 N3D LoRA 포함)로 diar_eval run --graft — AMI SDM 창 8 개와 회의 통째 1 개(--save-probs). 산출 tmp/e11-check/g2-eval-smoke-*.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python; G=$T/g2-smoke/final/graft.pt
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$1
$P experiments/diar_eval.py run --graft $G --out $T/g2-eval-smoke-win --corpora ami-sdm --limit-windows 8 > $T/g2-eval-smoke-win.log 2>&1
$P experiments/diar_eval.py summarize --out $T/g2-eval-smoke-win >> $T/g2-eval-smoke-win.log 2>&1
$P experiments/diar_eval.py run --graft $G --session --save-probs --out $T/g2-eval-smoke-ses --corpora ami-sdm --limit-windows 1 > $T/g2-eval-smoke-ses.log 2>&1
echo "G2_EVAL_SMOKE_DONE exit $? $(date +%T)" >> $T/g2-smoke-done.txt
