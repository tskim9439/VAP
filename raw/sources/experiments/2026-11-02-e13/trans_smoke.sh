#!/usr/bin/env bash
# 2026-11-02 E13 번역 평가 경로 스모크(호스트 GPU): cst_stage4_eval 슬롯 블록 번역 디코드를 E11 final(실제 모델)로 held-out 대화 창 몇 개에 — v3_eval_task.sh dlg 와 같은 인자,
#   코퍼스 nsc·otoSpeech 각 2 창, δ10. bash trans_smoke.sh <GPU>. 산출 tmp/e13-check/trans-smoke(새 폴더). 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e13-check; P=$D/conda/envs/vapasr/bin/python; P2=$D/data/phase2; S1=$D/data/stage4-dlg-units-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=$1
mkdir -p $T
$P experiments/cst_stage4_eval.py run --checkpoint /soundai/Model/VAPASR/stage3-q17-e11/final --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 \
  --corpora nsc,otoSpeech --deltas 10 --limit-windows 2 --trans-lead-space --batch-size 4 --backchannel 1.0,3 --out $T/trans-smoke > $T/trans-smoke.log 2>&1
$P experiments/cst_stage4_eval.py summarize --out $T/trans-smoke >> $T/trans-smoke.log 2>&1
echo "TRANS_SMOKE_DONE $? $(date +%T)" >> $T/trans-smoke.log
