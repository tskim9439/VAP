#!/usr/bin/env bash
# 2026-11-03 E13 게이트 진단(서버 호스트 CPU, 디코드 없음 — gate_diag.py): E13 과 E11(-follow — act_runs)의 회의 통째·창 원출력을 같은 기준으로 나눈다.
# 산출 tmp/e13-gate/diag-<이름>.{txt,json}(새 폴더). 원 결과 폴더는 읽기만 한다. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data; R=$D/results; T=$D/tmp/e13-gate; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
echo "[$(date '+%F %T')] 시작" > $T/gate-diag.log
for N in stage3-eval-amifull-c30-d10-q17-e13-final stage3-eval-amifull-c30-d10-q17-e11-final-follow stage3-eval-ami-d10-q17-e13-final stage3-eval-ami-d10-q17-e11-final-follow; do
  $P raw/sources/experiments/2026-11-03-gate/gate_diag.py $R/$N --json $T/diag-$N.json > $T/diag-$N.txt 2>&1
  echo "[$(date '+%F %T')] $N exit $?" >> $T/gate-diag.log
done
echo "[$(date '+%F %T')] GATE_DIAG_DONE" >> $T/gate-diag.log
