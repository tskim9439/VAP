#!/bin/bash
# 2026-11-03 E13 추론 게이트 판(묶음 q17-e13-gate): 판마다 회의 통째 평가가 끝나면(summary.json) 표준 재채점·출력 재귀속(창 3,1)·활동 추종 세부·게이트 진단을 돌고,
# 끝에 gate_report.py 로 E11·E13 과 견준다. 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
#   bash auto_gate.sh [판 태그 …(기본 gt03 gt01 gl0)] — 판은 제출한 것만(끝나지 않는 판이 있으면 그 자리에서 기다린다)
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e13-gate; R=$D/results; X=raw/sources/experiments; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
TAGS=("$@"); [ ${#TAGS[@]} -gt 0 ] || TAGS=(gt03 gt01 gl0)
for t in "${TAGS[@]}"; do
  N=stage3-eval-amifull-c30-$t-d10-q17-e13-final
  until [ -f $R/$N/summary.json ]; do sleep 300; done
  echo "[$(date '+%F %T')] 시작 $N" >> $T/auto-gate.log
  bash $X/2026-10-27-std-rescore/rescore_std.sh amifull $N 30 > $T/rescore-$N.log 2>&1
  bash $X/2026-10-30-ami-ext/relabel_rescore.sh amifull $N 3,1 >> $T/relabel-rescore.log 2>&1
  bash $X/2026-10-30-ami-ext/follow_detail.sh amifull $N >> $T/follow-detail.log 2>&1
  $P $X/2026-11-03-gate/gate_diag.py $R/$N --json $T/diag-$N.json > $T/diag-$N.txt 2>&1
  echo "[$(date '+%F %T')] AUTO_DONE $N" >> $T/auto-gate.log
done
python3 $X/2026-11-03-gate/gate_report.py "${TAGS[@]}" > $T/gate-report.txt 2>&1
echo "[$(date '+%F %T')] ALL_DONE" >> $T/auto-gate.log
