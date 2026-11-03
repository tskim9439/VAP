#!/bin/bash
# 2026-11-03 E14(학습 99729 → 평가 99730 q17-e14, 번역 q17-e14-trans): 창·회의 통째 평가가 끝나면 표준 재채점·출력 재귀속(창 3,1)·활동 추종 세부·게이트 진단을 돌고,
# 끝에 e14_report.py 로 E11·E13 과 견준다(번역은 결과가 있으면). 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
# 2026-11-03 23:10: 창 평가 몫 0·1 과 번역이 X 칸 <EOT> 로 멈춰 다시 낸다 — 이미 끝난 회의 통째를 먼저 돌고, 단계마다 보고를 새로 쓰고, 끝에 번역(dlg.log DONE)을 기다린다.
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e14-check; R=$D/results; X=raw/sources/experiments; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T $D/tmp/e13-gate
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
for KN in "amifull stage3-eval-amifull-c30-d10-q17-e14-final" "ami stage3-eval-ami-d10-q17-e14-final"; do
  set -- $KN; K=$1; N=$2
  until [ -f $R/$N/summary.json ]; do sleep 300; done
  echo "[$(date '+%F %T')] 시작 $N" >> $T/auto-e14.log
  bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
  bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore.log 2>&1
  bash $X/2026-10-30-ami-ext/follow_detail.sh $K $N >> $T/follow-detail.log 2>&1
  [ $K = amifull ] && $P $X/2026-11-03-gate/gate_diag.py $R/$N --json $D/tmp/e13-gate/diag-$N.json > $D/tmp/e13-gate/diag-$N.txt 2>&1
  echo "[$(date '+%F %T')] AUTO_DONE $N" >> $T/auto-e14.log
  python3 $X/2026-11-03-e14/e14_report.py > $T/e14-report.txt 2>&1
done
O=$R/stage4-eval-q17-e14-final
until grep -q "^DONE rank=" $O/dlg.log 2>/dev/null && [ $O/dlg-summary.log -nt $O/dlg.log ]; do sleep 300; done; sleep 120   # 요약 쓰기가 끝나게
python3 $X/2026-11-03-e14/e14_report.py > $T/e14-report.txt 2>&1
echo "[$(date '+%F %T')] ALL_DONE" >> $T/auto-e14.log
