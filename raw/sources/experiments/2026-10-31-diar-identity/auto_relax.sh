#!/bin/bash
# 2026-11-02 E11 지연 완화(묶음 q17-e11-relax, job 99459): 평가가 끝나면(summary.json) 표준 재채점·(회의 통째) 출력 재귀속(창 3,1)·활동 추종 세부를 돌고,
# 끝에 relax_report.py 로 E11 과 견준다. 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT
T=/soundai/users/tskim/VAPKT-data/tmp/e11-check; R=/soundai/users/tskim/VAPKT-data/results; X=raw/sources/experiments
for KN in "ami stage3-eval-ami-d10-n3dc124-q17-e11-final" "ami stage3-eval-ami-d10-rc13-q17-e11-final" "ami stage3-eval-ami-d10-n3dc124-rc13-q17-e11-final" \
          "amifull stage3-eval-amifull-c30-n3dc124-rc13-d10-q17-e11-final"; do
  set -- $KN; K=$1; N=$2
  until [ -f $R/$N/summary.json ]; do sleep 120; done
  echo "[$(date '+%F %T')] 시작 $N" >> $T/auto-relax.log
  bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
  if [ $K = amifull ]; then
    bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore.log 2>&1
    bash $X/2026-10-30-ami-ext/follow_detail.sh $K $N >> $T/follow-detail.log 2>&1
  fi
  echo "[$(date '+%F %T')] AUTO_DONE $N" >> $T/auto-relax.log
done
python3 $X/2026-10-31-diar-identity/relax_report.py > $T/relax-report.txt 2>&1
echo "[$(date '+%F %T')] ALL_DONE" >> $T/auto-relax.log
