#!/bin/bash
# 2026-11-01 G3: 평가(창·회의 통째 — 묶음 q17-g3)가 끝나면(summary.json) 표준 재채점·출력 재귀속(창 3,1)·활동 추종 세부를 돌고, 끝에 g3_report.py 로 E11 과 견준다.
# G2 인코더 + N3D 창(ASR 지킴 평가)도 표준 재채점한다(표의 둘째 줄). 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
#   bash auto_g3.sh [GRAFT_RUN(기본 q17-graft-g2-lora)]
cd /soundai/users/tskim/VAPKT
T=/soundai/users/tskim/VAPKT-data/tmp/e11-check; R=/soundai/users/tskim/VAPKT-data/results; X=raw/sources/experiments; GR=${1:-q17-graft-g2-lora}
W0=stage3-eval-ami-d10-enc$GR-q17-e11-final
if [ -f $R/$W0/summary.json ] && [ ! -f $R/$W0-wn/meeteval-whisper.txt ]; then bash $X/2026-10-27-std-rescore/rescore_std.sh ami $W0 30 > $T/rescore-$W0.log 2>&1; fi
for KN in "ami stage3-eval-ami-d10-enc$GR-gx$GR-q17-e11-final" "amifull stage3-eval-amifull-c30-enc$GR-gx$GR-d10-q17-e11-final"; do
  set -- $KN; K=$1; N=$2
  until [ -f $R/$N/summary.json ]; do sleep 120; done
  echo "[$(date '+%F %T')] 시작 $N" >> $T/auto-g3.log
  bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
  bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore.log 2>&1
  bash $X/2026-10-30-ami-ext/follow_detail.sh $K $N >> $T/follow-detail.log 2>&1
  echo "[$(date '+%F %T')] AUTO_DONE $N" >> $T/auto-g3.log
done
python3 $X/2026-10-31-diar-identity/g3_report.py $GR > $T/g3-report.txt 2>&1
echo "[$(date '+%F %T')] ALL_DONE" >> $T/auto-g3.log
