#!/usr/bin/env bash
# 디코드 때 재귀속 평가(q17-dretag 묶음) 뒤 자동 보고(2026-11-05, 서버 호스트 CPU, 디코드 없음): 두 판(E13 · N3D, E13 + G2b)의 회의 통째 산출이 끝나면
# rescore_std.sh(Whisper 정규화 단어마다 + meeteval 화자 통째)를 돌리고, 원 디코드 · 사후 재귀속(-relabel3-1-wn) · 디코드 때 재귀속을 한 표로 쓴다(dretag_report.py).
#   bash auto_dretag.sh [B-F(기본 3-1)] [SUB(기본 eval — 산출 이름 -sub<SUB>, 빈 값이면 없음)]   → VAPKT-data/tmp/dretag/report.txt, auto.log
# 원 결과 폴더는 읽기만 하고 새 폴더(-wn)에만 쓴다. 아무것도 지우지 않는다.
set -u
W=${1:-3-1}; SUB=${2-eval}; ST=${SUB:+-sub$SUB}
cd /soundai/users/tskim/VAPKT
R=/soundai/users/tskim/VAPKT-data/results; T=/soundai/users/tskim/VAPKT-data/tmp/dretag; X=raw/sources/experiments; mkdir -p $T
L=$T/auto.log; say() { echo "[$(date '+%F %T')] $*" >> $L; }
G=encq17-graft-g2b-gxq17-graft-g2b-
for N in stage3-eval-amifull-c30-dr$W$ST-d10-q17-e13-final stage3-eval-amifull-c30-${G}dr$W$ST-d10-q17-e13-final; do
  until [ -f $R/$N/summary.json ]; do sleep 300; done; say "시작 $N"
  [ -f $R/$N-wn/meeteval-whisper.txt ] || bash $X/2026-10-27-std-rescore/rescore_std.sh amifull $N 30 >> $T/rescore.log 2>&1
  say "재채점 끝 $N"
done
python3 $X/2026-11-05-dretag/dretag_report.py $W "$ST" > $T/report.txt 2>&1
say "ALL_DONE"
