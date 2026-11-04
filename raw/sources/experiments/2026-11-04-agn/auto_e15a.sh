#!/bin/bash
# 2026-11-04 E15a(평가 묶음 q17-e15a — 활동 = G2b 학생, 종단간): 창·회의 통째 평가가 끝나면 표준 재채점·출력 재귀속(창 3,1)·활동 추종 세부·게이트 진단·agn 분해를 돌고,
# 단계마다 e15a_report.py 로 E13 과 견준 표를 다시 쓴다. 끝에 번역(dlg.log DONE)을 기다린다. 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
#   bash auto_e15a.sh [RUN(기본 q17-e15a)] [GRAFT_RUN(기본 q17-graft-g2b)]   (2026-11-05: 절제 run 은 표에 q17-e15a 와 함께, agn·표 파일은 run 마다)
cd /soundai/users/tskim/VAPKT
RUN=${1:-q17-e15a}; GR=${2:-q17-graft-g2b}
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e15a-check; R=$D/results; X=raw/sources/experiments; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T $D/tmp/agn $D/tmp/e13-gate
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
L=$T/auto-e15a.log; say() { echo "[$(date '+%F %T')] $*" >> $L; }
RR=$RUN; [ $RUN != q17-e15a ] && RR=q17-e15a,$RUN; OUT=$T/report-$RUN.txt; [ $RUN = q17-e15a ] && OUT=$T/e15a-report.txt
for KN in "ami stage3-eval-ami-d10-gx$GR-$RUN-final" "amifull stage3-eval-amifull-c30-gx$GR-d10-$RUN-final"; do
  set -- $KN; K=$1; N=$2
  until [ -f $R/$N/summary.json ]; do sleep 300; done; say "시작 $N"
  bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
  if [ $K = amifull ]; then
    bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore.log 2>&1
    bash $X/2026-10-30-ami-ext/follow_detail.sh $K $N >> $T/follow-detail.log 2>&1
    $P $X/2026-11-03-gate/gate_diag.py $R/$N --json $D/tmp/e13-gate/diag-$N.json > $D/tmp/e13-gate/diag-$N.txt 2>&1
    $P $X/2026-11-04-agn/agn_diag.py $R/$N $R/stage3-eval-amifull-c30-enc$GR-gx$GR-d10-q17-e13-final --json $D/tmp/agn/agn-$RUN.json > $D/tmp/agn/agn-$RUN.txt 2>&1
  fi
  say "AUTO_DONE $N"
  python3 $X/2026-11-04-agn/e15a_report.py $RR $GR > $OUT 2>&1
done
O=$R/stage4-eval-$RUN-final
until grep -q "^DONE rank=" $O/dlg.log 2>/dev/null && [ $O/dlg-summary.log -nt $O/dlg.log ]; do sleep 300; done; sleep 120   # 요약 쓰기가 끝나게
python3 $X/2026-11-04-agn/e15a_report.py $RR $GR > $OUT 2>&1
say "ALL_DONE"
