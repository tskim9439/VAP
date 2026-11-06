#!/bin/bash
# 2026-11-06 E15b(평가 묶음 q17-e15b — 활동 = config 의 N3D, SUBSET 빠른 평가): 창·회의 통째 평가가 끝나면 표준 재채점·출력 재귀속(창 3,1)·agn 분해(E13 · N3D 와)·
# Parakeet + N3D 대비 격차 진단(gap_diag — 재귀속 전·후)을 돌고, 단계마다 e15a_report.py 로 E13·E15a 절제와 한 표를 다시 쓴다. 끝에 대화 번역을 기다린다.
# 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
#   bash auto_e15b.sh [RUN(기본 q17-e15b)] [SUBSET(기본 eval — 빈 값이면 없음)]   → VAPKT-data/tmp/e15b-check/report-<RUN>.txt · gap-<RUN>.txt · auto-e15b.log
cd /soundai/users/tskim/VAPKT
RUN=${1:-q17-e15b}; SUB=${2-eval}; ST=${SUB:+-sub$SUB}
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e15b-check; R=$D/results; X=raw/sources/experiments; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T $D/tmp/agn
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
L=$T/auto-e15b.log; say() { echo "[$(date '+%F %T')] $*" >> $L; }
RR=q17-e15a-enc:eval,$RUN:$SUB:n3d; OUT=$T/report-$RUN.txt; CK=/soundai/Model/VAPASR/stage3-$RUN/final
for KN in "ami stage3-eval-ami-d10$ST-$RUN-final" "amifull stage3-eval-amifull-c30$ST-d10-$RUN-final"; do
  set -- $KN; K=$1; N=$2
  until [ -f $R/$N/summary.json ]; do sleep 300; done; say "시작 $N"
  bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
  if [ $K = amifull ]; then
    bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore.log 2>&1
    $P $X/2026-11-04-agn/agn_diag.py $R/$N $R/stage3-eval-amifull-c30-d10-q17-e13-final --json $D/tmp/agn/agn-$RUN.json > $D/tmp/agn/agn-$RUN.txt 2>&1
    for rl in "" "--relabel 3,1"; do                                                              # 격차 진단: E15b · E13 · N3D 대 Parakeet + N3D(같은 참조·셈, 시각 없이)
      $P $X/2026-11-04-agn/gap_diag.py --data $D/data/ami-bench-v1 --tok $CK $rl --sys "$RUN=ours:$R/$N" --sys "Parakeet+N3D=ext:$R/ext-ami-mtp-n3d" ${SUB:+--subset $SUB} \
        2>&1 | grep -vE "^WARNING|regex pattern|order of word-level" >> $T/gap-$RUN.txt
    done
  fi
  say "AUTO_DONE $N"
  python3 $X/2026-11-04-agn/e15a_report.py $RR > $OUT 2>&1
done
O=$R/stage4-eval-$RUN-final
until { grep -q "^DONE rank=" $O/dlg.log 2>/dev/null && [ $O/dlg-summary.log -nt $O/dlg.log ]; } || [ -f $O/dlg8/summary.json ]; do sleep 300; done; sleep 120
python3 $X/2026-11-04-agn/e15a_report.py $RR > $OUT 2>&1
say "ALL_DONE"
