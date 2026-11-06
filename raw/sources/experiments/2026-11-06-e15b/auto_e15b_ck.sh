#!/bin/bash
# 2026-11-06 E15b 중간 체크포인트 평가(사용자 "E15b 현재까지 학습된 가장 최근에 저장된 체크포인트로 먼저 평가 수행해보자" — TARGET=checkpoint-N 묶음, 활동 = config 의 N3D,
# SUBSET 빠른 평가)가 끝나면 auto_e15b.sh 와 같은 분석을 그 태그로 돈다: 표준 재채점·출력 재귀속(창 3,1)·agn 분해(E13 · N3D 와)·Parakeet + N3D 대비 격차 진단
# (우리 정규화·Whisper 정규화, 재귀속 전·후)·오류 예산(err_budget — Whisper + 재귀속), 단계마다 e15a_report.py 로 E13·E15a 절제와 한 표. 끝에 대화 번역을 기다린다.
# 돌고 있는 auto_e15b.sh(final)를 고치지 않으려고 따로 둔다. 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
#   bash auto_e15b_ck.sh <태그 ck<N>> [RUN(기본 q17-e15b)] [SUBSET(기본 eval)]   → VAPKT-data/tmp/e15b-check/{report,gap,budget}-<RUN>-<태그>.txt · auto-e15b-<태그>.log
cd /soundai/users/tskim/VAPKT
TAG=${1:?태그 ck<N>}; [[ $TAG =~ ^ck[0-9]+$ ]] || { echo "태그는 ck<N>: $TAG"; exit 2; }
RUN=${2:-q17-e15b}; SUB=${3-eval}; ST=${SUB:+-sub$SUB}
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e15b-check; R=$D/results; X=raw/sources/experiments; P=$D/conda/envs/vapasr/bin/python; mkdir -p $T $D/tmp/agn
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
L=$T/auto-e15b-$TAG.log; say() { echo "[$(date '+%F %T')] $*" >> $L; }
RR=q17-e15a-enc:eval,$RUN:$SUB:n3d:$TAG; OUT=$T/report-$RUN-$TAG.txt; CK=/soundai/Model/VAPASR/stage3-$RUN/checkpoint-${TAG#ck}
G=$T/gap-$RUN-$TAG.txt; B=$T/budget-$RUN-$TAG.txt
for KN in "ami stage3-eval-ami-d10$ST-$RUN-$TAG" "amifull stage3-eval-amifull-c30$ST-d10-$RUN-$TAG"; do
  set -- $KN; K=$1; N=$2
  until [ -f $R/$N/summary.json ]; do sleep 300; done; say "시작 $N"
  bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
  if [ $K = amifull ]; then
    bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore-$TAG.log 2>&1
    $P $X/2026-11-04-agn/agn_diag.py $R/$N $R/stage3-eval-amifull-c30-d10-q17-e13-final --json $D/tmp/agn/agn-$RUN-$TAG.json > $D/tmp/agn/agn-$RUN-$TAG.txt 2>&1
    for nm in ours whisper; do for rl in "" "3,1"; do                                             # 격차 진단: 같은 참조·셈, 시각 없이 — whisper 는 표준 cpWER 와 같은 정규화
      J=$T/gap-$RUN-$TAG-$nm${rl:+-relabel}.json
      echo "=== 정규화 $nm${rl:+ · 출력 재귀속 $rl}" >> $G
      $P $X/2026-11-04-agn/gap_diag.py --data $D/data/ami-bench-v1 --tok $CK --norm $nm ${rl:+--relabel $rl} --sys "$RUN@$TAG=ours:$R/$N" --sys "Parakeet+N3D=ext:$R/ext-ami-mtp-n3d" \
        ${SUB:+--subset $SUB} --json $J 2>&1 | grep -vE "^WARNING|regex pattern|order of word-level" >> $G
    done; done
    python3 $X/2026-11-04-agn/err_budget.py $T/gap-$RUN-$TAG-whisper-relabel.json > $B 2>&1                # 오류 예산(표준 잣대 + 재귀속 — wiki 10 절과 같은 셈)
  fi
  say "AUTO_DONE $N"
  python3 $X/2026-11-04-agn/e15a_report.py $RR > $OUT 2>&1
done
O=$R/stage4-eval-$RUN-$TAG
until { grep -q "^DONE rank=" $O/dlg.log 2>/dev/null && [ $O/dlg-summary.log -nt $O/dlg.log ]; } || [ -f $O/dlg8/summary.json ]; do sleep 300; done; sleep 120
python3 $X/2026-11-04-agn/e15a_report.py $RR > $OUT 2>&1
say "ALL_DONE"
