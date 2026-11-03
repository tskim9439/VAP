#!/bin/bash
# 2026-11-03 G2b 학생과 종단간 G3(학습 99503 → 학생 평가 99504·99505, G3 평가 99712 E13 + G2b · 99713 E11 + G2b): 끝나는 대로 표준 재채점(창·회의 통째)·
# 출력 재귀속(창 3,1)·활동 추종 세부·게이트 진단을 돌고, 끝에 g3b_report.py 로 한 표. 서버 호스트 CPU, 디코드 없음. 원 폴더는 읽기만 하고 새 폴더에만 쓴다. 아무것도 지우지 않는다.
#   bash auto_g3b.sh [GRAFT_RUN(기본 q17-graft-g2b)]
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e13-gate; R=$D/results; X=raw/sources/experiments; P=$D/conda/envs/vapasr/bin/python; GR=${1:-q17-graft-g2b}; mkdir -p $T
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
L=$T/auto-g3b.log; say() { echo "[$(date '+%F %T')] $*" >> $L; }
W0=stage3-eval-ami-d10-enc$GR-q17-e11-final                                                     # ASR 지킴(99505)
until [ -f $R/$W0/summary.json ]; do sleep 300; done; say "시작 $W0"
[ -f $R/$W0-wn/meeteval-whisper.txt ] || bash $X/2026-10-27-std-rescore/rescore_std.sh ami $W0 30 > $T/rescore-$W0.log 2>&1; say "AUTO_DONE $W0"
for run in q17-e13 q17-e11; do
  for KN in "ami stage3-eval-ami-d10-enc$GR-gx$GR-$run-final" "amifull stage3-eval-amifull-c30-enc$GR-gx$GR-d10-$run-final"; do
    set -- $KN; K=$1; N=$2
    until [ -f $R/$N/summary.json ]; do sleep 300; done; say "시작 $N"
    bash $X/2026-10-27-std-rescore/rescore_std.sh $K $N 30 > $T/rescore-$N.log 2>&1
    if [ $K = amifull ]; then
      bash $X/2026-10-30-ami-ext/relabel_rescore.sh $K $N 3,1 >> $T/relabel-rescore.log 2>&1
      bash $X/2026-10-30-ami-ext/follow_detail.sh $K $N >> $T/follow-detail.log 2>&1
      $P $X/2026-11-03-gate/gate_diag.py $R/$N --json $T/diag-$N.json > $T/diag-$N.txt 2>&1
    fi
    say "AUTO_DONE $N"
  done
done
python3 $X/2026-11-03-gate/g3b_report.py $GR > $T/g3b-report.txt 2>&1
say "ALL_DONE"
