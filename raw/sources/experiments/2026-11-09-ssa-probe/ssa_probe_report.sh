#!/bin/bash
# P1 평가 자동 보고(2026-11-09): results/ssa-probe-<RUN>-<판> 네 판(parakeet-n3d·probe-n3d·probe-oracle·probe-n3d-rc13)이 회의 32 개(16 × IHM-mix·SDM)를 다 내면,
#   판마다 gap_diag.py --norm whisper(같은 참조·셈, ext 행)로 겹침별 놓침·귀속을 재고 err_budget.py 로 유형별 몫을 낸다(둘째 시스템 = 회의 통째 스트리밍 Parakeet + N3D).
#   [VARIANTS="…"] bash ssa_probe_report.sh [RUN(기본 p1)] → VAPKT-data/tmp/ssa-probe/report-<RUN>.txt. 서버 호스트 CPU, 읽기만 하고 새 파일에만 쓴다. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT
RUN=${1:-p1}; D=/soundai/users/tskim/VAPKT-data; R=$D/results; T=$D/tmp/ssa-probe; mkdir -p $T; X=raw/sources/experiments
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
P=$D/conda/envs/vapasr/bin/python; L=$T/auto-$RUN.log; OUT=$T/report-$RUN.txt
say() { echo "[$(date '+%F %T')] $*" >> $L; }
V=${VARIANTS:-"parakeet-n3d probe-n3d probe-oracle probe-n3d-rc13"}                     # P2(2026-11-10): VARIANTS="probe-n3d probe-oracle probe-n3d-rc13"(Parakeet 판은 P1 것을 쓴다)
NV=$(echo $V | wc -w)
n() { cat $R/ssa-probe-$RUN-$1/predictions-rank*.jsonl 2>/dev/null | wc -l; }
until [ "$(for v in $V; do n $v; done | awk '$1 >= 32' | wc -l)" -eq $NV ]; do sleep 300; done; say "평가 끝 — 보고 시작"
: > $OUT
for v in $V; do
  J=$T/gap-$RUN-$v.json
  echo "=== $v" >> $OUT
  $P $X/2026-11-04-agn/gap_diag.py --data $D/data/ami-bench-v1 --tok /soundai/Model/VAPASR/stage3-q17-e13/final --norm whisper \
    --sys "$v=ext:$R/ssa-probe-$RUN-$v" --sys "Parakeet+N3D(회의 통째)=ext:$R/ext-ami-mtp-n3d" --subset eval --json $J 2>&1 | grep -vE "^WARNING|regex pattern|order of word-level" >> $T/gap-$RUN-$v.txt
  python3 $X/2026-11-04-agn/err_budget.py $J >> $OUT 2>&1
  say "끝 $v"
done
say "ALL_DONE"
