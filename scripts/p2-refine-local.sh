#!/usr/bin/env bash
# Phase 2 보정을 SLURM 없이 mxc 로그인 노드에서 실행 (사용자 지시 2026-09-16: 보정은 CPU 작업이라 SLURM 에 내지 않는다).
#   nohup bash scripts/p2-refine-local.sh <정렬 잡 ID> <ASR 대조 잡 ID> > logs/train/p2refine-local.log 2>&1 &
# 코퍼스마다 정렬 로그에 "align <c> done", (AI Hub 는) ASR 로그에 "asr check <c> done" 이 찍히면 그 코퍼스의 보정을 백그라운드로 시작한다.
# 정렬만 있고 ASR 대조가 없는 코퍼스는 대조 없이 보정한다. 이미 보정본이 있으면 건너뛴다.
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
ALIGN_JOB=${1:?정렬 잡 ID}; ASR_JOB=${2:-}; OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}
AL=logs/train/SA_SFT_FullDuplexStage5-p2align-$ALIGN_JOB.out; AS=logs/train/SA_SFT_FullDuplexStage5-p2asrcheck-${ASR_JOB:-none}.out
ALL="aihub71631 aihub134-1 aihub134-2 otoSpeech ami notsofar icsi"; AIHUB="aihub71631 aihub134-1 aihub134-2"
declare -A started
echo "[$(date '+%F %T')] waiting: align log $AL, asr log $AS"
while :; do
  for c in $ALL; do
    [ -n "${started[$c]:-}" ] && continue
    [ -f "$OUT/$c.refined.dialogues.jsonl" ] && { echo "[$(date '+%F %T')] $c: 보정본 있음 → 건너뜀"; started[$c]=1; continue; }
    grep -q "align $c done" "$AL" 2>/dev/null || continue
    if [[ " $AIHUB " == *" $c "* ]] && [ -n "$ASR_JOB" ]; then grep -q "asr check $c done" "$AS" 2>/dev/null || continue; fi
    FL=""; [ -f "$OUT/asrcheck/$c.utts.jsonl" ] && FL="--asr-flags $OUT/asrcheck/$c.utts.jsonl"
    echo "[$(date '+%F %T')] refine $c 시작 ${FL:+(ASR 대조 사용)}"
    nohup python experiments/p2_refine.py --dialogues "$OUT/$c.dialogues.jsonl" --align "$OUT/align-asr-tn-v1/$c" $FL > "logs/train/p2refine-$c.log" 2>&1 &
    started[$c]=1
  done
  n=0; for c in $ALL; do [ -n "${started[$c]:-}" ] && n=$((n+1)); done
  [ $n -eq 7 ] && break
  sleep 120
done
wait; echo "[$(date '+%F %T')] 모든 보정 종료"; for c in $ALL; do tail -1 "logs/train/p2refine-$c.log" 2>/dev/null | cut -c1-300; done; echo EXIT=0
