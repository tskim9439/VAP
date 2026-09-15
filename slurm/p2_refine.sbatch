#!/bin/bash
# Phase 2 Q0 — 정렬 병합 + 화자 채널 VAD 시각 보정 + 결손 의심/ASR 불일치 quarantine → <corpus>.refined.dialogues.jsonl (CPU)
#   제출:  sbatch --partition=apex --export=ALL,CORPUS=aihub71631 slurm/p2_refine.sbatch      (ASR 대조 결과 $OUT/asrcheck/<corpus>.utts.jsonl 이 있으면 자동 사용)
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2refine
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" && mkdir -p "$TMPDIR"; fi   # NFS conda 에서 torch·transformers import 가 69 s(실측) → 노드 로컬 /tmp 사본으로 1 회 스테이징
export OMP_NUM_THREADS=8
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}; C=${CORPUS:?CORPUS=aihub71631 등}
[ -f "$OUT/$C.dialogues.jsonl" ] || { echo "!! $OUT/$C.dialogues.jsonl 없음"; exit 1; }
[ -d "$OUT/align-asr-tn-v1/$C/parts" ] || { echo "!! 정렬 없음 — slurm/p2_align.sbatch 먼저"; exit 1; }
FL=""; [ -f "$OUT/asrcheck/$C.utts.jsonl" ] && FL="--asr-flags $OUT/asrcheck/$C.utts.jsonl"
echo "[$(date '+%F %T')] refine $C ${FL:+(ASR 대조 사용)}"
python experiments/p2_refine.py --dialogues "$OUT/$C.dialogues.jsonl" --align "$OUT/align-asr-tn-v1/$C" $FL
echo "[$(date '+%F %T')] refine done"; echo EXIT=0
