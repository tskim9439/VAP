#!/bin/bash
# Phase 2 Q0 — 라벨 품질 점검(전량): 발화별 Qwen3-ASR 대조 CER/WER → $OUT/asrcheck/<corpus>.json + .utts.jsonl (p2_refine --asr-flags 입력)
#   제출:  sbatch --partition=apex --export=ALL,CORPUS=aihub71631 slurm/p2_asr_check.sbatch     (LIMIT 로 표본 상한, 기본 전량)
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2asrcheck
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}; C=${CORPUS:?CORPUS=aihub71631 등}; D="$OUT/$C.dialogues.jsonl"
[ -f "$D" ] || { echo "!! $D 없음 — slurm/p2_build_dialogues.sbatch 먼저"; exit 1; }
echo "[$(date '+%F %T')] asr check $C ← $D"
python experiments/p2_asr_check.py --dialogues "$D" --out "$OUT/asrcheck/$C.json" --limit-utts ${LIMIT:-100000000} --gpu 0
echo "[$(date '+%F %T')] asr check done"; echo EXIT=0
