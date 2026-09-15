#!/bin/bash
# Phase 2 Q0 — dialogues.jsonl 의 발화를 화자 채널에서 Qwen3-ForcedAligner 로 정렬 (GPU). task-phase2-data-prep 순서 4.
#   제출:  sbatch --partition=apex --export=ALL,CORPUS=ami slurm/p2_align.sbatch          (GPU 8 → 워커 8 이 rows[k::8] 분담, 재개 가능)
#   출력:  $OUT/align-asr-tn-v1/<corpus>/parts/*.jsonl + stats-*.json
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2align
#SBATCH --partition=apex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" && mkdir -p "$TMPDIR"; fi
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}; C=${CORPUS:?CORPUS=ami 등}; D="$OUT/$C.dialogues.jsonl"
[ -f "$D" ] || { echo "!! $D 없음 — slurm/p2_build_dialogues.sbatch 먼저"; exit 1; }
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); W=${WORKERS_PER_GPU:-6}   # 실측(H200, 2026-09-15): 워커당 peak 정렬 ≤19 GB·ASR ≤13 GB → 6 워커 합계 ≈80 GB(53 %), 프로세스 상한 0.8/6=13 %; N=$((NG * W)); FR=$(python -c "print(round(0.8/$W, 3))")
echo "[$(date '+%F %T')] align $C: $NG GPUs × $W workers = $N shards, per-process mem cap $FR ← $D"
for k in $(seq 0 $((NG - 1))); do for w in $(seq 0 $((W - 1))); do i=$((k * W + w))
  CUDA_VISIBLE_DEVICES=$k python experiments/p2_align.py --dialogues "$D" --shard "$i/$N" --gpu 0 --mem-frac $FR --io-threads 4 > "logs/train/p2align-$C-$SLURM_JOB_ID-$i.log" 2>&1 &
done; done
wait
python - "$OUT/align-asr-tn-v1/$C" <<'EOF'
import sys, os, json, glob, collections
d=sys.argv[1]; st=collections.Counter()
for f in glob.glob(os.path.join(d,"stats-*.json")):
    for k,v in json.load(open(f)).items(): st[k]+= v if isinstance(v,(int,float)) else 0
print("align total:", dict(st))
EOF
echo "[$(date '+%F %T')] p2 align done"; echo EXIT=0
