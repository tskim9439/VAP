#!/bin/bash
# Phase 2 Q0 — 라벨 품질 점검(전량): 발화별 Qwen3-ASR 대조 CER/WER → $OUT/asrcheck/<corpus>.json + .utts.jsonl (p2_refine --asr-flags 입력)
#   제출:  sbatch --partition=apex --export=ALL,CORPUS=aihub71631 slurm/p2_asr_check.sbatch     (LIMIT 로 표본 상한, 기본 전량; GPU 8 장 샤딩, 각 워커는 GPU 메모리 80 % 상한으로 배치 자동 보정)
#SBATCH --job-name=SA_SFT_FullDuplexStage5-p2asrcheck
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
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}; C=${CORPUS:?CORPUS=aihub71631 등}; D="$OUT/$C.dialogues.jsonl"
[ -f "$D" ] || { echo "!! $D 없음 — slurm/p2_build_dialogues.sbatch 먼저"; exit 1; }
echo "[$(date '+%F %T')] asr check $C ← $D"
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); W=${WORKERS_PER_GPU:-6}   # 실측(H200, 2026-09-15): 워커당 peak 정렬 ≤19 GB·ASR ≤13 GB → 6 워커 합계 ≈80 GB(53 %), 프로세스 상한 0.8/6=13 %; N=$((NG * W)); FR=$(python -c "print(round(0.8/$W, 3))"); mkdir -p "$OUT/asrcheck"
echo "  $NG GPUs × $W workers = $N shards, per-process mem cap $FR"
for k in $(seq 0 $((NG - 1))); do for w in $(seq 0 $((W - 1))); do i=$((k * W + w))
  CUDA_VISIBLE_DEVICES=$k python experiments/p2_asr_check.py --dialogues "$D" --out "$OUT/asrcheck/$C.json" --limit-utts ${LIMIT:-100000000} --shard "$i/$N" --gpu 0 --mem-frac $FR > "logs/train/p2asr-$C-$SLURM_JOB_ID-$i.log" 2>&1 &
done; done
wait
python - "$OUT/asrcheck" "$C" <<'EOF'
import sys, os, json, glob
d, c = sys.argv[1:3]; rows = []
for f in sorted(glob.glob(os.path.join(d, f"{c}.shard*.utts.jsonl"))): rows += [json.loads(l) for l in open(f, encoding="utf-8")]
with open(os.path.join(d, f"{c}.utts.jsonl"), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
errs = [r["err"] for r in rows]; n = len(errs)
rep = dict(corpus=c, n=n, err_median=sorted(errs)[n // 2] if n else None, share_ge_0_2=sum(e >= 0.2 for e in errs) / max(1, n), share_ge_0_3=sum(e >= 0.3 for e in errs) / max(1, n), shards=len(glob.glob(os.path.join(d, f"{c}.shard*.json"))))
json.dump(rep, open(os.path.join(d, f"{c}.json"), "w"), ensure_ascii=False, indent=1); print("merged:", rep)
EOF
echo "[$(date '+%F %T')] asr check done"; echo EXIT=0
