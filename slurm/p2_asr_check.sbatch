#!/bin/bash
# Phase 2 Q0 — 라벨 품질 점검(전량): 발화별 Qwen3-ASR 대조 CER/WER → $OUT/asrcheck/<corpus>.json + .utts.jsonl (p2_refine --asr-flags 입력)
#   제출:  sbatch --partition=apex --export=ALL,CORPUS=all slurm/p2_asr_check.sbatch     (노드 1 개: AI Hub 3 종을 차례로, 코퍼스마다 GPU 8 × 워커 6 = 48 샤드; LIMIT 로 표본 상한)
#          CORPUS=aihub71631 (하나) / CORPUS="aihub71631 otoSpeech" (여러 개 순차). 정렬 잡 뒤에 이어 돌리려면 --dependency=afterany:<정렬 잡 ID>
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
export VAPASR_LOCAL_ENV=${LOCAL_ENV-/tmp/sa_tskim/vapasr-env}
if [ -n "$VAPASR_LOCAL_ENV" ]; then bash scripts/stage-env-local.sh 2>&1 | grep -vE "^cp: .*File exists" | sed "s/^/  [stage] /"
  [ -x "$VAPASR_LOCAL_ENV/bin/python" ] && export PATH="$VAPASR_LOCAL_ENV/bin:$PATH" CONDA_PREFIX="$VAPASR_LOCAL_ENV" TMPDIR="$(dirname "$VAPASR_LOCAL_ENV")/tmp" && mkdir -p "$TMPDIR"; fi   # NFS conda 에서 torch·transformers import 가 69 s(실측) → 노드 로컬 /tmp 사본으로 1 회 스테이징
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
OUT=${OUT:-/soundai/users/tskim/VAPKT-data/data/phase2}
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); W=${WORKERS_PER_GPU:-6}; N=$((NG * W)); FR=$(python -c "print(round(0.8/$W, 3))"); mkdir -p "$OUT/asrcheck"   # 실측(H200): 워커당 peak ASR ≤13 GB → 6 워커 합계 ≈70 GB
if [ "${CORPUS:?CORPUS=aihub71631 | all}" = all ]; then LIST="aihub71631 aihub134-1 aihub134-2"; else LIST="$CORPUS"; fi
for C in $LIST; do
  D="$OUT/$C.dialogues.jsonl"
  [ -f "$D" ] || { echo "!! $D 없음 — slurm/p2_build_dialogues.sbatch 먼저"; continue; }
  echo "[$(date '+%F %T')] asr check $C: $NG GPUs × $W workers = $N shards, per-process mem cap $FR ← $D"
  for k in $(seq 0 $((NG - 1))); do for w in $(seq 0 $((W - 1))); do i=$((k * W + w))
    CUDA_VISIBLE_DEVICES=$k python experiments/p2_asr_check.py --dialogues "$D" --out "$OUT/asrcheck/$C.json" --limit-utts ${LIMIT:-100000000} --shard "$i/$N" --gpu 0 --mem-frac $FR > "logs/train/p2asr-$C-$SLURM_JOB_ID-$i.log" 2>&1 &
  done; done
  wait
  python - "$OUT/asrcheck" "$C" <<'PY'
import sys, os, json, glob
d, c = sys.argv[1:3]; rows = []
for f in sorted(glob.glob(os.path.join(d, f"{c}.shard*.utts.jsonl"))): rows += [json.loads(l) for l in open(f, encoding="utf-8")]
with open(os.path.join(d, f"{c}.utts.jsonl"), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
errs = [r["err"] for r in rows]; n = len(errs)
rep = dict(corpus=c, n=n, err_median=sorted(errs)[n // 2] if n else None, share_ge_0_2=sum(e >= 0.2 for e in errs) / max(1, n), share_ge_0_3=sum(e >= 0.3 for e in errs) / max(1, n), shards=len(glob.glob(os.path.join(d, f"{c}.shard*.json"))))
json.dump(rep, open(os.path.join(d, f"{c}.json"), "w"), ensure_ascii=False, indent=1); print("merged:", rep)
PY
  echo "[$(date '+%F %T')] asr check $C done"
done
echo "[$(date '+%F %T')] asr check done"; echo EXIT=0
