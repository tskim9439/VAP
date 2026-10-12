#!/usr/bin/env bash
# 단일 발화 평가(experiments/eval_single_turn_asr.py run)를 노드 GPU 8 장에 나눠 돌린다 — 호스트 GPU 가 모자랄 때(2026-10-12).
# TASKS 파일 한 줄 = "<manifest> <out> <world> <rank>"; srun task p 는 줄 번호 i % SLURM_NTASKS == p 인 줄을 차례로 한다.
# 설정은 호스트 대기열(evalq)과 같다: δ $DELTAS + final, batch $BATCH, 꼬리 무음 1 s, --verify, fp32.
# 선점으로 다시 시작하면 run 이 rank 별 예측 파일에서 이어 간다. 끝나면 out 마다 summarize(미완료면 summary.log 에 남김).
# 제출: slurm/submit-eval-q06-s3-ast.sh (입력은 sbatch --export 로만).
#SBATCH --job-name=SA_EVAL_Fullduplex
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
PY=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}
CK=${CK:?CK(체크포인트)를 --export 로 주세요}
TASKS=${TASKS:?TASKS(작업 목록 파일)를 --export 로 주세요}
DELTAS=${DELTAS:-4 6 8 10}
BATCH=${BATCH:-64}
export DELTAS BATCH CK TASKS PY
cd "$PROJECT"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} CK=$CK tasks=$(wc -l < "$TASKS") δ=[$DELTAS]"
test -f "$CK/config.json"
export PYTHONPATH="$PROJECT" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export VAPASR_STAGE_DIR=/dev/shm/tskim-vapasr-infer-eval
# Never override CUDA_VISIBLE_DEVICES: Slurm supplies the allocated device.
set +e
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=single:1 --label bash -c '
  i=0; rc=0
  while read -r man out world rank; do
    if (( i % SLURM_NTASKS == SLURM_PROCID )); then
      mkdir -p "$out"
      echo "[$(date +%T)] task $SLURM_PROCID → $out rank $rank/$world"
      "$PY" experiments/eval_single_turn_asr.py run --manifest "$man" --checkpoint "$CK" \
        --encoder /soundai/Model/nemotron-3.5-asr-streaming-0.6b --out "$out" --deltas $DELTAS --tail-s 1.0 --max-flush 8 \
        --batch-size "$BATCH" --rank "$rank" --world-size "$world" --limit 0 --verify --final >> "$out/rank$rank.log" 2>&1 || rc=1
    fi
    i=$((i + 1))
  done < "$TASKS"
  exit $rc'
worker_rc=$?
for out in $(awk "{print \$2}" "$TASKS" | sort -u); do
  "$PY" experiments/eval_single_turn_asr.py summarize --out "$out" > "$out/summary.log" 2>&1 \
    && echo "summarized $out" || echo "SUMMARY INCOMPLETE $out"
done
set -e
exit "$worker_rc"
