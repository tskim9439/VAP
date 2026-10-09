#!/usr/bin/env bash
# Monotonic AST targets, GPU stage (experiments/ast_mono_generate.py run): Qwen3.8-27B on vLLM 0.30.0, one engine per GPU,
# four directions over the prepared source units. Every node first unpacks the vLLM env tar to its local /tmp
# (scripts/setup-vllm-env.sh stage), then runs 8 workers. Preemption or the time limit requeues the job; finished
# work items (gen/<direction>/u-NNNNN.jsonl.gz) are skipped on restart. Submit with slurm/submit-ast-mono-generate.sh
# (inputs via sbatch --export only).
#SBATCH --job-name=SA_EVAL_Fullduplex
#SBATCH --partition=hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=slurmmxch200v5-hpc-52,slurmmxch200v5-hpc-4
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail
PROJECT=${PROJECT:-/soundai/users/tskim/VAPKT}
ROOT=${ROOT:?ROOT 를 --export 로 주세요}
VLLM_ENV=${VLLM_ENV:-/tmp/sa_tskim/vllm-0.30}
cd "$PROJECT"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) restart=${SLURM_RESTART_COUNT:-0} ROOT=$ROOT"
test -f "$ROOT/units/manifest.json"
test "${SLURM_NTASKS:?}" -eq "$((SLURM_JOB_NUM_NODES * 8))"
export VLLM_ENV
srun --export=ALL --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  /bin/bash scripts/setup-vllm-env.sh stage 2>&1 | sed "s/^/  [stage] /"
# Triton (vLLM kernels) builds its launcher stubs with the system C compiler at run time: stop before loading models if absent.
srun --export=ALL --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  /bin/bash -c 'cc=$(command -v gcc || command -v cc) || { echo "!! C 컴파일러 없음: $(hostname)"; exit 3; }; echo "  [cc] $(hostname) $cc"'
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0          # greedy only; its JIT build needs ninja/nvcc on the node (2026-10-09)
export PATH="$VLLM_ENV/bin:$PATH"
# Never override CUDA_VISIBLE_DEVICES: Slurm supplies the allocated device. Caches per local rank on node-local disk.
set +e
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=0 --label \
    /bin/bash -c 'C=/tmp/sa_tskim/cache/r$SLURM_LOCALID; mkdir -p $C;
      VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton XDG_CACHE_HOME=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor \
      exec "$VLLM_ENV/bin/python" -u experiments/ast_mono_generate.py run --root "$ROOT"'
worker_rc=$?
"$VLLM_ENV/bin/python" experiments/ast_mono_generate.py status --root "$ROOT"
set -e
exit "$worker_rc"
