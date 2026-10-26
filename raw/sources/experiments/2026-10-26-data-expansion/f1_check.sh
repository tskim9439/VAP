# F1 점검(mxc 호스트, 2026-10-26): slurm/submit-q17-f1.sh 가 낼 stage3_train.py 인자 그대로(--dry-run 출력의 ARGS_EXTRA) 학습 dry-run(CPU — 데이터 구성·시퀀스 불변식)
#   → SMOKE=1 이면 가장 빈 호스트 GPU 로 20 step 스모크. bash f1_check.sh [--allow-missing]. 산출 /tmp/sa_tskim/f1-{dry,smoke}-train — 지우지 않는다(있으면 이어 쓴다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
X=$(bash slurm/submit-q17-f1.sh --dry-run --no-eval "$@" 2>&1 | grep -o "ARGS_EXTRA=.*slurm/stage3-train-apex.sbatch" | sed -e "s/^ARGS_EXTRA=//" -e "s/ slurm\/stage3-train-apex.sbatch$//" -e "s/%2C/,/g")
[ -n "$X" ] || { echo "ARGS_EXTRA 를 못 읽었다"; exit 1; }
echo "[$(date "+%F %T")] f1 dry-run"; echo "  인자: ${X:0:300}…"
eval "$P experiments/stage3_train.py --dry-run --nemotron-dir $MXC_NEMOTRON_DIR --out-dir /tmp/sa_tskim/f1-dry-train --check-items 32 --max-windows 400 $X" 2>&1 | grep -v "regex pattern"
echo "DRY EXIT ${PIPESTATUS[0]}"
[ "${SMOKE:-0}" = 1 ] || { echo "F1_CHECK_DONE"; exit 0; }
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F", " "{print \$1, \$3 - \$2, \$4}" | sort -k2,2nr -k3,3n | head -1 | cut -d" " -f1); echo "GPU $GPU"
eval "$P experiments/stage3_train.py --gpu $GPU --nemotron-dir $MXC_NEMOTRON_DIR --out-dir /tmp/sa_tskim/f1-smoke-train --check-items 8 --max-windows 400 $X --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --report-to none" 2>&1 | grep -v "regex pattern"
echo "SMOKE EXIT ${PIPESTATUS[0]}"; echo "F1_CHECK_DONE"
