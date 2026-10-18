# Stage 4 평가 도구 실제 디코드 점검 + 출발점: Stage 3 최종(p11 ck3000)을 번역 prefix 로(합성 KO↔EN 시험 50 창). 호스트 GPU 하나(가장 빈 것).
# 산출 /tmp/sa_tskim/stage4-sim-xl-test2/eval-p11-* — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2}' | sort -k2,2nr | head -1 | cut -d' ' -f1)
echo "[$(date +%T)] GPU $GPU ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU))"; export CUDA_VISIBLE_DEVICES=$GPU
O=/tmp/sa_tskim/stage4-sim-xl-test2; CK=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000; ENC=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
$PY experiments/cst_stage4_eval.py run --checkpoint $CK --data $O --trans $O/targets-sim_ko-en --corpora sim_ko-en --deltas 4,10 --limit-windows 0 --batch-size 25 --out $O/eval-p11-xl 2>&1 | grep -vE "regex pattern|Warning" | tail -2
$PY experiments/cst_stage4_eval.py summarize --out $O/eval-p11-xl
echo "ZS_DONE"
