# Stage 4 TAXI 어댑터 점검 + 출발점: Stage 3 최종(p11 ck3000)을 교차 언어 prefix 로 TAXI natural 86 세션에. 호스트 GPU 하나(가장 빈 것).
# 산출 /soundai/users/tskim/VAPKT-data/results/stage4-taxi-p11ck3000 — 지우지 않는다(hyp·asr 은 TAXI 텍스트라 mxc 에만).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1 | cut -d' ' -f1)
echo "[$(date +%T)] GPU $GPU ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU))"; export CUDA_VISIBLE_DEVICES=$GPU
O=/soundai/users/tskim/VAPKT-data/results/stage4-taxi-p11ck3000; SES=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
$PY experiments/cst_stage4_taxi.py run --checkpoint /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000 --encoder $MXC_NEMOTRON_DIR --sessions $SES --out $O --deltas 4,10 2>&1 | grep -vE "regex pattern|Warning" | tail -3
$PY experiments/cst_stage4_taxi.py score --sessions $SES --out $O 2>&1 | tail -6
echo "TAXI_DONE"
