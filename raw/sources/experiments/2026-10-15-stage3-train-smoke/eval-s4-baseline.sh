# Stage 3 평가 기준점: Stage 2 최종(q06-s4-ast/final)을 두 슬롯 prefix 로 held-out 창 200 개씩(7 코퍼스), δ4·10. <EOT> 는 이 체크포인트에서 차단돼 나오지 않는다.
# 산출 /soundai/users/tskim/VAPKT-data/results/stage3-eval-s4-20261015 — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2; O=/soundai/users/tskim/VAPKT-data/results/stage3-eval-s4-20261015
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
$P experiments/cst_stage3_eval.py run --checkpoint /soundai/Model/VAPASR/semcommit-q06-s4-ast/final --encoder $MXC_NEMOTRON_DIR --data $D --split $D/splits/v2.json \
  --out $O --deltas 4,10 --limit-windows 200 --batch-size 32 --verify
$P experiments/cst_stage3_eval.py summarize --out $O
echo "EXIT $?"
