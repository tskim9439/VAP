# Stage 3 평가 스모크: 20 step 스모크 체크포인트로 KEC·134-2 held-out 창 16 개씩, δ4·10, 직렬/배치 검증. 산출 /tmp/sa_tskim/stage3-eval-smoke — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
$P experiments/cst_stage3_eval.py run --checkpoint /tmp/sa_tskim/stage3-smoke/final --encoder $MXC_NEMOTRON_DIR --data $D --split $D/splits/v2.json \
  --out /tmp/sa_tskim/stage3-eval-smoke --corpora kec,aihub134-2 --deltas 4,10 --limit-windows 16 --batch-size 8 --verify
$P experiments/cst_stage3_eval.py summarize --out /tmp/sa_tskim/stage3-eval-smoke
echo "EXIT $?"
