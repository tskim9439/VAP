# S4 오라클 다시(참조 번역도 miss 청크를 뺌) — 합성 KO↔EN 전부. 산출 /tmp/sa_tskim/stage4-sim-xl-test2/eval-oracle-xl2 — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
O=/tmp/sa_tskim/stage4-sim-xl-test2; CK=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
$PY experiments/cst_stage4_eval.py run --checkpoint $CK --data $O --trans $O/targets-sim_ko-en --corpora sim_ko-en --deltas 4,10 --limit-windows 0 --oracle --out $O/eval-oracle-xl2 2>&1 | grep -v "regex pattern" | tail -1
$PY experiments/cst_stage4_eval.py summarize --out $O/eval-oracle-xl2
