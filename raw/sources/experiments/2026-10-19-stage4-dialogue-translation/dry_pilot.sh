# Stage 4 S2 dry-run(시험 번역 타깃, 번역된 대화의 창만) — CPU. 산출 /soundai/users/tskim/VAPKT-data/tmp/stage4-dry-pilot2.json. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
$PY -m pytest -q -p no:cacheprovider tests/test_cst_dialogue.py 2>&1 | tail -1
D=/soundai/users/tskim/VAPKT-data/data/phase2
$PY experiments/cst_stage4_dryrun.py --data $D --tokenizer /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000 --split $D/splits/v2.json \
  --trans /soundai/users/tskim/VAPKT-data/data/stage4-dlg-units-pilot/targets-fix-gen-v3-bf16 --revive-dir $D/revive-asr-v2 --only-translated --show 2 \
  --out /soundai/users/tskim/VAPKT-data/tmp/stage4-dry-pilot2.json
echo "EXIT $?"
