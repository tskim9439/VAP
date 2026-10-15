# Stage 3 학습 스크립트 dry-run(모델 없이 데이터셋·불변식·샘플). 산출: /tmp/sa_tskim/stage3-dry/ — 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data/data/phase2
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=/soundai/users/tskim/VAPKT-data/data/cv4-en-v1/lists/cv4-en-words-train.list
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
AST=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16
AST_CV4=/soundai/users/tskim/VAPKT-data/data/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VAPASR_TAR_INDEX_ROOT=/soundai/users/tskim/VAPKT-data/data/speechlm-asr-tar-index-v1
/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/stage3_train.py --dry-run \
  --init /soundai/Model/VAPASR/semcommit-q06-s4-ast/final --out-dir /tmp/sa_tskim/stage3-dry --data $D --split $D/splits/v2.json \
  --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.05 \
  --ast-targets $AST,$AST_CV4 --ast-frac 0.8 --offline-frac 0.3 --check-items 32
echo "EXIT $?"
