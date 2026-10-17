# Stage 3 화자 슬롯 활동 헤드(--spk-act-weight 0.5, r1 레시피) 호스트 GPU 2 장 torchrun 스모크 20 step(DDP·replay step 의 0 기울기 묶음 확인). 산출 /tmp/sa_tskim/stage3-smoke-spk — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2
SNAP=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=/soundai/users/tskim/VAPKT-data/data/cv4-en-v1/lists/cv4-en-words-train.list
DE=/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list
AST=/soundai/users/tskim/VAPKT-data/data/ast-mono-v1/targets-fix-gen-v3-bf16
AST_CV4=/soundai/users/tskim/VAPKT-data/data/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
NB=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_TAR_INDEX_ROOT=/soundai/users/tskim/VAPKT-data/data/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
CUDA_VISIBLE_DEVICES=${GPUS:-2,4} /soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python -m torch.distributed.run --nproc_per_node 2 --master_port 29613 experiments/stage3_train.py \
  --init /soundai/Model/VAPASR/semcommit-q06-s4-ast/final --nemotron-dir $MXC_NEMOTRON_DIR --out-dir /tmp/sa_tskim/stage3-smoke-spk --data $D --split $D/splits/v2.json \
  --max-windows 300 --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.01 --replay-max-items 300 \
  --ast-targets $AST,$AST_CV4 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --speed-perturb 0.9,1.0,1.1 --noise-bank $NB \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --spk-act-weight 0.5 --backchannel 1.0,3 --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,rep_ko:0.75,rep_en:0.75,rep_de:0.75 --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --check-items 8 --report-to none
echo "EXIT $?"
