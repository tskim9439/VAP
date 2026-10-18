# Stage 3 격리 발화 되살리기(--revive-dir revive-asr-v1) + p10 레시피 스모크 20 step(단일 GPU --gpu, replay 포함). 산출 /tmp/sa_tskim/stage3-smoke-revive — 지우지 않는다.
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
/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/stage3_train.py --gpu ${GPUS:-5} \
  --init /soundai/Model/VAPASR/semcommit-q06-s4-ast/final --nemotron-dir $MXC_NEMOTRON_DIR --out-dir /tmp/sa_tskim/stage3-smoke-revive --data $D --split $D/splits/v2.json \
  --max-windows 300 --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.01 --replay-max-items 300 \
  --ast-targets $AST,$AST_CV4 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --speed-perturb 0.9,1.0,1.1 --noise-bank $NB \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --spk-act-weight 0.5 --backchannel 1.0,3 --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,rep_ko:1.2,rep_en:1.2,rep_de:1.2 --sim-data /soundai/users/tskim/VAPKT-data/data/sim-dialogue-v2 --revive-dir $D/revive-asr-v1 --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --check-items 8 --report-to none
echo "EXIT $?"
