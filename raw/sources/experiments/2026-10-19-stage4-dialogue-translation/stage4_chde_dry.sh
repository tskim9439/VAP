# CallHome 독일어 단일 화자 셋(--replay-set rep_chde) 학습 dry-run(mxc 호스트 CPU): st4-m1 설정 + rep_chde:0.3. 산출 /tmp/sa_tskim/stage4-chde-dry-train — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2
SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637; ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list; DE=$D/yodas-de-v1/lists/de-words-train-tn2.list
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1 VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
echo "[$(date "+%F %T")] dry-run"
/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/stage3_train.py --dry-run \
  --init /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000 --nemotron-dir $MXC_NEMOTRON_DIR --out-dir /tmp/sa_tskim/stage4-chde-dry-train --data $P2 --split $P2/splits/v2.json \
  --max-windows 300 --replay-words @$SNAP/main-words.list --replay-words @$ASR --replay-words @$CV4 --replay-de-words @$DE --replay-files-frac 0.01 --replay-max-items 300 \
  --ast-targets $D/ast-mono-v1/targets-fix-gen-v3-bf16,$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16 --ast-frac 0.8 --offline-frac 0.3 --gain-augment 0.5,-70,-20 --speed-perturb 0.9,1.0,1.1 --noise-bank $D/noise-bank-v1 \
  --train-encoder --encoder-right-context 3 --encoder-context-sampling fixed --spec-augment 2,27,2,0.03 --attn-impl varlen --batch-max-tokens 16384 --pack-max-bs 64 \
  --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $P2/revive-asr-v2 \
  --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16 --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 \
  --sim-data $D/sim-dialogue-v2,$D/sim-dialogue-xl-v1 --sim-trans-frac 0.8 --trans-lead-space --replay-set rep_chde:German:@$D/callhome-deu/words-v1/lists/callhome-de-words-train.list \
  --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2,rep_chde:0.3 \
  --lr 4e-5 --max-steps 20 --warmup 5 --save-every 1000 --log-every 2 --num-workers 2 --check-items 32 --report-to none 2>&1 | grep -v "regex pattern"
echo "DRY EXIT $?"
