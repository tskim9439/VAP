# 번역 행 앞 공백(--trans-lead-space) 점검(mxc 호스트 CPU): 실제 대화 7 코퍼스·교차 언어 합성 두 셋을 켠 상태로 dry-run(문제 0 이어야 한다),
# 이어서 타깃 번역 흐름에서 붙은 단어 비율을 끈 것과 켠 것으로 잰다. 산출 VAPKT-data/tmp/stage4-dry-ls*.json. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; O=$D/sim-dialogue-xl-v1
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; S1R=$D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; TOKZ=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
$PY experiments/cst_stage4_dryrun.py --data $P2 --tokenizer $TOKZ --split $P2/splits/v2.json --trans $S1 --trans-input $S1R --revive-dir $P2/revive-asr-v2 --max-windows 500 --audio-sample 3 \
  --trans-lead-space --out $D/../tmp/stage4-dry-ls-dlg.json 2>&1 | grep -E '^\{|all_ok'
$PY experiments/cst_stage4_dryrun.py --data $O --tokenizer $TOKZ --corpora sim_en-de,sim_en-ko --trans $O/targets-sim_en-de,$O/targets-sim_en-ko --max-windows 500 --audio-sample 3 --procs 2 \
  --trans-lead-space --out $D/../tmp/stage4-dry-ls-xl.json 2>&1 | grep -E '^\{|all_ok'
echo "LS_DRY_DONE $(date '+%F %T')"
