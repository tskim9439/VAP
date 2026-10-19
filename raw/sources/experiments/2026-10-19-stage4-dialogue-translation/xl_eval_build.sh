# 영↔독 held-out 평가 셋(300 세션, --heldout: held-out 대화만·words 없음) + dry-run. 산출 VAPKT-data/data/sim-dialogue-xl-eval-v1. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; OE=$D/sim-dialogue-xl-eval-v1
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
$PY experiments/sim_build_dialogues.py --lang English --lang2 German --out $OE --sessions 300 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1 --heldout \
  --phase2-corpora otoSpeech --pool-weights otoSpeech:1 --phase2-corpora2 kec,ztc,nsc --pool-weights2 kec:0.35,ztc:0.35,nsc:0.3 --trans $S1 2>&1 | grep -v "regex pattern" | tail -3
$PY experiments/cst_stage4_dryrun.py --data $OE --tokenizer /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000 --corpora sim_en-de --trans $OE/targets-sim_en-de --audio-sample 6 --procs 1 \
  --out $D/../tmp/stage4-dry-xl-eval.json 2>&1 | grep -E '^\{|all_ok'
echo "XL_EVAL_DONE"
