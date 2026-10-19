# Stage 4 S3 본 생성(mxc 호스트 CPU): 교차 언어 합성 학습 셋 sim_en-de·sim_ko-en(1 만 세션씩, v2 합성과 같은 규모·잡음 뱅크) + 영↔독 held-out 평가 셋(300 세션),
# 이어서 전체 타깃 dry-run(실제 대화 7 코퍼스 + 합성). 산출 VAPKT-data/data/sim-dialogue-xl-v1 · sim-dialogue-xl-eval-v1 · tmp/stage4-dry-full*. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list; CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; S1R=$D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16
A2=$D/ast-mono-v1/targets-fix-gen-v3-bf16; A2C=$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
O=$D/sim-dialogue-xl-v1; OE=$D/sim-dialogue-xl-eval-v1
COMMON="--sessions 10000 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1"
# 이름은 --lang·--lang2 순서로 정해진다(sim_<l1>-<l2>). 학습 가중치 키와 맞추려고 영어를 --lang 으로 둔다: sim_en-de, sim_en-ko
echo "[$(date '+%F %T')] sim_en-de 시작"
$PY experiments/sim_build_dialogues.py --lang English --lang2 German --out $O $COMMON --phase2-corpora otoSpeech --words @$SNAP/main-words.list --words @$ASR --words @$CV4 --words-frac 0.5 \
  --pool-weights otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1 --phase2-corpora2 kec,ztc,nsc --pool-weights2 kec:0.35,ztc:0.35,nsc:0.3 \
  --trans $S1 --trans $A2/en-de --trans $A2C/en-de 2>&1 | grep -v "regex pattern" | tail -3
echo "[$(date '+%F %T')] sim_en-ko 시작"
$PY experiments/sim_build_dialogues.py --lang English --lang2 Korean --out $O $COMMON --phase2-corpora otoSpeech --words @$SNAP/main-words.list --words @$ASR --words @$CV4 --words-frac 0.5 \
  --pool-weights otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1 --phase2-corpora2 aihub71631,aihub134-1,aihub134-2 --pool-weights2 aihub71631:0.4,aihub134-1:0.3,aihub134-2:0.3 \
  --trans $S1 --trans $A2/en-ko --trans $A2C/en-ko 2>&1 | grep -v "regex pattern" | tail -3
echo "[$(date '+%F %T')] 평가 sim_en-de(held-out) 시작"
$PY experiments/sim_build_dialogues.py --lang English --lang2 German --out $OE --sessions 300 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1 --heldout \
  --phase2-corpora otoSpeech --pool-weights otoSpeech:1 --phase2-corpora2 kec,ztc,nsc --pool-weights2 kec:0.35,ztc:0.35,nsc:0.3 --trans $S1 2>&1 | grep -v "regex pattern" | tail -3
echo "[$(date '+%F %T')] dry-run 시작"
TOKZ=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
$PY experiments/cst_stage4_dryrun.py --data $P2 --tokenizer $TOKZ --split $P2/splits/v2.json --trans $S1 --trans-input $S1R --revive-dir $P2/revive-asr-v2 --max-windows 1500 --audio-sample 6 \
  --out $D/../tmp/stage4-dry-full.json 2>&1 | grep -E '^\{|all_ok'
$PY experiments/cst_stage4_dryrun.py --data $O --tokenizer $TOKZ --corpora sim_en-de,sim_en-ko --trans $O/targets-sim_en-de,$O/targets-sim_en-ko --max-windows 1500 --audio-sample 6 --procs 2 \
  --out $D/../tmp/stage4-dry-xl.json 2>&1 | grep -E '^\{|all_ok'
$PY experiments/cst_stage4_dryrun.py --data $OE --tokenizer $TOKZ --corpora sim_en-de --trans $OE/targets-sim_en-de --audio-sample 6 --procs 1 \
  --out $D/../tmp/stage4-dry-xl-eval.json 2>&1 | grep -E '^\{|all_ok'
echo "XL_ALL_DONE $(date '+%F %T')"
