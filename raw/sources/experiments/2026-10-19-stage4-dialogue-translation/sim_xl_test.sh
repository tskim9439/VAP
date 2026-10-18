# Stage 4 S3 교차 언어 합성 대화 시험(KO↔EN, 시험 타깃): 한국어 = AI Hub 시험 대화(S1 ko→en), 영어 = otoSpeech 시험 대화(S1 en→ko) + words(Stage 2 en→ko)
# → cst_stage4_dryrun. CPU 호스트. 산출 /tmp/sa_tskim/stage4-sim-xl-test2 — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
D=/soundai/users/tskim/VAPKT-data/data
SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
O=/tmp/sa_tskim/stage4-sim-xl-test2
$PY experiments/sim_build_dialogues.py --lang Korean --lang2 English --out $O --sessions 24 --dur 60,90 --phase2 $D/phase2 --split $D/phase2/splits/v2.json \
  --phase2-corpora aihub71631,aihub134-1,aihub134-2 --pool-weights aihub71631:0.4,aihub134-1:0.3,aihub134-2:0.3 \
  --phase2-corpora2 otoSpeech --words2 @$SNAP/main-words.list --words2 @$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list --words2 @$D/cv4-en-v1/lists/cv4-en-words-train.list \
  --words-frac 0.03 --pool-weights2 otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1 \
  --trans $D/stage4-dlg-units-pilot/targets-fix-gen-v3-bf16 --trans $D/ast-mono-v1/targets-fix-gen-v3-bf16/en-ko --trans $D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16/en-ko \
  --min-units 5 --noise-bank $D/noise-bank-v1 --workers 8 2>&1 | grep -v "regex pattern"
echo "BUILD $?"
$PY experiments/cst_stage4_dryrun.py --data $O --tokenizer /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000 --corpora sim_ko-en --trans $O/targets-sim_ko-en \
  --backchannel 1.0,3 --show 2 --procs 1 --audio-sample 6 --out $O/dry.json 2>&1 | grep -v "regex pattern"
echo "EXIT $?"
# S4 오라클: 평가 도구가 정답을 만점으로 채점하는가(합성 KO↔EN 전부 + 시험 번역 대화가 있는 코퍼스 둘, 창 상한 60)
CK=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
$PY experiments/cst_stage4_eval.py run --checkpoint $CK --data $O --trans $O/targets-sim_ko-en --corpora sim_ko-en --deltas 4,10 --limit-windows 0 --oracle --out $O/eval-oracle-xl 2>&1 | grep -v "regex pattern" | tail -3
$PY experiments/cst_stage4_eval.py summarize --out $O/eval-oracle-xl
$PY experiments/cst_stage4_eval.py run --checkpoint $CK --data $D/phase2 --trans $D/stage4-dlg-units-pilot/targets-fix-gen-v3-bf16 --corpora otoSpeech,kec,aihub134-1 --deltas 4,10 \
  --limit-windows 60 --oracle --backchannel 1.0,3 --out $O/eval-oracle-pilot 2>&1 | grep -v "regex pattern" | tail -3
$PY experiments/cst_stage4_eval.py summarize --out $O/eval-oracle-pilot
echo "ORACLE_DONE"
