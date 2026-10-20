# V2 3–4 인 합성 대화 빌드(mxc 호스트 CPU): bash sim_build_multi.sh <train|eval> [en ko de] — 아무것도 지우지 않는다.
# train: 언어마다 10,000 세션 → data/sim-dialogue-v3-multi(화자 수 2:0.2·3:0.4·4:0.4, p_back 0.2) — v2(2 인, stage3 sim_build.sh, 영어 WORDS_FRAC=0.5)와 같은 재료·풀 가중·잡음.
# eval : held-out 대화 재료만(--heldout), 언어마다 300 세션 → data/sim-dialogue-multi-eval-v1(학습 셋과 겹치지 않는다).
# CODE(기본 /soundai/users/tskim/VAPKT) 의 코드를 쓴다 — 학습 전 개발 사본(VAPKT-dev)에서 돌릴 때 CODE 를 준다.
set -u
MODE=$1; shift; LANGS=${*:-en ko de}; CODE=${CODE:-/soundai/users/tskim/VAPKT}
cd $CODE; set -a; source /soundai/users/tskim/VAPKT/.env; set +a
D=/soundai/users/tskim/VAPKT-data; P2=$D/data/phase2; SNAP=$D/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=$D/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list; CV4=$D/data/cv4-en-v1/lists/cv4-en-words-train.list
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1
PY=$D/conda/envs/vapasr/bin/python; SPK="--speakers 2:0.2,3:0.4,4:0.4 --p-back 0.2"
if [ "$MODE" = train ]; then COMMON="--out $D/data/sim-dialogue-v3-multi --sessions 10000 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/data/noise-bank-v1 $SPK"
else COMMON="--out $D/data/sim-dialogue-multi-eval-v1 --sessions 300 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/data/noise-bank-v1 --heldout --seed 1 $SPK"; fi
for L in $LANGS; do
  echo "[$(date '+%F %T')] sim $MODE $L 시작"
  case $L in
    en) if [ "$MODE" = train ]; then $PY experiments/sim_build_dialogues.py --lang English $COMMON --phase2-corpora otoSpeech --words @$SNAP/main-words.list --words @$ASR --words @$CV4 --words-frac 0.5 \
          --pool-weights otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1
        else $PY experiments/sim_build_dialogues.py --lang English $COMMON --phase2-corpora otoSpeech --pool-weights otoSpeech:1; fi ;;
    ko) $PY experiments/sim_build_dialogues.py --lang Korean $COMMON --phase2-corpora aihub71631,aihub134-1,aihub134-2 --pool-weights aihub71631:0.4,aihub134-1:0.3,aihub134-2:0.3 ;;
    de) $PY experiments/sim_build_dialogues.py --lang German $COMMON --phase2-corpora kec,ztc,nsc --pool-weights kec:0.35,ztc:0.35,nsc:0.3 ;;
  esac
  rc=$?; echo "[$(date '+%F %T')] sim $MODE $L 끝 rc=$rc"                 # rc 를 먼저 받는다($(date) 가 $? 를 덮는다)
done
echo "SIM_MULTI_${MODE}_DONE"
