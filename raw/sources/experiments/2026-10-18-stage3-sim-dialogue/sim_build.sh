# 합성 대화 빌드: bash sim_build.sh <출력 폴더> <세션 수> <워커> [en|ko|de ...] — 아무것도 지우지 않는다.
set -u
OUT=$1; N=$2; W=$3; shift 3; LANGS=${*:-en ko de}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; P2=$D/data/phase2; SNAP=$D/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=$D/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list; CV4=$D/data/cv4-en-v1/lists/cv4-en-words-train.list
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 VAPASR_TAR_INDEX_ROOT=$D/data/speechlm-asr-tar-index-v1
PY=$D/conda/envs/vapasr/bin/python; COMMON="--out $OUT --sessions $N --workers $W --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/data/noise-bank-v1"
for L in $LANGS; do
  echo "[$(date '+%F %T')] sim $L 시작"
  case $L in
    en) $PY experiments/sim_build_dialogues.py --lang English $COMMON --phase2-corpora otoSpeech --words @$SNAP/main-words.list --words @$ASR --words @$CV4 --words-frac ${WORDS_FRAC:-0.1} \
          --pool-weights otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1 ;;
    ko) $PY experiments/sim_build_dialogues.py --lang Korean $COMMON --phase2-corpora aihub71631,aihub134-1,aihub134-2 --pool-weights aihub71631:0.4,aihub134-1:0.3,aihub134-2:0.3 ;;
    de) $PY experiments/sim_build_dialogues.py --lang German $COMMON --phase2-corpora kec,ztc,nsc --pool-weights kec:0.35,ztc:0.35,nsc:0.3 ;;
  esac
  echo "[$(date '+%F %T')] sim $L 끝 rc=$?"
done
