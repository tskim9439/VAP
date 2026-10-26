# D6 회의형 합성 v4(mxc 호스트, 2026-10-26): 3–4 인, 끼어들기 + 맞장구 많이, 화자별 원거리(한 방의 다른 RIR) + 잡음 — 키 sim_<l>-meet.
#   bash sim_v4_build.sh [en|ko|de …](기본 셋 다, 차례로). 산출 VAPKT-data/data/sim-dialogue-v4-meet/(sim_<l>-meet.refined.dialogues.jsonl·stats.json·mono/). 아무것도 지우지 않는다.
#   교대 간격 통계: 영어 = 회의(AMI·ICSI·NOTSOFAR 학습 분할), 한국어·독일어 = 그 언어 대화(회의 자연 데이터가 아직 없다 — 겹침은 끼어들기로 늘린다).
#   재료: split v4 학습 분할만(held-out — AMI·NOTSOFAR 공식 dev·eval 포함 — 은 쓰지 않는다). 영어 words 는 v3-multi 와 같은 목록.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; O=$D/sim-dialogue-v4-meet
W=/soundai/users/tskim/VAPKT-data/data/semcommit-work
COMMON="--out $O --sessions 10000 --dur 60,120 --phase2 $P2 --split $P2/splits/v4.json --speakers 3:0.4,4:0.6 --p-back 0.2 --bc-scale 2.0
  --p-interrupt 0.15 --interrupt-fto=-2.0,-0.3 --noise-bank $D/noise-bank-v1 --noise-p 0.7 --noise-snr 5,25 --rir-p 0.3 --far-per-speaker 0.7 --name-suffix meet --workers 24 --seed 0"
for l in ${@:-en ko de}; do
  echo "[$(date "+%F %T")] sim v4 $l"
  case $l in
    en) $P experiments/sim_build_dialogues.py --lang English $COMMON --phase2-corpora otoSpeech,ami,icsi,notsofar --stats-corpora ami,icsi,notsofar \
          --words @$W/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list --words @$W/asr-words-v1/lists/asr-words-20260929.list \
          --words @$D/cv4-en-v1/lists/cv4-en-words-train.list --words-frac 0.5 \
          --pool-weights otoSpeech:0.2,ami:0.15,icsi:0.15,notsofar:0.05,swbd:0.2,cv:0.1,vp:0.1,ls:0.05 ;;
    ko) $P experiments/sim_build_dialogues.py --lang Korean $COMMON --phase2-corpora aihub71631,aihub134-1,aihub134-2 --pool-weights aihub71631:0.4,aihub134-1:0.3,aihub134-2:0.3 ;;
    de) $P experiments/sim_build_dialogues.py --lang German $COMMON --phase2-corpora kec,ztc,nsc --pool-weights kec:0.35,ztc:0.35,nsc:0.3 ;;
  esac
  rc=$?; echo "[$(date "+%F %T")] SIM_V4_EXIT $l $rc"
done
