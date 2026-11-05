#!/usr/bin/env bash
# E15b 겹침 많은 합성 영어 회의 v6-ovl 본 셋(2026-11-06, mxc 호스트 CPU, 제출 없음) — 키 sim_en-ovl, 산출 VAPKT-data/data/sim-dialogue-v6-ovl/(새 폴더).
#   설정 = sim_v6_trial.sh 판 C(300 세션 시험: 말소리 프레임 중 겹침 22.8 % — 둘 20.5 · 셋+ 2.3; AMI 공식 dev·eval 15.2 %, 지금 sim_en-meet 13.9 %).
#   AMI 에 맞추는 셋이 아니라 슬롯 조건 인코더(E15b)가 겹친 말을 가르는 예시를 늘리는 셋이다. 재료는 큰 단일 화자 셋 쪽(swbd·cv·vp·ls)으로 옮겨 AMI·ICSI 발화 재사용을 줄인다.
#   split v4 학습 분할만(AMI·NOTSOFAR 공식 dev·eval 은 쓰지 않는다). bash sim_v6_build.sh. 아무것도 지우지 않는다(산출 폴더가 있으면 멈춘다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; O=$D/sim-dialogue-v6-ovl
W=/soundai/users/tskim/VAPKT-data/data/semcommit-work
[ -e $O/sim_en-ovl.refined.dialogues.jsonl ] && { echo "이미 있다: $O"; exit 1; }
echo "[$(date "+%F %T")] sim v6-ovl en"
$P experiments/sim_build_dialogues.py --lang English --out $O --sessions 10000 --dur 60,120 --phase2 $P2 --split $P2/splits/v4.json \
  --phase2-corpora otoSpeech,ami,icsi,notsofar --stats-corpora ami,icsi,notsofar \
  --words @$W/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list --words @$W/asr-words-v1/lists/asr-words-20260929.list \
  --words @$D/cv4-en-v1/lists/cv4-en-words-train.list --words-frac 0.5 \
  --pool-weights otoSpeech:0.15,ami:0.08,icsi:0.08,notsofar:0.04,swbd:0.2,cv:0.15,vp:0.15,ls:0.15 --p-back 0.2 \
  --speakers 3:0.3,4:0.7 --p-interrupt 0.6 --interrupt-fto=-3.5,-0.5 --bc-scale 3.5 --p-hold 0.2 \
  --noise-bank $D/noise-bank-v1 --noise-p 0.7 --noise-snr 5,25 --rir-p 0.3 --far-per-speaker 0.7 --name-suffix ovl --workers 24 --seed 0
echo "[$(date "+%F %T")] SIM_V6_EXIT $?"
