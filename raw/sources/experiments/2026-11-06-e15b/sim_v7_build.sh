#!/usr/bin/env bash
# 겹침 데이터 v7-ovl2 본 셋(2026-11-08, 사용자 "판 B 로 split-s 8 도 같이 시험해줘" → 판 D 추천 → "응 시작해.") — 키 sim_en-ovl2, 산출 VAPKT-data/data/sim-dialogue-v7-ovl2/(새 폴더).
#   설정 = sim_v7_trial.sh 판 D(300 세션 시험: 말소리 중 겹침 43.9 % — 둘 38.3 · 셋+ 5.6, 겹침 구간 중앙값 1.36 s · 90 % 4.64 s, 90 % 이상 겹친 발화 41.2 %;
#   AMI 공식 dev·eval 15.2 % · 0.64 s · 2.00 s · 13.9 %, v6 sim_en-ovl 22.8 %). 내용 있는 말끼리 길게·완전히 겹치는 예시를 늘려, 인코더를 고정한 E13 계열이
#   겹친 말을 받아쓰는 법을 데이터로만 배우는지 본다(wiki output-ssa-papers-review-20261106). 재료·원거리·잡음은 v6 와 같다(split v4 학습 분할만).
#   bash sim_v7_build.sh. mxc 호스트 CPU, 제출 없음. 아무것도 지우지 않는다(산출 폴더가 있으면 멈춘다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; O=$D/sim-dialogue-v7-ovl2
W=/soundai/users/tskim/VAPKT-data/data/semcommit-work
[ -e $O/sim_en-ovl2.refined.dialogues.jsonl ] && { echo "이미 있다: $O"; exit 1; }
echo "[$(date "+%F %T")] sim v7-ovl2 en"
$P experiments/sim_build_dialogues.py --lang English --out $O --sessions 20000 --dur 60,120 --phase2 $P2 --split $P2/splits/v4.json \
  --phase2-corpora otoSpeech,ami,icsi,notsofar --stats-corpora ami,icsi,notsofar \
  --words @$W/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list --words @$W/asr-words-v1/lists/asr-words-20260929.list \
  --words @$D/cv4-en-v1/lists/cv4-en-words-train.list --words-frac 0.5 \
  --pool-weights otoSpeech:0.15,ami:0.08,icsi:0.08,notsofar:0.04,swbd:0.2,cv:0.15,vp:0.15,ls:0.15 --p-back 0.2 \
  --speakers 2:0.4,3:0.4,4:0.2 --p-interrupt 0.8 --interrupt-fto=-8,-1 --bc-scale 1.0 --p-hold 0.2 --p-full-overlap 0.3 --split-s 8 \
  --noise-bank $D/noise-bank-v1 --noise-p 0.7 --noise-snr 5,25 --rir-p 0.3 --far-per-speaker 0.7 --name-suffix ovl2 --workers 32 --seed 0
echo "[$(date "+%F %T")] SIM_V7_EXIT $?"
