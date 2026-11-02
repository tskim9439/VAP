#!/usr/bin/env bash
# 2026-11-02 SDM 진단(사용자 "우리 목표에서는 G3 형태로 가야 하는데, SDM 에서 성능이 안 좋네"):
# ① 청크 진단: G2 학생과 교사 N3D 를 학습 청크 16(1.28 s)으로 AMI SDM 회의 통째(eval·dev)에 돌린다. 추론 청크 4(320 ms) 값과 견줘,
#    회의 중간부터 화자가 합쳐지는 표류가 학습·추론 청크 차이에서 오는지 본다(GPU A 학생, GPU B 교사).
# ② G2b 스모크: G2 final 에서 회의 셋만(근·원거리, 원거리 가중), 4–6 분 창, 배치 1 로 4 step + 저장·재로드 대조 — 긴 창의 메모리·시간(GPU A).
#   bash g2b_diag.sh <GPU A> <GPU B>. 산출은 tmp/e11-check 아래 새 폴더. 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
N3=$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo; G=/soundai/Model/VAPASR/graft-q17-graft-g2-lora/final/graft.pt
CUDA_VISIBLE_DEVICES=$1 $P experiments/diar_eval.py run --graft $G --n3d-chunk 16 --session --save-probs --corpora ami-sdm --out $T/g2-c16-session > $T/g2-c16-session.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 $P experiments/diar_eval.py run --n3d $N3 --n3d-chunk 16 --session --save-probs --corpora ami-sdm --out $T/n3d-c16-session > $T/n3d-c16-session.log 2>&1 &
wait
$P experiments/diar_eval.py summarize --out $T/g2-c16-session/graft > $T/g2-c16-session/summary.log 2>&1
$P experiments/diar_eval.py summarize --out $T/n3d-c16-session/n3d > $T/n3d-c16-session/summary.log 2>&1
echo "C16_DONE $(date +%T)" >> $T/g2b-diag-done.txt
CUDA_VISIBLE_DEVICES=$1 $P experiments/n3d_graft_train.py --init /soundai/Model/VAPASR/stage3-q17-e9/final --n3d $N3 --nemotron-dir $MXC_NEMOTRON_DIR \
  --out-dir $T/g2b-smoke --corpora ami,icsi,notsofar,chime6tr,ami-far,icsi-far,notsofar-far,chime6tr-far --window 240,360 --batch 1 --num-workers 2 \
  --weights ami:2,icsi:1,notsofar:0.5,chime6tr:1,ami-far:4,icsi-far:2,notsofar-far:1,chime6tr-far:2 --noise-bank $D/data/noise-bank-v1 \
  --init-graft $G --stream-chunk 16 --hid-weight 0.1 --ref-weight 0.5 --unfreeze-layers 12 --enc-lr 2e-5 --lr 5e-5 --guard-weight 10 \
  --n3d-train lora --n3d-lr 1e-4 --lora-rank 32 --lora-alpha 32 --log-every 1 --smoke 4 > $T/g2b-smoke.log 2>&1
echo "G2B_SMOKE_DONE exit $? $(date +%T)" >> $T/g2b-diag-done.txt
