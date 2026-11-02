#!/usr/bin/env bash
# 2026-11-02 G2b 스모크 둘째: g2b_diag.sh ②와 같은 설정(G2 final 에서 회의 셋 4–6 분 창·배치 1·원거리 가중)에 --stream-chunk 4,16(step 마다 번갈아 — 추론 청크 4 와 맞춘다).
#   4 step(청크 4·16·4·16) + 저장·재로드 대조 — 청크 4 의 걸음 시간·메모리. bash g2b_smoke2.sh <GPU>. 산출 tmp/e11-check/g2b-smoke2(새 폴더). 아무것도 지우지 않는다.
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data; T=$D/tmp/e11-check; P=$D/conda/envs/vapasr/bin/python
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=$1
N3=$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo; G=/soundai/Model/VAPASR/graft-q17-graft-g2-lora/final/graft.pt
$P experiments/n3d_graft_train.py --init /soundai/Model/VAPASR/stage3-q17-e9/final --n3d $N3 --nemotron-dir $MXC_NEMOTRON_DIR \
  --out-dir $T/g2b-smoke2 --corpora ami,icsi,notsofar,chime6tr,ami-far,icsi-far,notsofar-far,chime6tr-far --window 240,360 --batch 1 --num-workers 2 \
  --weights ami:2,icsi:1,notsofar:0.5,chime6tr:1,ami-far:4,icsi-far:2,notsofar-far:1,chime6tr-far:2 --noise-bank $D/data/noise-bank-v1 \
  --init-graft $G --stream-chunk 4,16 --hid-weight 0.1 --ref-weight 0.5 --unfreeze-layers 12 --enc-lr 2e-5 --lr 5e-5 --guard-weight 10 \
  --n3d-train lora --n3d-lr 1e-4 --lora-rank 32 --lora-alpha 32 --log-every 1 --smoke 4 > $T/g2b-smoke2.log 2>&1
echo "G2B_SMOKE2_DONE exit $? $(date +%T)" >> $T/g2b-diag-done.txt
