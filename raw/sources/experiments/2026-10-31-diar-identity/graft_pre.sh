#!/bin/bash
# 2026-10-31 아래쪽 이식 판가름: 우리 서브샘플링 출력(+ 아래 층)으로 N3D 층 입력 h_m 을 능형 회귀하고 선형 이식 DER(오프라인 창, SDM)을 잰다(n3d_graft_probe --pre).
# 쓰기: graft_pre.sh <GPU> <이름> <탭> <이식 층>. 산출은 이 폴더 아래에만. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
T=/soundai/users/tskim/VAPKT-data/tmp/e11-check/layer-probe; N3=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
CUDA_VISIBLE_DEVICES=$1 python experiments/n3d_graft_probe.py --ckpt /soundai/Model/VAPASR/stage3-q17-e11/final --n3d $N3 --out $T/graft-$2 --taps $3 --pre --graft-layers $4 > $T/graft-$2.out 2>&1
echo "[$(date "+%F %T")] GRAFT_DONE $2" >> $T/done-graft.txt
