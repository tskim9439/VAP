#!/bin/bash
# 2026-10-31 화자 정체 진단: 교사 N3D 와 U3 학생의 AMI 회의 통째 화자 분리를 프레임 확률까지 남겨(--save-probs) 다시 돈다(호스트 GPU 1–6, 앞선 평가와 같은 설정).
# 산출은 tmp 아래 새 폴더에만 쓴다. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
T=/soundai/users/tskim/VAPKT-data/tmp/e11-check/diar-probs; mkdir -p $T
N3=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
G=/soundai/Model/VAPASR/graft-q17-graft-u3-l12/final/graft.pt
for r in 0 1 2; do CUDA_VISIBLE_DEVICES=$((1 + r)) python experiments/diar_eval.py run --n3d $N3 --n3d-chunk 4 --session --save-probs --out $T/n3d --rank $r --world-size 3 > $T/n3d-rank$r.log 2>&1 & done
for r in 0 1 2; do CUDA_VISIBLE_DEVICES=$((4 + r)) python experiments/diar_eval.py run --graft $G --session --save-probs --out $T/u3 --rank $r --world-size 3 > $T/u3-rank$r.log 2>&1 & done
wait
echo "[$(date '+%F %T')] PROBS_DONE" >> $T/done.txt
