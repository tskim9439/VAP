#!/bin/bash
# 2026-10-31 인코더 층별 화자 구분력(experiments/layer_speaker_probe.py): 우리 인코더·U3 인코더(+ 학생 캐시 ĥ_8)·E7a 캐시 임베딩·N3D 층.
# 쓰기: run.sh <GPU 목록 쉼표> [추가 인자…] — GPU 마다 rank 하나. 산출은 이 폴더 아래에만. 아무것도 지우지 않는다.
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
T=/soundai/users/tskim/VAPKT-data/tmp/e11-check/layer-probe; OUT=${OUT:-$T/full}; mkdir -p $OUT
N3=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
G=/soundai/Model/VAPASR/graft-q17-graft-u3-l12/final/graft.pt
IFS=, read -ra GP <<< "$1"; shift; W=${#GP[@]}
for r in "${!GP[@]}"; do
  CUDA_VISIBLE_DEVICES=${GP[$r]} python experiments/layer_speaker_probe.py run --ckpt /soundai/Model/VAPASR/stage3-q17-e11/final --n3d $N3 --graft $G \
    --diar /soundai/Model/VAPASR/diar-q17-diar-e7a/final --out $OUT --rank $r --world-size $W "$@" > $OUT/run-rank$r.out 2>&1 &
done
wait
python experiments/layer_speaker_probe.py report --out $OUT > $OUT/report.out 2>&1
echo "[$(date "+%F %T")] PROBE_DONE" >> $OUT/done.txt
