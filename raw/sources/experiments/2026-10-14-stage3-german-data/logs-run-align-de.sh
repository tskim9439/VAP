# 표준 철자로 바꾼 KEC·ZTC 를 새 정렬 폴더(align-asr-tn-de-v1)에 다시 정렬한다. align-asr-tn-v1/{kec,ztc} 는 축약 표기본 정렬(쓰지 않음, 지우지 않음).
# NSC 의사 전사와 GPU 를 나눠 쓰므로 GPU 마다 워커 1. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export MXC_QWEN_ASR_DIR=/soundai/Model/Qwen3-ASR-0.6B MXC_ALIGNER_DIR=/soundai/Model/Qwen3-ForcedAligner-0.6B
D=/soundai/users/tskim/VAPKT-data/data/phase2; L=$D/logs-de; A=$D/align-asr-tn-de-v1
GPUS="0 4 6"; N=3
for C in ${CORPORA:-kec ztc}; do
  echo "[$(date '+%F %T')] $C align (de-v1)" >> $L/status; i=0
  for g in $GPUS; do
    CUDA_VISIBLE_DEVICES=$g $P experiments/p2_align.py --dialogues $D/$C.dialogues.jsonl --out-root $A --shard $i/$N --gpu 0 --mem-frac 0.12 > $L/align-de-$C-$i.log 2>&1 &
    i=$((i + 1)); sleep 5
  done; wait
  echo "[$(date '+%F %T')] $C align (de-v1) done" >> $L/status
done
