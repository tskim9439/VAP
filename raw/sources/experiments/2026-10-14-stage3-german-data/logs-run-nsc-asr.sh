# NSC 의사 전사: Qwen3-ASR-1.7B + Whisper large-v3, 호스트 GPU 0·4·6 × 워커 3. 재개 가능(완료 세션 건너뜀). 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
BL=/soundai/users/tskim/VAPKT-data/vapkt-baselines; H=/tmp/sa_tskim/bl-envs/hibiki/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT:$BL/code/SimulStreaming PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
D=/soundai/users/tskim/VAPKT-data/data/phase2; L=$D/logs-de
GPUS="0 4 6"; W=3; N=$(( $(echo $GPUS | wc -w) * W )); i=0
echo "[$(date '+%F %T')] nsc asr start ($N shards)" >> $L/status
for g in $GPUS; do for w in $(seq 1 $W); do
  CUDA_VISIBLE_DEVICES=$g $H experiments/nsc_pseudo_label.py asr --out $D/nsc-work --shard $i/$N --mem-frac 0.26 > $L/nsc-asr-$i.log 2>&1 &
  i=$((i + 1)); sleep 5
done; done; wait
echo "[$(date '+%F %T')] nsc asr done: $(cat $D/nsc-work/nsc-asr/shard-*-of-$N.jsonl | wc -l) sessions" >> $L/status
