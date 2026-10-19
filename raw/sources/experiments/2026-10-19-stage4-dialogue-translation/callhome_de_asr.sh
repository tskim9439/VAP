# CallHome 독일어 의사 전사(mxc 호스트): Qwen3-ASR-1.7B + Whisper large-v3, GPU 셋 × 워커 2 (hibiki overlay venv — 휠 wheels/overlay-hibiki 로 다시 만든 것).
# 재개 가능(완료 세션 건너뜀). 끝나면 build(WER < 0.03 만 감독). 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
BL=/soundai/users/tskim/VAPKT-data/vapkt-baselines; H=/tmp/sa_tskim/bl-envs/hibiki/bin/python; P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT:$BL/code/SimulStreaming PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
W=/soundai/users/tskim/VAPKT-data/data/callhome-deu/work; P2=/soundai/users/tskim/VAPKT-data/data/phase2
GPUS=${GPUS:-"4 5 6"}; WK=2; N=$(( $(echo $GPUS | wc -w) * WK )); i=0
echo "[$(date '+%F %T')] asr start ($N shards, GPU $GPUS)"
for g in $GPUS; do for w in $(seq 1 $WK); do
  CUDA_VISIBLE_DEVICES=$g $H experiments/callhome_de_pseudo_label.py asr --out $W --shard $i/$N --mem-frac 0.2 > $W/asr-$i.log 2>&1 &
  i=$((i + 1)); sleep 5
done; done; wait
echo "[$(date '+%F %T')] asr done: $(cat $W/asr/shard-*-of-$N.jsonl | wc -l) sessions"
$P experiments/callhome_de_pseudo_label.py build --out $W --dest $P2 --max-wer 0.03
echo "CHDE_ASR_DONE $(date '+%F %T')"
