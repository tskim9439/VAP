# otoSpeech 재빌드 3(2026-10-15): 하위 구간 합치기 + 비음성 밖 모든 SRT 라벨을 발화로. 직전 산출(합치기만)을 *.v2-mergeonly.* 로 복사해 두고 빌드 → 정렬(align-asr-tn-v1-oto3, GPU 2) → 정제.
# 아무것도 지우지 않는다(옛 정렬 align-asr-tn-v1/otoSpeech·…-oto2 는 그대로 둔다).
set -u
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data/data/phase2; L=$D/logs-oto3; mkdir -p $L
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
for f in otoSpeech.dialogues.jsonl otoSpeech.stats.json otoSpeech.refined.dialogues.jsonl otoSpeech.refined.dialogues.stats.json; do
  cp --update=none $D/$f $D/${f/otoSpeech./otoSpeech.v2-mergeonly.}
done
echo "[$(date '+%F %T')] build" >> $L/status
$P experiments/p2_build_dialogues.py --corpus otoSpeech --root /soundai/DB/raw/otoSpeech16k --out $D > $L/build.log 2>&1 || { echo "build 실패" >> $L/status; exit 1; }
tail -1 $L/build.log >> $L/status
echo "[$(date '+%F %T')] align" >> $L/status
export MXC_QWEN_ASR_DIR=/soundai/Model/Qwen3-ASR-0.6B MXC_ALIGNER_DIR=/soundai/Model/Qwen3-ForcedAligner-0.6B
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=2 $P experiments/p2_align.py --dialogues $D/otoSpeech.dialogues.jsonl --out-root $D/align-asr-tn-v1-oto3 --shard $i/3 --gpu 0 --mem-frac 0.26 > $L/align-$i.log 2>&1 &
  sleep 5
done; wait
echo "[$(date '+%F %T')] refine" >> $L/status
OMP_NUM_THREADS=2 $P experiments/p2_refine.py --dialogues $D/otoSpeech.dialogues.jsonl --align $D/align-asr-tn-v1-oto3/otoSpeech --workers 12 > $L/refine.log 2>&1 || { echo "refine 실패" >> $L/status; exit 1; }
tail -1 $L/refine.log >> $L/status
echo "[$(date '+%F %T')] DONE" >> $L/status
