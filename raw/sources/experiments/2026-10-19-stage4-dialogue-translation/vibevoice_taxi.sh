# VibeVoice 외부 기준선 — TAXI(실제 독↔영 전화 86 세션): 세션 통째 디코드 + 세션 단위 cpWER(cst_stage4_taxi 의 wer_cp 와 같은 정의). GPU 둘 쉼표.
# 산출 results/baseline-vibevoice-{stream,offline}-taxi/{taxi-pred.jsonl(mxc 에만), taxi-report.json}. 아무것도 지우지 않는다.
set -u
G1=${1%%,*}; G2=${1##*,}
V=/soundai/users/tskim/VAPKT-data/models/vibevoice; R=/soundai/users/tskim/VAPKT-data/results; TAXI=/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; SRC=$(ls -d $V/src/VibeVoice-* | head -1)
export PYTHONPATH=/soundai/users/tskim/VAPKT:$V/pydeps PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=$G1 $P experiments/baseline_vibevoice.py taxi --mode stream --model $V/VibeVoice-ASR-Streaming-7B --src $SRC --sessions $TAXI --out $R/baseline-vibevoice-stream-taxi > $R/baseline-vibevoice-stream-taxi.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 $P experiments/baseline_vibevoice.py taxi --mode offline --model $V/VibeVoice-ASR --src $SRC --lm-tokenizer $V/Qwen2.5-7B-tokenizer --sessions $TAXI --out $R/baseline-vibevoice-offline-taxi > $R/baseline-vibevoice-offline-taxi.log 2>&1 &
wait
for m in stream offline; do echo "== $m"; tail -1 $R/baseline-vibevoice-$m-taxi.log; done
echo "VV_TAXI_DONE"
