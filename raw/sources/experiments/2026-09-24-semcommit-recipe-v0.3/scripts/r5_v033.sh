#!/usr/bin/env bash
# r5: same settings as r3 but v0.3.3 labels (v0.3.2 + rule negatives reply_prefix · conn_final · conn_mid). Eval: bias −2…2 only.
set -uo pipefail
cd /home/tskim/VAP
export HF_HOME=/data3/tskim/cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/opt/conda/envs/vapasr/bin/python; GT="$P experiments/semcommit_gold.py"
GD=/data4/tskim/semcommit/gold/v1; V0=/data4/tskim/semcommit/data/v0; L31=/data4/tskim/semcommit/labels/v0.3.3
R=/data4/tskim/semcommit/runs/v0.3.3-r5; O=/data4/tskim/semcommit/eval/v0.3.3-r5
NEMO=/data3/tskim/cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/1c8deaecc64b91f034d73e08dd8b64625eb3395d
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || { echo "!! FAILED rc=$rc: $*"; exit $rc; }; }
bash /data4/tskim/semcommit/tools/retune_v033.sh || { echo "!! FAILED retune_v033"; exit 1; }
echo "[$(date "+%F %T")] waiting for GPU jobs"
until grep -qE "RECIPE_V03_DONE|!! FAILED" /data3/tskim/logs/semcommit-recipe-v03.log; do sleep 30; done
while pgrep -f "semcommit_(eval|train).py" > /dev/null; do sleep 30; done
declare -A W=([ls-test]=$GD/words-ls-test.jsonl [gs-test]=$GD/words-gs-test.jsonl [ks-eval]=$GD/words-ks-eval.jsonl [ks-long]=$GD/words-ks-long.jsonl)
step $P experiments/semcommit_train.py --init /data4/tskim/VAPASR/exports/hf-E2-final --nemotron-dir $NEMO \
  --train-words $V0/words-ls-train.jsonl --train-labels $L31/labels-ls-train.jsonl --train-words $V0/words-ks-train.jsonl --train-labels $L31/labels-ks-train.jsonl \
  --out-dir $R --epochs 3 --lr 3e-5 --lr-adapter 1e-4 --warmup 50 --bs-en 6 --bs-ko 16 --save-every 300 --save-total-limit 0 --log-every 10 --num-workers 6 --gpu 0
$P -c "import json,sys; d=json.load(open('$R/parity.json')); print('PARITY', d.get('parity'), {k: d[k] for k in d if k != 'parity'} if len(str(d)) < 600 else ''); sys.exit(0 if d['parity'] else 1)" || { echo "!! FAILED parity"; exit 1; }
mkdir -p $O
for s in ls-test gs-test ks-eval ks-long; do
  step $P experiments/semcommit_eval.py --model $R/final --encoder $NEMO --words ${W[$s]} --labels $L31/labels-$s.jsonl --delay 4 --sem-bias -2 -1 0 1 2 --gpu 0 --batch-size 32 --out $O/r5-d4-$s.json
  $GT score-eval --words ${W[$s]} --gold $GD/gold-$s.jsonl --streams $O/r5-d4-$s.streams.jsonl --out $GD/scores/models-r5-$s.json
done
echo "[$(date "+%F %T")] R5_DONE"
