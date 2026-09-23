#!/usr/bin/env bash
# After the r1 eval finishes (GPU 0 free): train r2 (no hard-negative upweighting) then evaluate it like r1 at δ4.
set -uo pipefail
until grep -q "EVAL_DONE" /data3/tskim/logs/semcommit-eval-r1.log; do sleep 30; done
bash /data4/tskim/semcommit/tools/train_r2.sh > /data3/tskim/logs/semcommit-train-r2.log 2>&1
grep -q "PARITY OK" /data3/tskim/logs/semcommit-train-r2.log || { echo "r2 train failed or parity not OK"; exit 1; }
cd /home/tskim/VAP
D=/data4/tskim/semcommit/data/v0; L=/data4/tskim/semcommit/labels/v0.2; O=/data4/tskim/semcommit/eval/v0.2-r2; mkdir -p $O
NEMO=/data3/tskim/cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/1c8deaecc64b91f034d73e08dd8b64625eb3395d
for s in ls-test ks-eval; do
  echo "[$(date "+%F %T")] >>> r2 eval $s"
  /opt/conda/envs/vapasr/bin/python experiments/semcommit_eval.py --model /data4/tskim/semcommit/runs/v0.2-r2/final --encoder $NEMO --words $D/words-$s.jsonl --labels $L/labels-$s.jsonl \
    --delay 4 --sem-bias -2 -1 0 1 2 --theta 0.2 0.35 0.5 --gpu 0 --batch-size 32 --out $O/r2-d4-$s.json; rc=$?
  echo "[$(date "+%F %T")] <<< rc=$rc"
done
echo "[$(date "+%F %T")] R2_DONE"
