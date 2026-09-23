#!/usr/bin/env bash
# semcommit v0.2 r1 evaluation on rack4 GPU 0: oracle (metric upper bound), E2-sem-init baseline, r1 final (δ4 sweep + δ2 sweep). Resumable per stream.
set -uo pipefail
cd /home/tskim/VAP
export HF_HOME=/data3/tskim/cache/huggingface TOKENIZERS_PARALLELISM=false
P=/opt/conda/envs/vapasr/bin/python; E="$P experiments/semcommit_eval.py"
D=/data4/tskim/semcommit/data/v0; L=/data4/tskim/semcommit/labels/v0.2; O=/data4/tskim/semcommit/eval/v0.2-r1; mkdir -p $O
NEMO=/data3/tskim/cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/1c8deaecc64b91f034d73e08dd8b64625eb3395d
R1=/data4/tskim/semcommit/runs/v0.2-r1/final; B0=/data4/tskim/semcommit/models/E2-sem-init
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || echo "!! FAILED rc=$rc: $*"; }
for s in ls-test ks-eval; do
  step $E --model $R1 --words $D/words-$s.jsonl --labels $L/labels-$s.jsonl --delay 4 --oracle --out $O/oracle-d4-$s.json
  step $E --model $R1 --encoder $NEMO --words $D/words-$s.jsonl --labels $L/labels-$s.jsonl --delay 4 --sem-bias -2 -1 0 1 2 --theta 0.2 0.35 0.5 --gpu 0 --batch-size 32 --verify --out $O/r1-d4-$s.json
  step $E --model $B0 --encoder $NEMO --words $D/words-$s.jsonl --labels $L/labels-$s.jsonl --delay 4 --sem-bias 0 --gpu 0 --batch-size 32 --out $O/base-d4-$s.json
  step $E --model $R1 --encoder $NEMO --words $D/words-$s.jsonl --labels $L/labels-$s.jsonl --delay 2 --sem-bias -1 0 1 --gpu 0 --batch-size 32 --out $O/r1-d2-$s.json
done
echo "[$(date "+%F %T")] EVAL_DONE"
