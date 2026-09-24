#!/usr/bin/env bash
# semcommit labeling recipe v0.3 on rack4 GPU 0: judge only the extra candidates (last word · segment ends · punctuated sentence ends),
# tune v0.3 thresholds on the gold dev half, grade all sets, score vs gold, train r3 (<SEM_END> only) and evaluate it vs gold.
set -uo pipefail
cd /home/tskim/VAP
export HF_HOME=/data3/tskim/cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/opt/conda/envs/vapasr/bin/python; T="$P experiments/semcommit_teacher.py"; GT="$P experiments/semcommit_gold.py"; G="--gpu 0"
GD=/data4/tskim/semcommit/gold/v1; V0=/data4/tskim/semcommit/data/v0; L2=/data4/tskim/semcommit/labels/v0.2; LG=/data4/tskim/semcommit/labels/v0.2-gold
L3=/data4/tskim/semcommit/labels/v0.3; mkdir -p $L3
Q=/data4/tskim/OpenSource/qwen3-8b; X=/data4/tskim/OpenSource/EXAONE-3.5-7.8B-Instruct-rev0ff6b5e
EXTRA="--extra-candidates last,seg_end,punct_final --only-extra"
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || { echo "!! FAILED rc=$rc: $*"; exit $rc; }; }
declare -A W=([ls-test]=$GD/words-ls-test.jsonl [gs-test]=$GD/words-gs-test.jsonl [ks-eval]=$GD/words-ks-eval.jsonl [ks-long]=$GD/words-ks-long.jsonl [ls-train]=$V0/words-ls-train.jsonl [ks-train]=$V0/words-ks-train.jsonl)
declare -A BASE=([ls-test]=$L2 [ks-eval]=$L2 [ls-train]=$L2 [ks-train]=$L2 [gs-test]=$LG [ks-long]=$LG)
SETS="ls-test gs-test ks-eval ks-long ls-train ks-train"
for s in $SETS; do step $T stageB --words ${W[$s]} --stageA ${BASE[$s]}/A-$s.jsonl --model $Q --kind qwen3 --out $L3/B.qwen3-$s.x.jsonl $G --batch-size 32 $EXTRA; done
for s in $SETS; do step $T stageC --words ${W[$s]} --stageA ${BASE[$s]}/A-$s.jsonl --model $Q --kind qwen3 --out $L3/C-$s.x.jsonl $G --batch-size 32 $EXTRA; done
for s in $SETS; do step $T stageB --words ${W[$s]} --stageA ${BASE[$s]}/A-$s.jsonl --model $X --kind exaone35 --out $L3/B.exaone35-$s.x.jsonl $G --batch-size 32 $EXTRA; done
# thresholds: gold dev half (sha1 parity), precision floor 0.90
GS="ls-test gs-test ks-eval ks-long"; A=""; B=""; C=""; WW=""; GG=""
for s in $GS; do A="$A ${BASE[$s]}/A-$s.jsonl"; B="$B ${BASE[$s]}/B.qwen3-$s.jsonl ${BASE[$s]}/B.exaone35-$s.jsonl $L3/B.qwen3-$s.x.jsonl $L3/B.exaone35-$s.x.jsonl"; C="$C ${BASE[$s]}/C-$s.jsonl $L3/C-$s.x.jsonl"; WW="$WW ${W[$s]}"; GG="$GG $GD/gold-$s.jsonl"; done
step $GT tune --words $WW --gold $GG --stageA $A --stageB $B --stageC $C --floor 0.90 --out $L3/thresholds.json --report $L3/thresholds.report.json
for s in $SETS; do
  if [[ $s == ls-* || $s == gs-* ]]; then J="--judges-en qwen3,exaone35"; else J="--judges-ko exaone35,qwen3"; fi
  step $T grade --recipe v0.3 --thresholds $L3/thresholds.json --words ${W[$s]} --stageA ${BASE[$s]}/A-$s.jsonl \
       --stageB ${BASE[$s]}/B.qwen3-$s.jsonl ${BASE[$s]}/B.exaone35-$s.jsonl $L3/B.qwen3-$s.x.jsonl $L3/B.exaone35-$s.x.jsonl \
       --stageC ${BASE[$s]}/C-$s.jsonl $L3/C-$s.x.jsonl $J --out $L3/labels-$s.jsonl --stats $L3/labels-$s.stats.json
done
mkdir -p $GD/scores
for s in $GS; do $GT score-labels --words ${W[$s]} --labels $L3/labels-$s.jsonl --gold $GD/gold-$s.jsonl > $GD/scores/teacher-v0.3-$s.json; echo "[v0.3 labels vs gold] $s $(cat $GD/scores/teacher-v0.3-$s.json)"; done
echo "[$(date "+%F %T")] LABELS_V03_DONE"
# r3: E2 + <SEM_END> only, v0.3 labels (same training settings as r1/r2)
NEMO=/data3/tskim/cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/1c8deaecc64b91f034d73e08dd8b64625eb3395d
step $P experiments/semcommit_train.py --init /data4/tskim/VAPASR/exports/hf-E2-final --nemotron-dir $NEMO \
  --train-words $V0/words-ls-train.jsonl --train-labels $L3/labels-ls-train.jsonl --train-words $V0/words-ks-train.jsonl --train-labels $L3/labels-ks-train.jsonl \
  --out-dir /data4/tskim/semcommit/runs/v0.3-r3 --epochs 3 --lr 3e-5 --lr-adapter 1e-4 --warmup 50 --bs-en 6 --bs-ko 16 --save-every 300 --save-total-limit 0 --log-every 10 --num-workers 6 --gpu 0
grep -q "PARITY OK" /data4/tskim/semcommit/runs/v0.3-r3/parity.json 2>/dev/null || $P -c "import json,sys; d=json.load(open('/data4/tskim/semcommit/runs/v0.3-r3/parity.json')); sys.exit(0 if d['parity'] else 1)" || { echo "!! FAILED parity"; exit 1; }
O=/data4/tskim/semcommit/eval/v0.3-r3; mkdir -p $O
for s in $GS; do
  step $P experiments/semcommit_eval.py --model /data4/tskim/semcommit/runs/v0.3-r3/final --encoder $NEMO --words ${W[$s]} --labels $L3/labels-$s.jsonl --delay 4 --sem-bias -2 -1 0 1 2 --theta 0.2 0.35 0.5 --gpu 0 --batch-size 32 --out $O/r3-d4-$s.json
  $GT score-eval --words ${W[$s]} --gold $GD/gold-$s.jsonl --streams $O/r3-d4-$s.streams.jsonl --out $GD/scores/models-r3-$s.json
done
echo "[$(date "+%F %T")] RECIPE_V03_DONE"
