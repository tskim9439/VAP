#!/usr/bin/env bash
# v0.3.3: v0.3.2 + rule negatives (reply_prefix · conn_final · conn_mid → N, semcommit_llm.rule_negatives), tuned per branch with rule positions excluded from A. CPU only.
set -uo pipefail
cd /home/tskim/VAP
P=/opt/conda/envs/vapasr/bin/python; T="$P experiments/semcommit_teacher.py"; GT="$P experiments/semcommit_gold.py"
GD=/data4/tskim/semcommit/gold/v1; V0=/data4/tskim/semcommit/data/v0; L2=/data4/tskim/semcommit/labels/v0.2; LG=/data4/tskim/semcommit/labels/v0.2-gold
L3=/data4/tskim/semcommit/labels/v0.3; L31=/data4/tskim/semcommit/labels/v0.3.3; mkdir -p $L31
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || { echo "!! FAILED rc=$rc: $*"; exit $rc; }; }
declare -A W=([ls-test]=$GD/words-ls-test.jsonl [gs-test]=$GD/words-gs-test.jsonl [ks-eval]=$GD/words-ks-eval.jsonl [ks-long]=$GD/words-ks-long.jsonl [ls-train]=$V0/words-ls-train.jsonl [ks-train]=$V0/words-ks-train.jsonl)
declare -A BASE=([ls-test]=$L2 [ks-eval]=$L2 [ls-train]=$L2 [ks-train]=$L2 [gs-test]=$LG [ks-long]=$LG)
SETS="ls-test gs-test ks-eval ks-long ls-train ks-train"; GS="ls-test gs-test ks-eval ks-long"; A=""; B=""; C=""; WW=""; GG=""
for s in $GS; do A="$A ${BASE[$s]}/A-$s.jsonl"; B="$B ${BASE[$s]}/B.qwen3-$s.jsonl ${BASE[$s]}/B.exaone35-$s.jsonl $L3/B.qwen3-$s.x.jsonl $L3/B.exaone35-$s.x.jsonl"; C="$C ${BASE[$s]}/C-$s.jsonl $L3/C-$s.x.jsonl"; WW="$WW ${W[$s]}"; GG="$GG $GD/gold-$s.jsonl"; done
step $GT tune --words $WW --gold $GG --stageA $A --stageB $B --stageC $C --floor 0.90 --out $L31/thresholds.json --report $L31/thresholds.report.json
for s in $SETS; do
  if [[ $s == ls-* || $s == gs-* ]]; then J="--judges-en qwen3,exaone35"; else J="--judges-ko exaone35,qwen3"; fi
  step $T grade --recipe v0.3 --thresholds $L31/thresholds.json --words ${W[$s]} --stageA ${BASE[$s]}/A-$s.jsonl \
       --stageB ${BASE[$s]}/B.qwen3-$s.jsonl ${BASE[$s]}/B.exaone35-$s.jsonl $L3/B.qwen3-$s.x.jsonl $L3/B.exaone35-$s.x.jsonl \
       --stageC ${BASE[$s]}/C-$s.jsonl $L3/C-$s.x.jsonl $J --out $L31/labels-$s.jsonl --stats $L31/labels-$s.stats.json
done
for s in $GS; do $GT score-labels --words ${W[$s]} --labels $L31/labels-$s.jsonl --gold $GD/gold-$s.jsonl > $GD/scores/teacher-v0.3.3-$s.json; echo "[v0.3.3 labels vs gold] $s $(cat $GD/scores/teacher-v0.3.3-$s.json)"; done
echo "[$(date "+%F %T")] LABELS_V033_DONE"
