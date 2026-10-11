#!/bin/bash
# q06-s2-ast/final full evaluation: shard queue over mxc host GPUs (per-process env only; no deletion).
#   Tasks = (manifest, out, world, rank); each GPU worker takes the next task under flock, then every out is summarized.
set -u
PROJECT=/soundai/users/tskim/VAPKT; DATA=/soundai/users/tskim/VAPKT-data
PY=$DATA/conda/envs/vapasr/bin/python
CK=/soundai/Model/VAPASR/semcommit-q06-s2-ast/final
ENC=/soundai/Model/nemotron-3.5-asr-streaming-0.6b
EV=$DATA/data/evaluation; RES=$DATA/results
Q=$RES/evalq-q06-s2-ast; mkdir -p $Q
GPUS=${GPUS:-0 1 2 5}
A=$EV/ast-fleurs-covost2-v1
[ -f $A/manifest-max30s-fleurs.jsonl ] || grep '"dataset": "fleurs-' $A/manifest-max30s.jsonl > $A/manifest-max30s-fleurs.jsonl
[ -f $A/manifest-max30s-covost2.jsonl ] || grep '"dataset": "covost2-' $A/manifest-max30s.jsonl > $A/manifest-max30s-covost2.jsonl
GROUPS_=(
  "$A/manifest-max30s-fleurs.jsonl $RES/ast-fleurs-q06-s2-ast 4"
  "$EV/single-turn-vpaa-kspon-v1/manifest.jsonl $RES/single-turn-vpaa-kspon-q06-s2-ast 4"
  "$EV/single-turn-yodas-de-dev-v1/manifest.jsonl $RES/single-turn-yodas-de-q06-s2-ast 4"
  "$EV/single-turn-librispeech-v1/manifest-test-clean.jsonl $RES/single-turn-librispeech-clean-q06-s2-ast 2"
  "$EV/single-turn-librispeech-v1/manifest-test-other.jsonl $RES/single-turn-librispeech-other-q06-s2-ast 2"
  "$A/manifest-max30s-covost2.jsonl $RES/ast-covost2-q06-s2-ast 8"
)
if [ ! -f $Q/todo ]; then
  for g in "${GROUPS_[@]}"; do set -- $g; for ((r = 0; r < $3; r++)); do echo "$1 $2 $3 $r"; done; done > $Q/todo
fi
wc -l $A/manifest-max30s-fleurs.jsonl $A/manifest-max30s-covost2.jsonl $Q/todo
worker() {
  local gpu=$1 t
  while :; do
    t=$(flock $Q/lock bash -c 'f='$Q'/todo; l=$(head -1 $f); [ -n "$l" ] && sed -i 1d $f; echo "$l"')
    [ -z "$t" ] && break
    set -- $t; mkdir -p $2
    echo "$(date +%T) gpu=$gpu start $t" >> $Q/driver.log
    if (cd $PROJECT && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PROJECT OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VAPASR_STAGE_DIR=/dev/shm/tskim-vapasr-infer-35000 \
        $PY experiments/eval_single_turn_asr.py run --manifest $1 --checkpoint $CK --encoder $ENC --out $2 \
        --deltas 4 6 8 10 --tail-s 1.0 --max-flush 8 --batch-size 64 --rank $4 --world-size $3 --limit 0 --verify --final \
        >> $2/rank$4.log 2>&1); then
      echo "$(date +%T) gpu=$gpu done $t" >> $Q/driver.log; echo "$t" >> $Q/done
    else
      echo "$(date +%T) gpu=$gpu FAIL $t" >> $Q/driver.log; echo "$t" >> $Q/failed
    fi
  done
}
for g in $GPUS; do worker $g & done
wait
for g in "${GROUPS_[@]}"; do set -- $g
  (cd $PROJECT && PYTHONPATH=$PROJECT $PY experiments/eval_single_turn_asr.py summarize --out $2 > $2/summary.log 2>&1) \
    && echo "$(date +%T) summarized $2" >> $Q/driver.log || echo "$(date +%T) SUMMARY FAIL $2" >> $Q/driver.log
done
echo "$(date +%T) ALL DONE" >> $Q/driver.log
