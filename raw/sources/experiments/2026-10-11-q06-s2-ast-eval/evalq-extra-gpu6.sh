#!/bin/bash
# Extra workers for evalq-q06-s2-ast: pop the remaining tasks now and run them all on host GPU 6 concurrently (same command as driver.sh),
# so every shard records today's code hashes before ast_dataset.py changes. The main driver's own summarize may run before these finish;
# summarize is re-run by hand afterwards.
set -u
PROJECT=/soundai/users/tskim/VAPKT; DATA=/soundai/users/tskim/VAPKT-data
PY=$DATA/conda/envs/vapasr/bin/python
CK=/soundai/Model/VAPASR/semcommit-q06-s2-ast/final
ENC=/soundai/Model/nemotron-3.5-asr-streaming-0.6b
Q=$DATA/results/evalq-q06-s2-ast; gpu=6
run_task() {
  set -- $1
  echo "$(date +%T) gpu=$gpu start $*" >> $Q/driver.log
  if (cd $PROJECT && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PROJECT OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VAPASR_STAGE_DIR=/dev/shm/tskim-vapasr-infer-35000 \
      $PY experiments/eval_single_turn_asr.py run --manifest $1 --checkpoint $CK --encoder $ENC --out $2 \
      --deltas 4 6 8 10 --tail-s 1.0 --max-flush 8 --batch-size 64 --rank $4 --world-size $3 --limit 0 --verify --final \
      >> $2/rank$4.log 2>&1); then
    echo "$(date +%T) gpu=$gpu done $*" >> $Q/driver.log; echo "$*" >> $Q/done
  else
    echo "$(date +%T) gpu=$gpu FAIL $*" >> $Q/driver.log; echo "$*" >> $Q/failed
  fi
}
while :; do
  t=$(flock $Q/lock bash -c 'f='$Q'/todo; l=$(head -1 $f); [ -n "$l" ] && sed -i 1d $f; echo "$l"')
  [ -z "$t" ] && break
  run_task "$t" &
  sleep 20
done
wait
echo "$(date +%T) gpu=$gpu extra workers finished" >> $Q/driver.log
