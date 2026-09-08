S=/soundai/Model/VAPASR/hf-C2/eval-ckpts; R=/soundai/Model/VAPASR/hf-C2
for n in 21960 12000 14000 16000 18000 20000; do
  [ -f $R/eval/select-$n.json ] && continue
  echo "[$(date '+%F %T')] select eval checkpoint-$n"
  CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc_per_node=2 --master_port=29591 experiments/s3_train_hf.py --eval-only --init $S/checkpoint-$n --out-dir $R --sentinel-stream 50 --sentinel-utt 300 --eval-seed 7 --eval-tag select --num-workers 0 2>&1 | grep -E "select_score|Traceback|Error" | cut -c1-600
done
echo "[$(date '+%T')] select evals done"
