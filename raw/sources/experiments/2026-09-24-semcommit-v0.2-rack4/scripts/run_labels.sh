#!/usr/bin/env bash
# semcommit v0.2 LLM relabeling on rack4 GPU 0 (one model at a time). Resumable: rerun this script to continue.
set -uo pipefail
cd /home/tskim/VAP
export HF_HOME=/data3/tskim/cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/opt/conda/envs/vapasr/bin/python; T="$P experiments/semcommit_teacher.py"; G="--gpu 0"
D=/data4/tskim/semcommit/data/v0; L=/data4/tskim/semcommit/labels/v0.2
Q=/data4/tskim/OpenSource/qwen3-8b; X=/data4/tskim/OpenSource/EXAONE-3.5-7.8B-Instruct-rev0ff6b5e; O=/data4/tskim/OpenSource/gpt-oss-20b
SETS="ks-eval ls-test ks-train ls-train"
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || echo "!! FAILED rc=$rc: $*"; }
for s in $SETS; do step $T stageA --words $D/words-$s.jsonl --model $Q --kind qwen3 --out $L/A-$s.jsonl $G --batch-size 8; done
for s in $SETS; do step $T stageB --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --model $Q --kind qwen3 --out $L/B.qwen3-$s.jsonl $G --batch-size 32; done
for s in $SETS; do step $T stageC --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --model $Q --kind qwen3 --out $L/C-$s.jsonl $G --batch-size 32; done
for s in ks-eval ks-train; do step $T stageB --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --model $X --kind exaone35 --out $L/B.exaone35-$s.jsonl $G --batch-size 32 --lang-filter Korean; done
for s in ls-test ls-train; do step $T stageB --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --model $O --kind gptoss --out $L/B.gptoss-$s.jsonl $G --batch-size 16 --lang-filter English; done
for s in ks-eval ks-train; do step $T tiebreak --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --stageB $L/B.exaone35-$s.jsonl $L/B.qwen3-$s.jsonl --model $O --kind gptoss --out $L/T-$s.jsonl $G --batch-size 16; done
for s in ks-eval ks-train; do step $T grade --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --stageB $L/B.exaone35-$s.jsonl $L/B.qwen3-$s.jsonl --stageC $L/C-$s.jsonl --tiebreak $L/T-$s.jsonl --out $L/labels-$s.jsonl --stats $L/labels-$s.stats.json; done
for s in ls-test ls-train; do step $T grade --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --stageB $L/B.qwen3-$s.jsonl $L/B.gptoss-$s.jsonl --stageC $L/C-$s.jsonl --out $L/labels-$s.jsonl --stats $L/labels-$s.stats.json; done
echo "[$(date "+%F %T")] LABELS_DONE"
