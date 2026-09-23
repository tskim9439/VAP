#!/usr/bin/env bash
# semcommit v0.2 part 2 (rack4 GPU 0). gpt-oss-20b answered WAIT on 570/587 ls-test candidates (SAFE 17) → dropped as the
# English primary judge; EXAONE-3.5 (bilingual) is the second English judge instead. gpt-oss stays the Korean tie-break,
# run on ks-eval only as a diagnostic (strict grading keeps judge-resolved decisions at B). Grading masks CONTINUATION and
# QUALIFICATION REVISION calls (Qwen3 Stage C over-calls QUALIFICATION on v0.2). Resumable: rerun to continue.
set -uo pipefail
cd /home/tskim/VAP
export HF_HOME=/data3/tskim/cache/huggingface TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/opt/conda/envs/vapasr/bin/python; T="$P experiments/semcommit_teacher.py"; G="--gpu 0"
D=/data4/tskim/semcommit/data/v0; L=/data4/tskim/semcommit/labels/v0.2
X=/data4/tskim/OpenSource/EXAONE-3.5-7.8B-Instruct-rev0ff6b5e; O=/data4/tskim/OpenSource/gpt-oss-20b
MASK="--c-mask-types CONTINUATION,QUALIFICATION"
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || echo "!! FAILED rc=$rc: $*"; }
for s in ls-test ls-train; do step $T stageB --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --model $X --kind exaone35 --out $L/B.exaone35-$s.jsonl $G --batch-size 32 --lang-filter English; done
step $T tiebreak --words $D/words-ks-eval.jsonl --stageA $L/A-ks-eval.jsonl --stageB $L/B.exaone35-ks-eval.jsonl $L/B.qwen3-ks-eval.jsonl --model $O --kind gptoss --out $L/T-ks-eval.jsonl $G --batch-size 16
step $T grade --words $D/words-ks-eval.jsonl --stageA $L/A-ks-eval.jsonl --stageB $L/B.exaone35-ks-eval.jsonl $L/B.qwen3-ks-eval.jsonl --stageC $L/C-ks-eval.jsonl --tiebreak $L/T-ks-eval.jsonl $MASK --out $L/labels-ks-eval.jsonl --stats $L/labels-ks-eval.stats.json
step $T grade --words $D/words-ks-train.jsonl --stageA $L/A-ks-train.jsonl --stageB $L/B.exaone35-ks-train.jsonl $L/B.qwen3-ks-train.jsonl --stageC $L/C-ks-train.jsonl $MASK --out $L/labels-ks-train.jsonl --stats $L/labels-ks-train.stats.json
for s in ls-test ls-train; do step $T grade --words $D/words-$s.jsonl --stageA $L/A-$s.jsonl --stageB $L/B.qwen3-$s.jsonl $L/B.exaone35-$s.jsonl --stageC $L/C-$s.jsonl --judges-en qwen3,exaone35 $MASK --out $L/labels-$s.jsonl --stats $L/labels-$s.stats.json; done
echo "[$(date "+%F %T")] LABELS_DONE"
