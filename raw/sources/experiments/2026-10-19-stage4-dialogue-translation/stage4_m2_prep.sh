# st4-m2 준비(mxc 호스트): 전화 대역 증강 테스트 → CallHome 독일어 평가 셋 → 기준점(st4-m1 ck1500·p11) CallHome 평가 → 학습 dry-run·스모크(stage4_m2_check.sh). 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
$P -m pytest -q tests/test_phone_aug.py 2>&1 | tail -1
E=/soundai/users/tskim/VAPKT-data/data/evaluation/single-turn-callhome-de-v1
[ -f $E/manifest.jsonl ] || $P raw/sources/experiments/2026-10-19-stage4-dialogue-translation/build_chde_eval.py /soundai/users/tskim/VAPKT-data/data/callhome-deu/words-v1/callhome-de-words-heldout.jsonl $E
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F", " "{print \$1, \$3 - \$2, \$4}" | sort -k2,2nr -k3,3n | head -1 | cut -d" " -f1)
for x in "st4-m1-ck1500 /soundai/Model/VAPASR/stage3-st4-m1/checkpoint-1500" "st3-p11-revive-ck3000 /soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000"; do
  set -- $x; O=$R/stage3-regress-$1/callhome-de; mkdir -p $O
  [ -f $O/summary.log ] || { CUDA_VISIBLE_DEVICES=$GPU $P experiments/eval_single_turn_asr.py run --manifest $E/manifest.jsonl --checkpoint $2 --encoder $MXC_NEMOTRON_DIR --out $O --deltas 4 10 --batch-size 64 > $O/run.log 2>&1
    $P experiments/eval_single_turn_asr.py summarize --out $O > $O/summary.log 2>&1; }
  echo "== $1"; grep -i -E "callhome|wer" $O/summary.log | head -4
done
bash raw/sources/experiments/2026-10-19-stage4-dialogue-translation/stage4_m2_check.sh 2>&1 | grep -E "phone|rep_chde|불일치 [1-9]|DRY EXIT|SMOKE EXIT|PARITY|Error|Traceback|OOM" | head -20
echo "M2_PREP_DONE $(date '+%F %T')"
