# V2 M0 — 인코더 층별 화자 분리도(mxc 호스트 GPU 하나): Nemotron 원본 → Stage 1 → 2 → 3 → 4(V1 = st4-m1 ck1500), Stage 3 회귀 창(코퍼스마다 200).
# 요약은 V1 의 오류 분해(stage3-attr-errors-st4-m1-ck1500)와 묶는다. 산출 results/spk-probe-m0. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; D=/soundai/users/tskim/VAPKT-data/data/phase2; M=/soundai/Model/VAPASR
G=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F", " "{print \$1, \$3 - \$2, \$4}" | sort -k2,2nr -k3,3n | head -1 | cut -d" " -f1)
CUDA_VISIBLE_DEVICES=$G $P experiments/spk_probe.py run --data $D --split $D/splits/v2.json --out $R/spk-probe-m0 \
  --ckpt s0orig=orig,s1=$M/semcommit-q06-s1m3-d10/final,s2=$M/semcommit-q06-s4-ast/final,s3=$M/stage3-st3-p11-revive/checkpoint-3000,s4v1=$M/stage3-st4-m1/checkpoint-1500 2>&1 | grep -v -i "warn\|regex\|warprnnt\|^ *$"
$P experiments/spk_probe.py summarize --out $R/spk-probe-m0 --attr $R/stage3-attr-errors-st4-m1-ck1500 > $R/spk-probe-m0/summary.log 2>&1
echo "M0_DONE $(date '+%F %T')"
