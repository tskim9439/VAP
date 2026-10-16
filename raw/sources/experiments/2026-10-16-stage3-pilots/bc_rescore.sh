# 맞장구 규칙 참조로 p1b·p3 재채점(디코드 없음, CPU): 창·전사 참조는 같고 EOT 참조만 맞장구 규칙(1.0 s·3 토큰). 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
D=/soundai/users/tskim/VAPKT-data/data/phase2; R=/soundai/users/tskim/VAPKT-data/results
export PYTHONPATH=/soundai/users/tskim/VAPKT HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
for run in st3-p1b-base st3-p3-bc; do
  $P experiments/cst_stage3_eval.py run --from-predictions $R/stage3-eval-$run-20261016 --checkpoint /soundai/Model/VAPASR/stage3-$run/final --data $D --split $D/splits/v2.json \
    --out $R/stage3-eval-$run-20261016-bcref --deltas 4,10 --limit-windows 200 --batch-size 64 --backchannel 1.0,3 > /dev/null 2>&1 &
done; wait
for run in st3-p1b-base st3-p3-bc; do $P experiments/cst_stage3_eval.py summarize --out $R/stage3-eval-$run-20261016-bcref > /dev/null 2>&1; done
echo EXIT
