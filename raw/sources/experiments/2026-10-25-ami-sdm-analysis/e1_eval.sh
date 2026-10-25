# E1 평가(mxc 호스트, 2026-10-25): 화자 분리 가지(diar-q17-diar-e1 — 1.7B 회의 + 4 슬롯 인코더 동결)를 AMI 창 평가와 같은 창(ami-bench-v1 IHM-mix·SDM,
#   20–90 s draw 1)에서 80 ms 프레임 DER(collar 0)로 잰다 — experiments/diar_eval.py. 몫마다 GPU 하나, 다 끝나면 summarize 와 임계 훑기(sweep — dev 에서 고른다).
#   창 확률을 probs/<id>.npz 로 남긴다(--save-probs — E3 에서 E2 대상 화자 디코드에 주입할 예측 활동). 행이 이미 있는 창은 확률만 다시 만든다.
#   GPUS="4 6" bash e1_eval.sh [대상(기본 final, 또는 checkpoint-N)]. 산출 VAPKT-data/results/diar-eval-ami-q17-diar-e1-<대상> — 지우지 않는다(끝난 창은 건너뛰고 이어 한다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage                                            # .env 의 /scratch 는 컨테이너 전용
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; t=${1:-final}
O=/soundai/users/tskim/VAPKT-data/results/diar-eval-ami-q17-diar-e1-$t; mkdir -p $O
G=(${GPUS:-0}); w=${#G[@]}
echo "[$(date "+%F %T")] e1 평가 $t · GPU ${G[*]} → $O"
for r in $(seq 0 $((w - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$r]} $P experiments/diar_eval.py run --diar /soundai/Model/VAPASR/diar-q17-diar-e1/$t --nemotron-dir $MXC_NEMOTRON_DIR --out $O \
    --rank $r --world-size $w --save-probs >> $O/run-rank$r.log 2>&1 &
done
wait
for r in $(seq 0 $((w - 1))); do grep -q "^DONE rank=$r " $O/run-rank$r.log || { echo "몫 $r 실패 — $O/run-rank$r.log"; tail -5 $O/run-rank$r.log; exit 1; }; done
$P experiments/diar_eval.py summarize --out $O | tee $O/summary.txt
$P experiments/diar_eval.py sweep --out $O --thresholds 0.2,0.25,0.3,0.35,0.4,0.45,0.5 | tee $O/sweep.txt
echo "[$(date "+%F %T")] E1_EVAL_DONE"
