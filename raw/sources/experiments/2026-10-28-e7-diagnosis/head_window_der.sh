# 화자 분리 헤드의 AMI 창 DER(mxc 호스트 GPU, 2026-10-28 — E8 이 E6 보다 cp 가 조금 나빠 헤드 자체가 나아졌는지 가린다). e1c_eval.sh 와 같은 절차:
#   AMI 창 ami-bench-v1 IHM-mix·SDM, 80 ms 프레임 DER(collar 0), 창 확률 저장 → 임계 훑기. diar.pt 의 shift(E8)는 diar_eval 이 맞춰 잰다(창 끝 R 프레임 제외).
#   GPUS="6 7" bash head_window_der.sh <이름> <diar 폴더(diar.pt)>   예: e6-head /soundai/Model/VAPASR/diar-q17-e6-head · e8-head /soundai/Model/VAPASR/diar-q17-e8-head
#   산출 VAPKT-data/results/diar-eval-ami-<이름> — 지우지 않는다(끝난 창은 건너뛰고 이어 한다).
set -u
NAME=${1:?이름}; DIAR=${2:?diar 폴더}
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
O=/soundai/users/tskim/VAPKT-data/results/diar-eval-ami-$NAME; mkdir -p $O
G=(${GPUS:-0}); w=${#G[@]}
echo "[$(date "+%F %T")] 헤드 창 DER $NAME ← $DIAR · GPU ${G[*]} → $O"
for r in $(seq 0 $((w - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$r]} $P experiments/diar_eval.py run --diar $DIAR --nemotron-dir $MXC_NEMOTRON_DIR --out $O --rank $r --world-size $w --save-probs >> $O/run-rank$r.log 2>&1 &
done
wait
for r in $(seq 0 $((w - 1))); do grep -q "^DONE rank=$r " $O/run-rank$r.log || { echo "몫 $r 실패 — $O/run-rank$r.log"; tail -5 $O/run-rank$r.log; exit 1; }; done
$P experiments/diar_eval.py summarize --out $O | tee $O/summary.txt
$P experiments/diar_eval.py sweep --out $O --thresholds 0.3,0.35,0.4,0.45,0.5 | tee $O/sweep.txt
echo "[$(date "+%F %T")] HEAD_DER_DONE $NAME"
