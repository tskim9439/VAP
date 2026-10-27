# E1c 평가(mxc 호스트, 2026-10-27): 인과 화자 분리 가지(diar-q17-diar-e1c — lookahead 0, 5 단계 ck3000 인코더 동결, 원거리·새 코퍼스 포함 10,000 step)를
#   E1 과 같은 절차로 잰다(raw/sources/experiments/2026-10-25-ami-sdm-analysis/e1_eval.sh 와 같은 명령 — AMI 창 ami-bench-v1 IHM-mix·SDM, 80 ms 프레임 DER, collar 0).
#   비교: E1(양방향, mt4 final 인코더) eval 최적 짝 DER IHM-mix 18.67 · SDM 23.03(임계 0.45). E6 은 이 가지에서 시작해 임계 0.45 를 쓴다 — 임계 훑기로 E1c 의 dev 최적도 본다.
#   GPUS="6 7" bash e1c_eval.sh [대상(기본 final)]. 산출 VAPKT-data/results/diar-eval-ami-q17-diar-e1c-<대상> — 지우지 않는다(끝난 창은 건너뛰고 이어 한다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; t=${1:-final}
O=/soundai/users/tskim/VAPKT-data/results/diar-eval-ami-q17-diar-e1c-$t; mkdir -p $O
G=(${GPUS:-0}); w=${#G[@]}
echo "[$(date "+%F %T")] e1c 평가 $t · GPU ${G[*]} → $O"
for r in $(seq 0 $((w - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$r]} $P experiments/diar_eval.py run --diar /soundai/Model/VAPASR/diar-q17-diar-e1c/$t --nemotron-dir $MXC_NEMOTRON_DIR --out $O \
    --rank $r --world-size $w --save-probs >> $O/run-rank$r.log 2>&1 &
done
wait
for r in $(seq 0 $((w - 1))); do grep -q "^DONE rank=$r " $O/run-rank$r.log || { echo "몫 $r 실패 — $O/run-rank$r.log"; tail -5 $O/run-rank$r.log; exit 1; }; done
$P experiments/diar_eval.py summarize --out $O | tee $O/summary.txt
$P experiments/diar_eval.py sweep --out $O --thresholds 0.2,0.25,0.3,0.35,0.4,0.45,0.5 | tee $O/sweep.txt
echo "[$(date "+%F %T")] E1C_EVAL_DONE"
