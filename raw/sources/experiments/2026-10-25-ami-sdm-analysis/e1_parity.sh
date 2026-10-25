# E1 저장·재로드 대조(mxc 호스트, 2026-10-25 — 메모리 규칙 checkpoint-save-load-parity, 본 학습 95482 는 끝에 대조하지 않았다):
#   (1) 끝난 run diar-q17-diar-e1 의 final 을 새로 읽어(인코더는 init 에서) 학습 설정(config.json) 분포 배치 N 개의 손실·frame_acc 를 잰다 — 학습 로그 끝 값과 비교.
#   (2) 고친 diar_train.py 의 학습 끝 대조 경로(가지·인코더 가중치 차이 0, 같은 배치 손실 일치)를 작은 스모크(AMI, 가지 2 층, 10 step)로 확인한다.
#   GPU=4 bash e1_parity.sh [N(기본 16)]. 산출 diar-q17-diar-e1/parity.json, /tmp/sa_tskim/e1-parity-smoke — 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=${GPU:-0}
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; INIT=/soundai/Model/VAPASR/stage3-q17-v3-p1-tel-xlw-mt4/final; O=/soundai/Model/VAPASR/diar-q17-diar-e1
echo "[$(date "+%F %T")] e1 check-final (GPU $CUDA_VISIBLE_DEVICES)"
$P experiments/diar_train.py --init $INIT --nemotron-dir $MXC_NEMOTRON_DIR --out-dir $O --check-final ${1:-16} 2>&1 | grep -v "regex pattern"
echo "CHECK EXIT ${PIPESTATUS[0]}"
echo "[$(date "+%F %T")] e1 parity smoke"
$P experiments/diar_train.py --init $INIT --nemotron-dir $MXC_NEMOTRON_DIR --out-dir /tmp/sa_tskim/e1-parity-smoke --corpora ami --smoke 10 --batch 2 --layers 2 \
  --log-every 2 --num-workers 0 2>&1 | grep -v "regex pattern"
echo "SMOKE EXIT ${PIPESTATUS[0]}"
echo "[$(date "+%F %T")] E1_PARITY_DONE"
