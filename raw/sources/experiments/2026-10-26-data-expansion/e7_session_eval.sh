# E7 회의 단위 화자 분리 평가(mxc 호스트, 2026-10-27): AMI 회의 통째(ami-bench-v1 IHM-mix·SDM, 4 명 이하 회의)를 한 행씩, 회의 전체 도착 순서 참조로
#   80 ms 프레임 DER(회의 전체 한 순열의 최적 짝 — 조각마다 바뀐 슬롯은 혼동으로 센다)을 잰다 — experiments/diar_eval.py run --session.
#   GPUS="6 7" bash e7_session_eval.sh <이름> <diar 폴더(diar.pt)> <방식들(쉼표 — aosc · resetS, 예 aosc,reset90)>
#     예: e6  /soundai/Model/VAPASR/diar-q17-e6-head aosc,reset90(E6 헤드 그대로 AOSC — AOSC 학습 없음 · 90 s 조각마다 새로 — 기준선)
#         e7a /soundai/Model/VAPASR/diar-q17-diar-e7a/final aosc(AOSC 로 미세조정한 헤드)
#   산출 VAPKT-data/results/diar-session-<이름>/<방식> — 지우지 않는다(끝난 회의는 건너뛰고 이어 한다). 인코딩은 회의마다 한 번(방식끼리 같이 쓴다).
set -u
NAME=$1; DIAR=$2; MODE=$3
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
O=/soundai/users/tskim/VAPKT-data/results/diar-session-$NAME; mkdir -p $O
G=(${GPUS:-0}); w=${#G[@]}
echo "[$(date "+%F %T")] e7 회의 평가 $NAME ← $DIAR · $MODE · GPU ${G[*]} → $O"
for r in $(seq 0 $((w - 1))); do
  CUDA_VISIBLE_DEVICES=${G[$r]} $P experiments/diar_eval.py run --diar $DIAR --nemotron-dir $MXC_NEMOTRON_DIR --out $O --threshold 0.45 --session --session-modes $MODE --rank $r --world-size $w >> $O/run-rank$r.log 2>&1 &
done
wait
for r in $(seq 0 $((w - 1))); do grep -q "^DONE rank=$r " $O/run-rank$r.log || { echo "몫 $r 실패 — $O/run-rank$r.log"; tail -5 $O/run-rank$r.log; exit 1; }; done
for m in ${MODE//,/ }; do echo "== $m"; $P experiments/diar_eval.py summarize --out $O/$m | tee $O/$m/summary.txt; done
echo "[$(date "+%F %T")] E7_SESSION_EVAL_DONE $NAME"
