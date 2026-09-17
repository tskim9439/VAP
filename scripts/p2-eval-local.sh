#!/usr/bin/env bash
# Phase 2 lane 평가를 mxc 로그인 노드의 빈 GPU 1 장에서 실행 — 빈 GPU(사용 < 1 GB)가 생길 때까지 기다린다(최대 WAIT_MIN 분, 기본 240).
#   nohup env SETS=aihub71631:seen:8,ami:seen:6 OUT=/soundai/users/tskim/VAPKT-data/eval/p2-D1-seen bash scripts/p2-eval-local.sh > logs/train/p2eval-seen.log 2>&1 &
#   옵션: MODEL(기본 p2-D1/final), DATA, EXTRA(추가 인자)
set -uo pipefail
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
SETS=${SETS:?SETS=corpus:tag:n,...}; OUT=${OUT:?OUT}; MODEL=${MODEL:-/soundai/Model/VAPASR/p2-D1/final}; DATA=${DATA:-/soundai/users/tskim/VAPKT-data/data/phase2}
for i in $(seq 1 $(( ${WAIT_MIN:-240} / 2 ))); do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F", " -v lim="${MAX_USED_MIB:-30000}" '$2<lim{print $2, $1}' | sort -n | head -1 | awk '{print $2}')   # 사용량이 가장 적은 GPU(상한 MAX_USED_MIB, 기본 30 GB — 평가는 ~10 GB 면 충분)
  [ -n "$FREE" ] && break; [ $((i % 5)) -eq 1 ] && echo "[$(date '+%F %T')] 빈 GPU 없음 — 대기 ($i)"; sleep 120
done
[ -n "${FREE:-}" ] || { echo "빈 GPU 를 얻지 못함"; exit 1; }
echo "[$(date '+%F %T')] GPU $FREE 에서 평가 시작: SETS=$SETS OUT=$OUT"
CUDA_VISIBLE_DEVICES=$FREE python experiments/p2_eval_lanes.py --model "$MODEL" --data "$DATA" --mono-cache "$DATA/mono" --sets "$SETS" --out "$OUT" ${EXTRA:-}
echo "[$(date '+%F %T')] 평가 종료 rc=$?"
