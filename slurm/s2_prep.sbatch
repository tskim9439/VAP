#!/bin/bash
# Stage 2 준비: asr-tn-v1.0.0 manifest 의 강제 정렬(align-asr-tn-v1/) + 항목 캐시. 노드 1 × GPU 8 (manifest 마다 8 shard). 특징은 학습 중 온라인(FEATURES=1 이면 캐시도 생성).
#   제출:  sbatch --export=ALL,MANIFESTS=librispeech-dev,librispeech-test,kspon-dev,kspon-eval,kspon-full,librispeech-960 slurm/s2_prep.sbatch
#   재개 가능(파일 단위 skip) — 선점되면 requeue 로 이어서. 완료 표식: $ROOT/prep-DONE
#SBATCH --job-name=SA_SFT_FullduplexStage5
#SBATCH --partition=low_p_hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/Model/VAPASR/logs/%x-%j-prep.out
set -uo pipefail
MANIFESTS=${MANIFESTS:?쉼표 구분 manifest 이름}; ENC=${ENC:-nemotron-c0}; FEATURES=${FEATURES:-0}; ROOT=/soundai/Model/VAPASR; mkdir -p $ROOT/logs
echo "[$(date '+%F %T')] prep job $SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} manifests=$MANIFESTS"
[ -f "$ROOT/prep-DONE" ] && { echo "이미 완료(prep-DONE)"; exit 0; }
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
case "$(which python)" in "$MXC_CONDA_DIR"/envs/vapasr/*) ;; *) echo "!! conda env 아님"; exit 1;; esac
M=$MXC_DATA_MANIFEST_DIR; F=$MXC_DATA_FEATURE_CACHE_DIR
PIDS=""; on_preempt() { echo "[$(date '+%T')] 선점 신호 → 워커 종료 후 requeue"; kill $PIDS 2>/dev/null; wait; scontrol requeue "$SLURM_JOB_ID"; exit 0; }; trap on_preempt USR1 TERM
n_rows() { python - "$1" "$2" <<'PY'
import json, sys; p, mode = sys.argv[1], sys.argv[2]; print(sum(1 for l in open(p) if mode == "all" or json.loads(l)["mode"] == mode))
PY
}
n_done_feat() { python - "$1" <<'PY'
import json, sys, os; p = sys.argv[1]; print(len({json.loads(l)["id"] for l in open(p)}) if os.path.exists(p) else 0)
PY
}
run8() {  # run8 <script> <manifest> [extra…]: 8 GPU shard 병렬
  local s=$1 m=$2; shift 2; PIDS=""
  for k in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$k python -u experiments/$s --manifest $m --mode all --shard $k/8 "$@" > $ROOT/logs/prep-$s-$m-$k.log 2>&1 & PIDS="$PIDS $!"; done
  wait $PIDS
}
for m in ${MANIFESTS//,/ }; do
  ROWS=$(n_rows $M/$m/streams.jsonl all); echo "[$(date '+%T')] == $m ($ROWS 행)"
  # 1) 이어받기(동결 전 캐시·정렬): 100 h 파일럿과 겹치는 id
  case $m in librispeech-960) python experiments/s2_seed_cache.py --new $m --old librispeech-100 --encoder $ENC --feat $FEATURES ;; kspon-full) python experiments/s2_seed_cache.py --new $m --old kspon-100 --encoder $ENC --feat $FEATURES ;;
              *) python experiments/s2_seed_cache.py --new $m --old $m --encoder $ENC --old-align align2 --feat $FEATURES ;; esac   # dev/test: 같은 이름, 옛 캐시·정렬은 다른 루트에 그대로
  # 2) 정렬
  A=$(ls $M/align-asr-tn-v1/$m 2>/dev/null | grep -c '\.jsonl$'); if [ "$A" -lt "$ROWS" ]; then echo "[$(date '+%T')] align $m: $A/$ROWS → 8 shard"; run8 s1_align.py $m; fi
  A=$(ls $M/align-asr-tn-v1/$m | grep -c '\.jsonl$'); echo "[$(date '+%T')] align $m: $A/$ROWS"
  # 3) 특징 캐시(선택, 기본 off: 학습이 온라인으로 인코딩)
  D=$ROWS
  if [ "$FEATURES" = 1 ]; then
    D=$(n_done_feat $F/$ENC/$m/index.jsonl); if [ "$D" -lt "$ROWS" ]; then echo "[$(date '+%T')] feat $m: $D/$ROWS → 8 shard"; run8 s1_extract_features.py $m --encoder $ENC; fi
    D=$(n_done_feat $F/$ENC/$m/index.jsonl); echo "[$(date '+%T')] feat $m: $D/$ROWS"
  fi
  [ "$A" -lt "$ROWS" ] || [ "$D" -lt "$ROWS" ] && { echo "!! $m 미완료 — requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
done
echo "[$(date '+%T')] 항목 캐시 prewarm"; python -u experiments/s2_prewarm_items.py ${MANIFESTS//,/ } && touch $ROOT/prep-DONE && echo "[$(date '+%T')] prep-DONE"
