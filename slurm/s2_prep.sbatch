#!/bin/bash
# Stage 2 준비: asr-tn-v1.0.0 manifest 의 강제 정렬(align-asr-tn-v1/) + 항목 캐시. 노드 1 × GPU 8 (manifest 마다 8 shard). 특징은 학습 중 온라인(FEATURES=1 이면 캐시도 생성).
#   제출:  sbatch --export=ALL,MANIFESTS=librispeech-dev:librispeech-test:kspon-dev:kspon-eval:kspon-full:librispeech-960 slurm/s2_prep.sbatch
#          (manifest 구분자는 콜론 — --export 의 쉼표와 충돌. WPG=6: GPU 당 워커 수, BATCH=128 BATCH_SEC=960)
#   로그:   /soundai/users/tskim/VAPKT/logs/prep/<job>-<id>.out|.err, 워커별 logs/prep/<id>/align-<manifest>-<k>.log
#   재개 가능(파일 단위 skip) — 선점되면 requeue 로 이어서. 완료 표식: $ROOT/prep-DONE-<manifest 목록 hash>
#SBATCH --job-name=SA_SFT_FullduplexStage5
#SBATCH --partition=low_p_hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/prep/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/prep/%x-%j.err
set -uo pipefail
MANIFESTS=${MANIFESTS:?manifest 이름 목록 — sbatch --export 는 쉼표를 구분자로 먹으므로 콜론(:)으로 구분}; MANIFESTS=${MANIFESTS//:/,}; ENC=${ENC:-nemotron-c0}; FEATURES=${FEATURES:-0}; ROOT=/soundai/Model/VAPASR; LOGROOT=/soundai/users/tskim/VAPKT/logs; WLOG=$LOGROOT/prep/$SLURM_JOB_ID; mkdir -p "$WLOG"   # 워커(shard) 로그는 job 별 하위 폴더
DONE=$ROOT/prep-DONE-$(echo "$MANIFESTS" | md5sum | cut -c1-8)   # manifest 목록별 완료 표식(여러 job 으로 나눠 제출 가능)
echo "[$(date '+%F %T')] prep job $SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0} manifests=$MANIFESTS"
[ -f "$DONE" ] && { echo "이미 완료($DONE)"; exit 0; }
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
MXC_CONDA_DIR=${CONDA_OVERRIDE:-$MXC_CONDA_DIR}; source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
case "$(which python)" in "$MXC_CONDA_DIR"/envs/vapasr/*) ;; *) echo "!! conda env 아님"; exit 1;; esac
M=${MAN_IN:-$MXC_DATA_MANIFEST_DIR}; F=$MXC_DATA_FEATURE_CACHE_DIR; ALIGN_OUT=${OUT_ROOT:-$M/align-asr-tn-v1}   # MAN_IN: manifest 읽기 루트(Lustre 가능), OUT_ROOT: 정렬 출력 루트(/lustre 에 쓰지 않는다)
export VAPASR_ALIGN_ROOT=$ALIGN_OUT MXC_DATA_MANIFEST_DIR=$M; echo "manifest ← $M | 정렬 → $ALIGN_OUT | conda $MXC_CONDA_DIR"
PIDS=""; on_preempt() { echo "[$(date '+%T')] 선점 신호 → 워커 종료 후 requeue"; kill $PIDS 2>/dev/null; wait; scontrol requeue "$SLURM_JOB_ID"; exit 0; }; trap on_preempt USR1 TERM
n_rows() { python - "$1" "$2" <<'PY'
import json, sys; p, mode = sys.argv[1], sys.argv[2]; print(sum(1 for l in open(p) if mode == "all" or json.loads(l)["mode"] == mode))
PY
}
n_done_feat() { python - "$1" <<'PY'
import json, sys, os; p = sys.argv[1]; print(len({json.loads(l)["id"] for l in open(p)}) if os.path.exists(p) else 0)
PY
}
n_aligned() { python - "$1" <<'PY'
import json, sys, os; d = sys.argv[1]; ids = {f[:-6] for f in os.listdir(d) if f.endswith(".jsonl")} if os.path.isdir(d) else set(); pd = os.path.join(d, "parts")
for pf in (sorted(os.listdir(pd)) if os.path.isdir(pd) else []):                      # glob 대신 listdir(정렬기와 동일 경로)
    if not pf.endswith(".jsonl"): continue
    for l in open(os.path.join(pd, pf), encoding="utf-8"):
        try: ids.add(json.loads(l)["id"])
        except Exception: pass
print(len(ids), file=sys.stdout)
PY
}
WPG=${WPG:-6}                       # GPU 당 워커 수: 정렬기는 CPU(특징 추출) 병목이라 GPU 하나에 여러 워커(GPU 메모리 워커당 ≈5 GB)
run8() {  # run8 <script> <manifest> [extra…]: 8 GPU × WPG 워커 shard 병렬
  local s=$1 m=$2; shift 2; PIDS=""; local N=$((8 * WPG))
  for k in $(seq 0 $((N - 1))); do CUDA_VISIBLE_DEVICES=$((k % 8)) OMP_NUM_THREADS=2 python -u experiments/$s --manifest $m --mode all --shard $k/$N "$@" > $WLOG/${s%.py}-$m-$k.log 2>&1 & PIDS="$PIDS $!"; done
  wait $PIDS
}
for m in ${MANIFESTS//,/ }; do
  ROWS=$(n_rows $M/$m/streams.jsonl all); echo "[$(date '+%T')] == $m ($ROWS 행)"
  # 1) 이어받기(동결 전 캐시·정렬): 100 h 파일럿과 겹치는 id
  case $m in librispeech-960) python experiments/s2_seed_cache.py --new $m --old librispeech-100 --encoder $ENC --feat $FEATURES --new-align-root $ALIGN_OUT ;; kspon-full) python experiments/s2_seed_cache.py --new $m --old kspon-100 --encoder $ENC --feat $FEATURES --new-align-root $ALIGN_OUT ;;
              *) python experiments/s2_seed_cache.py --new $m --old $m --encoder $ENC --old-align align2 --feat $FEATURES --new-align-root $ALIGN_OUT ;; esac   # dev/test: 같은 이름, 옛 캐시·정렬은 다른 루트에 그대로
  # 2) 정렬
  A=$(n_aligned $ALIGN_OUT/$m); if [ "$A" -lt "$ROWS" ]; then echo "[$(date '+%T')] align $m: $A/$ROWS → $((8 * WPG)) shard"; run8 s1_align.py $m --batch ${BATCH:-128} --batch-sec ${BATCH_SEC:-960} --io-threads 8 --out-root $ALIGN_OUT; fi
  A=$(n_aligned $ALIGN_OUT/$m); echo "[$(date '+%T')] align $m: $A/$ROWS"
  # 3) 특징 캐시(선택, 기본 off: 학습이 온라인으로 인코딩)
  D=$ROWS
  if [ "$FEATURES" = 1 ]; then
    D=$(n_done_feat $F/$ENC/$m/index.jsonl); if [ "$D" -lt "$ROWS" ]; then echo "[$(date '+%T')] feat $m: $D/$ROWS → 8 shard"; run8 s1_extract_features.py $m --encoder $ENC; fi
    D=$(n_done_feat $F/$ENC/$m/index.jsonl); echo "[$(date '+%T')] feat $m: $D/$ROWS"
  fi
  [ "$A" -lt "$ROWS" ] || [ "$D" -lt "$ROWS" ] && { echo "!! $m 미완료 — requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
done
echo "[$(date '+%T')] 항목 캐시 prewarm"; python -u experiments/s2_prewarm_items.py ${MANIFESTS//,/ } && touch "$DONE" && echo "[$(date '+%T')] prep-DONE → $DONE"
