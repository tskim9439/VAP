#!/bin/bash
# Stage 1 mono 스트리밍 ASR — run A(LoRA) / run B(thinker full FT), 노드 1 × GPU 8 (H200 140 GB), 선점 시 자동 재개.
#   제출:  sbatch --export=ALL,RUN=A slurm/s1_mono.sbatch
#          sbatch --export=ALL,RUN=B slurm/s1_mono.sbatch
#   산출물: /soundai/Model/VAPASR/s1-<RUN>/  (ckpt-last.pt 재개용 · ckpt-<step>.pt · results.json · eval/ · DONE)
#   로그:   /soundai/Model/VAPASR/logs/SA_SFT_FullduplexStage5-<jobid>.out
#SBATCH --job-name=SA_SFT_FullduplexStage5
#SBATCH --partition=low_p_hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=/soundai/Model/VAPASR/logs/%x-%j.out
set -uo pipefail
RUN=${RUN:?RUN=A|B 를 --export 로 주세요}
ROOT=/soundai/Model/VAPASR; OUT=$ROOT/s1-$RUN; mkdir -p "$OUT" "$ROOT/logs"
echo "[$(date '+%F %T')] job $SLURM_JOB_ID RUN=$RUN node=$(hostname) restart=${SLURM_RESTART_COUNT:-0}"
[ -f "$OUT/DONE" ] && { echo "이미 완료(DONE) — 종료"; exit 0; }

cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
source "$MXC_CONDA_DIR/etc/profile.d/conda.sh"; conda activate vapasr
export MXC_CKPT_EXP_DIR=$ROOT          # 학습 산출물·증류 adapter 를 모두 /soundai/Model/VAPASR 아래로
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false
case "$(which python)" in "$MXC_CONDA_DIR"/envs/vapasr/*) ;; *) echo "!! conda env 아님: $(which python)"; exit 1;; esac

# ── 1. adapter 증류 초기화(공유, 없으면 1 GPU 로 생성; A/B 동시 실행 시 mkdir 잠금으로 한쪽만 만들고 다른 쪽은 기다린다)
ADAPTER=$ROOT/uslm/s1-adapter-distill/adapter.pt
if [ ! -f "$ADAPTER" ]; then
  # 증류 타깃(AuT block8s) 캐시는 추출이 끝나야 stats.json 이 생긴다(index.jsonl 은 진행 중에도 존재). 없으면 최대 8 h 대기.
  for m in librispeech-100 kspon-100; do
    n=0; until [ -f "$MXC_DATA_FEATURE_CACHE_DIR/qwen-aut-block8s/$m/stats.json" ]; do
      [ $n -eq 0 ] && echo "[$(date '+%T')] 증류 타깃 캐시 미완성: qwen-aut-block8s/$m — 추출 완료(stats.json) 대기"
      n=$((n+1)); [ $n -gt 96 ] && { echo "!! 8 h 대기 후에도 없음 — experiments/s1_extract_features.py --encoder qwen-aut-block8s --manifest $m 확인"; exit 1; }; sleep 300
    done
  done
  if mkdir "$ROOT/uslm/.distill.lock" 2>/dev/null; then
    echo "[$(date '+%T')] adapter 증류 시작"; CUDA_VISIBLE_DEVICES=0 python -u experiments/s1_distill_adapter.py --epochs 2 --gpu 0 || { rmdir "$ROOT/uslm/.distill.lock"; exit 1; }
    rmdir "$ROOT/uslm/.distill.lock"
  else
    echo "다른 job 이 증류 중 — 대기"; until [ -f "$ADAPTER" ]; do sleep 60; done
  fi
fi
echo "adapter: $ADAPTER"

# ── 2. run 별 설정 (유효 배치 = 8 GPU × 언어별 bs: EN 96 스트림 / KO 384 발화 → lr 은 단일 GPU 대비 약 2 배)
COMMON="--tag $RUN --out-dir $OUT --epochs 15 --bs-en 12 --bs-ko 48 --init-adapter $ADAPTER --next-weight 0.3 --next-weight-ko 0.15
        --eval-every 1000 --eval-bias 0 --sentinel-stream 10 --sentinel-utt 100 --save-every 300 --warmup 200 --resume auto"
case "$RUN" in
  A) EXTRA="--lora-r 16 --lr 4e-4 --lr-adapter 1e-3" ;;
  B) EXTRA="--full-ft --lr 4e-5 --lr-adapter 1e-3" ;;
  *) echo "RUN 은 A 또는 B"; exit 1 ;;
esac

# ── 3. 선점 신호(USR1, 종료 180 s 전): PREEMPT 파일 → 학습기가 다음 step 경계에서 ckpt-last 저장 후 종료 → requeue
rm -f "$OUT/PREEMPT"
on_preempt() { echo "[$(date '+%T')] USR1 수신 → PREEMPT 표시, 저장 대기"; touch "$OUT/PREEMPT"; wait "$PID"; echo "requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_preempt USR1

echo "[$(date '+%T')] torchrun 시작: $COMMON $EXTRA"
torchrun --nproc_per_node=8 --master_port=$((20000 + SLURM_JOB_ID % 10000)) experiments/s1_train_mono.py $COMMON $EXTRA &
PID=$!; wait "$PID"; RC=$?
if [ "$RC" -eq 0 ] && grep -q '"steps_per_epoch"' "$OUT/results.json" 2>/dev/null && python - "$OUT" <<'PY'
import json, sys, os, torch
o = sys.argv[1]; ck = torch.load(os.path.join(o, "ckpt-last.pt"), map_location="cpu"); sys.exit(0 if ck["step"] >= ck["args"]["steps"] else 1)
PY
then touch "$OUT/DONE"; echo "[$(date '+%T')] 학습 완료 → DONE"
elif [ "$RC" -eq 0 ]; then echo "[$(date '+%T')] 미완료 종료(선점 저장 등) → requeue"; scontrol requeue "$SLURM_JOB_ID"
else echo "[$(date '+%T')] 학습 실패 rc=$RC"; exit "$RC"; fi
