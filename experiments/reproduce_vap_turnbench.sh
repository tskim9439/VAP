#!/usr/bin/env bash
# VAP baseline 을 TurnBench dev 셋에서 재현한다 (task-reproduce-vap-turnbench-baseline).
#   1) dev 데이터셋 확보 (HF, gated → HF_TOKEN 필요)
#   2) 저장소 동봉 baselines/vap/predictions-dev.json 재채점  → 리더보드 수치와 대조
#   3) --pretrained 체크포인트로 직접 예측 + sweep + 채점
#   4) oto fine-tune 체크포인트(viks66/VAP_checkpoints)로 직접 예측 + sweep + 채점
# 컨테이너에서: source scripts/activate-env.sh && CUDA_VISIBLE_DEVICES=3 bash experiments/reproduce_vap_turnbench.sh
set -euo pipefail
TB=/data3/tskim/third_party/turnbench
OUT="${DATA_LOG_DIR:-/tmp}/vap-repro-$(date +%Y%m%d-%H%M)"; mkdir -p "$OUT"
log(){ printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }
cd "$TB"

log "1) dev 데이터셋 (HF 캐시 $HF_HOME)"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("mundo-ai/turn-benchmark-dev", repo_type="dataset", token=os.environ["HF_TOKEN"])
print("dev at", p)
PY

log "2) 동봉 predictions-dev.json 재채점 (oto 체크포인트, θ_eot 0.9161 / θ_int 0.8591)"
python -m turnbench.score baselines/vap/predictions-dev.json 2>&1 | tee "$OUT/score-shipped-dev.txt"

log "3) --pretrained 직접 예측 (사전학습 원본 VAP_3mmz3t0u)"
bash baselines/vap/run.sh --dev --pretrained 2>&1 | tee "$OUT/run-pretrained.txt" | grep -vE "it/s\]|^\s*$" | tail -40
cp -v baselines/vap/pretrained-predictions-dev.json "$OUT/" 2>/dev/null || true
python -m turnbench.score baselines/vap/pretrained-predictions-dev.json 2>&1 | tee "$OUT/score-pretrained-dev.txt"

log "4) oto 체크포인트 직접 예측 (리더보드 설정)"
bash baselines/vap/run.sh --dev 2>&1 | tee "$OUT/run-oto.txt" | grep -vE "it/s\]|^\s*$" | tail -40
cp -v baselines/vap/predictions-dev.json "$OUT/oto-predictions-dev.json" 2>/dev/null || true
python -m turnbench.score baselines/vap/predictions-dev.json 2>&1 | tee "$OUT/score-oto-dev.txt"

log "완료 → $OUT"; ls -la "$OUT"
