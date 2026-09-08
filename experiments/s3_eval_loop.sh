#!/usr/bin/env bash
# 컨테이너에서 run 디렉토리의 새 checkpoint-N 을 오프라인 sentinel 평가하는 루프 (2026-09-08).
#   scripts/sync-mxc.sh bg evalloop "experiments/s3_eval_loop.sh /soundai/Model/VAPASR/hf-C2 2000 2"
#   인자: <run dir> [step 간격=2000] [GPU=2]. 결과 <run>/eval/offline-N.json, hist.json, TensorBoard <run>/tb (offline_* 스칼라).
# Trainer 의 save_total_limit 로 checkpoint-N 이 곧 지워지므로(500 step 간격이면 약 16 분 수명) 먼저 <run>/eval-ckpts/checkpoint-N 에 모델만 스냅샷(2.4 GB)해 두고
# 그 스냅샷을 평가한다. 스냅샷은 지우지 않는다(정리는 사람이). DONE 이 생기고 final 까지 평가하면 종료.
set -uo pipefail
RUN=${1:?run dir}; EVERY=${2:-2000}; GPU=${3:-2}; SLEEP=${SLEEP:-60}; SNAP=$RUN/eval-ckpts; mkdir -p "$SNAP"
snapshot() {                                                                       # 새 checkpoint-N(N % EVERY == 0) → 스냅샷
  for d in $(ls -d "$RUN"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n); do
    [ $((d % EVERY)) -eq 0 ] || continue; [ -f "$RUN/checkpoint-$d/model.safetensors" ] || continue; [ -f "$SNAP/checkpoint-$d/model.safetensors" ] && continue
    echo "[$(date '+%F %T')] snapshot checkpoint-$d"; mkdir -p "$SNAP/checkpoint-$d.tmp"
    cp "$RUN/checkpoint-$d/config.json" "$SNAP/checkpoint-$d.tmp/" && cp "$RUN/checkpoint-$d/model.safetensors" "$SNAP/checkpoint-$d.tmp/" || { echo "  ! 복사 실패(저장 중이거나 지워짐)"; continue; }
    for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json special_tokens_map.json; do [ -f "$RUN/checkpoint-$d/$f" ] && cp "$RUN/checkpoint-$d/$f" "$SNAP/checkpoint-$d.tmp/"; done
    mv "$SNAP/checkpoint-$d.tmp" "$SNAP/checkpoint-$d"
  done
  [ -f "$RUN/final/config.json" ] && [ ! -f "$SNAP/final/model.safetensors" ] && { mkdir -p "$SNAP/final.tmp" && cp "$RUN"/final/* "$SNAP/final.tmp/" && mv "$SNAP/final.tmp" "$SNAP/final"; }
  return 0
}
while true; do
  snapshot
  for ck in $(ls -d "$SNAP"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n) final; do
    if [ "$ck" = final ]; then [ -f "$SNAP/final/model.safetensors" ] || continue; n=final; dir="$SNAP/final"; else n=$ck; dir="$SNAP/checkpoint-$ck"; fi
    [ -f "$RUN/eval/offline-$n.json" ] && continue
    echo "[$(date '+%F %T')] eval $dir"
    CUDA_VISIBLE_DEVICES=$GPU python experiments/s3_train_hf.py --eval-only --init "$dir" --out-dir "$RUN" --sentinel-stream 10 --sentinel-utt 100 --num-workers 0 2>&1 | grep -E "offline_score|Traceback|Error" | cut -c1-400
    snapshot                                                                       # 평가(≈17 분) 중 생긴 checkpoint 를 놓치지 않도록
  done
  [ -f "$RUN/DONE" ] && [ -f "$RUN/eval/offline-final.json" ] && { echo "[$(date '+%T')] DONE + final 평가 완료 → 종료"; exit 0; }
  sleep "$SLEEP"
done
