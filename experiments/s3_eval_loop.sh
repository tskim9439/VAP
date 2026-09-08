#!/usr/bin/env bash
# 컨테이너에서 run 디렉토리의 새 checkpoint-N 을 오프라인 sentinel 평가하는 루프 (2026-09-08).
#   scripts/sync-mxc.sh bg evalloop "experiments/s3_eval_loop.sh /soundai/Model/VAPASR/hf-C2 2000 2"
#   인자: <run dir> [step 간격=2000] [GPU=2]. 결과 <run>/eval/offline-N.json, hist.json, TensorBoard <run>/tb (offline_* 스칼라).
# 학습 중 평가가 다중 노드에서 멈추는 문제(66103·66201·66227)를 우회: 학습은 EVAL_EVERY 를 크게 두고 여기서 평가한다. DONE 이 생기고 final 까지 평가하면 종료.
set -uo pipefail
RUN=${1:?run dir}; EVERY=${2:-2000}; GPU=${3:-2}; SLEEP=${SLEEP:-300}
done_set=""
while true; do
  for d in $(ls -d "$RUN"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n) final; do
    [ "$d" = final ] && { [ -f "$RUN/final/config.json" ] || continue; n=final; ck="$RUN/final"; } || { n=$d; ck="$RUN/checkpoint-$d"; [ $((d % EVERY)) -eq 0 ] || continue; }
    [ -f "$RUN/eval/offline-$n.json" ] && continue; case " $done_set " in *" $n "*) continue;; esac
    [ -f "$ck/model.safetensors" ] || continue                                   # 저장 중인 디렉토리는 건너뛴다(다음 라운드에)
    echo "[$(date '+%F %T')] eval $ck"
    CUDA_VISIBLE_DEVICES=$GPU python experiments/s3_train_hf.py --eval-only --init "$ck" --out-dir "$RUN" --sentinel-stream 10 --sentinel-utt 100 --num-workers 0 2>&1 | grep -E "offline_score|Traceback|Error" | cut -c1-400
    done_set="$done_set $n"
  done
  [ -f "$RUN/DONE" ] && [ -f "$RUN/eval/offline-final.json" ] && { echo "[$(date '+%T')] DONE + final 평가 완료 → 종료"; exit 0; }
  sleep "$SLEEP"
done
