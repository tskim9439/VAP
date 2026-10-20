# VibeVoice-ASR(스트리밍·오프라인) 대 st4-m1 ck1500·st4-m2a·m2b — 같은 held-out 대화 창(Stage 3 회귀 평가 창, δ4·10)의 cp·agn(baseline_vibevoice score --compare).
# 1차: m2a·m2b ck1500·ck2000 의 Stage 3 회귀(평가 큐 4 단계)가 끝나면 → score-m2.json. 2차: final 이어 돌리기(stage4_eval_resume.sh)가 끝나면 final 까지 → score-m2-final.json.
# VibeVoice 원출력은 다시 돌리지 않는다(results/baseline-vibevoice-{stream,offline}). 기존 score.json 은 건드리지 않는다. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; T=/soundai/users/tskim/VAPKT-data/tmp
done4() { grep -q "4 Stage 3 회귀 끝" $T/stage4-evalq-$1.log; }
score() {   # $1 = 산출 이름, $2 = compare 목록
  for m in stream offline; do
    $P experiments/baseline_vibevoice.py score --out $R/baseline-vibevoice-$m --compare "$2" --dest $R/baseline-vibevoice-$m/$1.json > $R/baseline-vibevoice-$m/$1.log 2>&1
    echo "[$(date '+%F %T')] $m → $R/baseline-vibevoice-$m/$1.json (exit $?)"
  done
}
C="m1ck1500=$R/stage3-eval-st4-m1-ck1500"
echo "[$(date '+%F %T')] m2 ck1500·ck2000 Stage 3 회귀 대기"
until done4 st4-m2a-checkpoint-1500 && done4 st4-m2a-checkpoint-2000 && done4 st4-m2b-checkpoint-1500 && done4 st4-m2b-checkpoint-2000; do sleep 120; done
C="$C,m2a-ck1500=$R/stage3-eval-st4-m2a-ck1500,m2a-ck2000=$R/stage3-eval-st4-m2a-ck2000,m2b-ck1500=$R/stage3-eval-st4-m2b-ck1500,m2b-ck2000=$R/stage3-eval-st4-m2b-ck2000"
score score-m2 "$C"
echo "[$(date '+%F %T')] final 이어 돌리기 대기"
until grep -q "이어 돌리기 끝" $T/stage4-evalresume-st4-m2a-final.log && grep -q "이어 돌리기 끝" $T/stage4-evalresume-st4-m2b-final.log; do sleep 300; done
score score-m2-final "$C,m2a-final=$R/stage3-eval-st4-m2a-final,m2b-final=$R/stage3-eval-st4-m2b-final"
echo "[$(date '+%F %T')] VV_CMP_DONE"
