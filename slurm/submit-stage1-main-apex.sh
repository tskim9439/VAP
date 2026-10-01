#!/usr/bin/env bash
# Stage 1 본학습(1.7B) 두 run 을 한 번에 제출한다(스케줄러 셸에서 `bash slurm/submit-stage1-main-apex.sh [옵션]`). 레시피는 파일럿 q17-s1 과 같고
# (스트리밍 δ + final 30 %, ASR 전용, 인코더 학습, SpecAugment·속도·잡음·잔향), 다른 점은 다음과 같다:
#   - 출발점 q17-s0(Stage 0 adapter 워밍업) · 기본 3 epoch · checkpoint 1000 step 마다
#   - 인코더 학습 context 고정(--encoder-context-sampling fixed): NeMo 기본(multi-lookahead, step 마다 [56,3]·[56,0]·[56,6]·[56,13] 무작위)을 끄고
#     평가·추론과 같은 context 하나로만 학습한다(2026-10-01 발견, wiki/outputs/output-stage1-pilot-eval-20261001.md)
#   r0  q17-s1m-r0  [56,0](80 ms)                       r3  q17-s1m-r3  [56,3](4 프레임 청크, 최대 320 ms)
#   두 run 모두 δ 1,2,3,4,6,8 로 학습한다 — 같은 총지연끼리 비교([56,0]+δd = 80·d ms, [56,3]+δd = 평균 80·d+120 · 최대 80·d+240 ms).
# 산출물: /soundai/Model/VAPASR/semcommit-q17-s1m-r{0,3}. 시간 한도를 넘기면 같은 명령(--runs 로 그 run 만)을 다시 제출하면 checkpoint 에서 이어서 한다.
# 환경 변수를 만들지 않는다: 모든 값은 main 의 지역 변수이고 job 에는 submit-semcommit-train-apex.sh 의 sbatch --export 로만 넘어간다(`bash` 로 실행).
#   --runs LIST      r0·r3 중(기본 r0,r3)         --nodes N(run 마다, 기본 2)   --epochs E(기본 3)   --lr LR(기본 6e-5)   --warmup N(기본 500)
#   --sampling S     fixed(기본) | multi            --job-name NAME(기본 SA_EVAL_Fullduplex)   --time HH:MM:SS(기본 24:00:00)   --dry-run
set -euo pipefail
cd "$(dirname "$0")/.."

main() {
  local runs="r0,r3" nodes=2 epochs=3 lr=6e-5 warmup=500 samp=fixed job="SA_EVAL_Fullduplex" time_limit="24:00:00" dry=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --runs) runs=$2; shift 2 ;;     --nodes) nodes=$2; shift 2 ;;   --epochs) epochs=$2; shift 2 ;;   --lr) lr=$2; shift 2 ;;
      --warmup) warmup=$2; shift 2 ;; --sampling) samp=$2; shift 2 ;; --job-name) job=$2; shift 2 ;;    --time) time_limit=$2; shift 2 ;;
      --dry-run) dry="--dry-run"; shift ;;
      *) echo "알 수 없는 옵션: $1" >&2; return 2 ;;
    esac
  done
  [[ $samp == fixed || $samp == multi ]] || { echo "--sampling 은 fixed|multi" >&2; return 2; }
  local snap=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
  local asr=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
  local nb=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
  local init=/soundai/Model/VAPASR/semcommit-q17-s0/final
  [[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; return 1; }
  local r rc out id summary=""
  for r in ${runs//,/ }; do
    case "$r" in r0) rc=0 ;; r3) rc=3 ;; *) echo "--runs 는 r0·r3 중에서: $r" >&2; return 2 ;; esac
    echo "=== q17-s1m-$r: 인코더 [56,$rc] ($samp), $nodes 노드, $epochs epoch, lr $lr, init $init" >&2
    out=$(bash slurm/submit-semcommit-train-apex.sh --run "q17-s1m-$r" --snap "$snap" --init "$init" --job-name "$job" --nodes "$nodes" --time "$time_limit" \
            --asr-only --offline-frac 0.3 --asr-list "$asr" --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$nb" --varlen \
            --delays 1,2,3,4,6,8 --batch-tokens 16384 --lr "$lr" --warmup "$warmup" --epochs "$epochs" --save-every 1000 \
            --extra "--encoder-right-context $rc --encoder-context-sampling $samp" $dry)
    echo "$out"; id=$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | awk '{print $4}')
    [[ -n $id ]] || { echo "제출 실패(q17-s1m-$r)" >&2; return 1; }
    summary+="q17-s1m-$r ([56,$rc], $samp): job $id\n"
  done
  printf "\n제출 요약 (job 이름 %s)\n%b" "$job" "$summary"
}

main "$@"
