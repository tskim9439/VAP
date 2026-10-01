#!/usr/bin/env bash
# Stage 1 후속 두 실험을 한 번에 제출한다(스케줄러 셸에서 `bash slurm/submit-lookahead-ablation-apex.sh [옵션]`). 둘 다 q17-s1(1.7B Stage 1 파일럿)에서 출발해
# 같은 레시피·같은 step 으로 1 epoch 더 학습하고, 인코더 att_context 만 다르다.
# 주의(2026-10-01 확인): NeMo 인코더는 train 모드에서 step 마다 att_context 를 [56,3]·[56,0]·[56,6]·[56,13] 중 무작위로 고른다(multi-lookahead).
#   --train-encoder 로 학습한 run(E2·q17-s1 포함)은 모두 그렇게 학습됐고, --encoder-right-context 는 평가 context 만 바꾼다.
#   그래서 r0·r3(multi)는 학습이 같고(job 79150·79151) 평가 context 만 다르다 — 학습 context 를 고정한 r0f·r3f 가 진짜 비교다.
#   r0   q17-s1b-r0   multi 학습, 평가 [56,0]  → q17-s1 대비 "1 epoch 더" 효과(본 학습 epoch 수 근거; q17-s1 도 multi)
#   r3   q17-s1b-r3   multi 학습, 평가 [56,3]  → r0 과 학습이 같으므로 제출하지 않는다(r0 체크포인트를 rc3 로 평가하면 같다)
#   r0f  q17-s1b-r0f  [56,0] 고정 학습         → multi 대 고정(r0) · lookahead 비교의 기준
#   r3f  q17-s1b-r3f  [56,3] 고정 학습(4 프레임 청크, 최대 320 ms) → r0f 와 같은 총지연끼리 비교(인코더 lookahead 대 디코더 δ)
#      총지연(단어 끝 프레임 기준, 계산 시간 제외): [56,0]+δd = 80·d ms, [56,3]+δd = 평균 80·d+120 ms(최대 80·d+240)
#      → 비교 짝: A δ3/δ4(240/320) ↔ B δ1(평균 200, 최대 320), A δ6(480) ↔ B δ3(평균 360, 최대 480)
# 두 run 모두 δ 1,2,3,4,6,8 로 학습한다(B 의 δ1 비교를 위해 1 을 넣었다; A 도 같게 맞춘다). checkpoint 는 1000 step 마다(1.7B 하나 27 GB).
# 계획·근거: wiki/outputs/output-stage1-pilot-eval-20261001.md. 산출물: /soundai/Model/VAPASR/semcommit-q17-s1b-r{0,3}
# 환경 변수를 만들지 않는다: 모든 값은 main 의 지역 변수이고 job 에는 submit-semcommit-train-apex.sh 의 sbatch --export 로만 넘어간다(`bash` 로 실행).
#   --nodes N        run 마다 노드 수(기본 1 — 두 run 이 노드 2 개에서 동시에 돈다; 2 면 스케줄러가 차례로 돌린다)
#   --runs LIST      제출할 run(기본 r0f,r3f; r0·r3·r0f·r3f)       --epochs E(기본 1)   --lr LR(기본 3e-5)   --warmup N(기본 300)
#   --job-name NAME  기본 SA_EVAL_Fullduplex      --time HH:MM:SS(기본 24:00:00)   --dry-run  sbatch 없이 명령만 출력
set -euo pipefail
cd "$(dirname "$0")/.."

main() {
  local nodes=1 runs="r0f,r3f" epochs=1 lr=3e-5 warmup=300 job="SA_EVAL_Fullduplex" time_limit="24:00:00" dry=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --nodes) nodes=$2; shift 2 ;;   --runs) runs=$2; shift 2 ;;   --epochs) epochs=$2; shift 2 ;;   --lr) lr=$2; shift 2 ;;
      --warmup) warmup=$2; shift 2 ;; --job-name) job=$2; shift 2 ;; --time) time_limit=$2; shift 2 ;;   --dry-run) dry="--dry-run"; shift ;;
      *) echo "알 수 없는 옵션: $1" >&2; return 2 ;;
    esac
  done
  local snap=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
  local asr=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
  local nb=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
  local init=/soundai/Model/VAPASR/semcommit-q17-s1/final
  [[ -f $init/config.json ]] || { echo "init 없음: $init" >&2; return 1; }
  local r rc samp out id summary=""
  for r in ${runs//,/ }; do
    case "$r" in r0) rc=0; samp=multi ;; r3) rc=3; samp=multi ;; r0f) rc=0; samp=fixed ;; r3f) rc=3; samp=fixed ;; *) echo "--runs 는 r0·r3·r0f·r3f 중에서: $r" >&2; return 2 ;; esac
    echo "=== q17-s1b-$r: 인코더 평가 [56,$rc] · 학습 context $samp, $nodes 노드, $epochs epoch, lr $lr" >&2
    out=$(bash slurm/submit-semcommit-train-apex.sh --run "q17-s1b-$r" --snap "$snap" --init "$init" --job-name "$job" --nodes "$nodes" --time "$time_limit" \
            --asr-only --offline-frac 0.3 --asr-list "$asr" --train-encoder --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$nb" --varlen \
            --delays 1,2,3,4,6,8 --batch-tokens 16384 --lr "$lr" --warmup "$warmup" --epochs "$epochs" --save-every 1000 \
            --extra "--encoder-right-context $rc --encoder-context-sampling $samp" $dry)
    echo "$out"; id=$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | awk '{print $4}')
    [[ -n $id ]] || { echo "제출 실패(q17-s1b-$r)" >&2; return 1; }
    summary+="q17-s1b-$r ([56,$rc], $samp): job $id\n"
  done
  printf "\n제출 요약 (job 이름 %s)\n%b" "$job" "$summary"
}

main "$@"
