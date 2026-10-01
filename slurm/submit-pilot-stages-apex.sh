#!/usr/bin/env bash
# 0.6B·1.7B thinker 파일럿 단계 학습을 한 번에 제출한다(스케줄러 셸에서 `bash slurm/submit-pilot-stages-apex.sh [옵션]`).
#   Stage 0  adapter 워밍업(thinker 동결, final 모드만, ASR 전용) — 크기마다 1 노드, 두 크기가 동시에(노드 2 개)
#   Stage 1  스트리밍(δ 2–8) + final 30 %, 인코더 학습, SpecAugment·속도·잡음·잔향, SEM 중립 ASR 데이터 — 2 노드, 그 크기의 Stage 0 이
#            성공한 뒤에만 시작(afterok). 두 크기의 Stage 1 은 노드가 2 개뿐이면 차례로 돈다(스케줄러가 줄 세운다).
# 계획·근거: wiki/outputs/output-staged-training-plan-20261001.md. 산출물: /soundai/Model/VAPASR/semcommit-q{17,06}-s{0,1}
# 환경 변수를 만들지 않는다: 모든 값은 main 함수의 지역 변수이고 job 에는 submit-semcommit-train-apex.sh 의 sbatch --export 로만 넘어간다
# (호출한 셸에는 아무것도 남지 않는다 — `source` 로 부르지 말고 `bash` 로 실행).
#   --sizes 1.7B,0.6B     제출할 크기(기본 둘 다; 하나만이면 --sizes 1.7B)
#   --job-name NAME       기본 SA_EVAL_Fullduplex
#   --s0-steps N          Stage 0 step(기본 2000)       --s1-epochs E   Stage 1 epoch(기본 1)
#   --s1-nodes N          Stage 1 노드(기본 2)           --time HH:MM:SS 단계마다 시간 한도(기본 24:00:00)
#   --dry-run             sbatch 없이 명령만 출력
set -euo pipefail
cd "$(dirname "$0")/.."

main() {
  local sizes="1.7B,0.6B" job="SA_EVAL_Fullduplex" s0_steps=2000 s1_epochs=1 s1_nodes=2 time_limit="24:00:00" dry=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --sizes) sizes=$2; shift 2 ;;      --job-name) job=$2; shift 2 ;;     --s0-steps) s0_steps=$2; shift 2 ;;
      --s1-epochs) s1_epochs=$2; shift 2 ;;  --s1-nodes) s1_nodes=$2; shift 2 ;;  --time) time_limit=$2; shift 2 ;;
      --dry-run) dry="--dry-run"; shift ;;
      *) echo "알 수 없는 옵션: $1" >&2; return 2 ;;
    esac
  done
  local snap=/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
  local asr=/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list
  local nb=/soundai/users/tskim/VAPKT-data/data/noise-bank-v1
  local q tag qdir out id0 id1 summary=""
  for q in ${sizes//,/ }; do
    case "$q" in 1.7B) tag=q17 ;; 0.6B) tag=q06 ;; *) echo "--sizes 는 1.7B·0.6B 중에서: $q" >&2; return 2 ;; esac
    qdir=/soundai/Model/Qwen3-ASR-$q
    echo "=== $q: Stage 0 ($tag-s0, 1 노드, $s0_steps step)" >&2
    out=$(bash slurm/submit-semcommit-train-apex.sh --run "$tag-s0" --snap "$snap" --qwen-dir "$qdir" --job-name "$job" --nodes 1 --time "$time_limit" \
            --asr-only --offline-frac 1.0 --freeze-thinker --asr-list "$asr" --delays 2,3,4,6,8 --batch-tokens 16384 --max-steps "$s0_steps" --warmup 100 --varlen $dry)
    echo "$out"; id0=$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | awk '{print $4}')
    [[ -n $id0 ]] || { echo "Stage 0 제출 실패($q)" >&2; return 1; }
    echo "=== $q: Stage 1 ($tag-s1, $s1_nodes 노드, $s1_epochs epoch, afterok:$id0)" >&2
    out=$(bash slurm/submit-semcommit-train-apex.sh --run "$tag-s1" --snap "$snap" --init "/soundai/Model/VAPASR/semcommit-$tag-s0/final" --afterok "$id0" \
            --job-name "$job" --nodes "$s1_nodes" --time "$time_limit" --asr-only --offline-frac 0.3 --asr-list "$asr" --train-encoder \
            --spec-augment 2,27,2,0.03 --speed-perturb 0.9,1.0,1.1 --noise-bank "$nb" --varlen --delays 2,3,4,6,8 \
            --batch-tokens 16384 --lr 6e-5 --warmup 500 --epochs "$s1_epochs" $dry)
    echo "$out"; id1=$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | awk '{print $4}')
    [[ -n $id1 ]] || { echo "Stage 1 제출 실패($q) — Stage 0 job $id0 은 이미 제출됨" >&2; return 1; }
    summary+="$q: Stage 0 = job $id0 → Stage 1 = job $id1 (afterok)\n"
  done
  printf "\n제출 요약 (job 이름 %s)\n%b" "$job" "$summary"
}

main "$@"
