#!/usr/bin/env bash
# E19 — 슬롯 활동 입력을 N3D 예측만으로(2026-11-12, 사용자 "이게 오히려 악영향을 주지 않았을까? N3D 정답만을 기준으로 학습시켜보고 싶어. 기존 모델을 추가학습 시켜보자.").
#   지금까지(E11–E18)는 표본마다 확률 ρ 로 N3D 예측 활동, 아니면 흔든 정답 활동(vapasr/data/act_noise — 화자 통째 누락·경계 ±0.24 s·겹친 구간 30 % 에서 한 슬롯 끄기·거짓 구간)을
#   LLM 슬롯 활동 입력으로 줬다. ρ 는 실행마다 0.3 → 0.9 로 오른다(--act-pred-ramp 기본) — E13 5,000 step 평균 0.6, 곧 표본의 40 % 는 흔든 정답이었다.
#   흔든 정답의 오류 모양은 E1 화자 분리 헤드(2026-10-26)를 본뜬 것이라 N3D 와 다르다(N3D 는 둘 겹친 단어의 97.1 / 93.9 % 에서 그 화자 슬롯을 켠다).
#   E19 는 E13 final 에서 E13 레시피 그대로(데이터·셋 가중·활동 게이트 12·헛 구간 0.1·슬롯 블록 번역·인코더 고정·N3D 학습 청크 16·학습률 일정) 이어 학습하되
#   **--act-pred-ramp 1.0,1.0 하나만 바꾼다** — 모든 표본이 N3D 예측 활동(정답 슬롯 자리로 맞춘 것 — align_slots)을 받는다. 추론 구조·디코드는 그대로다.
#     그대로 두는 것: 태그 게이트는 여전히 정답 활동으로 연다(데이터셋 쪽 — cst_dialogue gate_rounds), 헤드 손실도 그대로, N3D 채널 → 슬롯 자리 맞춤도 정답 기준.
#   견줄 값: E13 final 24.39 / 31.33(둘 겹침 놓침 41.8 / 47.1 %). 같은 꼴로 5,000 step 이어 간 E16(ρ 0.3 → 0.9 + 겹침 셋 22 %)은 24.92 / 31.70.
#     E19 가 E13 보다 좋아지면 이어 학습 자체의 효과와 가르기 위해 대조를 낸다: --ramp 0.3,0.9 --run q17-e19-ctl(나머지 같음).
#   평가: q17-e19* 묶음(submit-evals — q17-e16* 와 같은 묶음, 활동 N3D, 회의 통째 + 대화 번역)을 1 노드 job --nodes-eval 개(기본 2)로 학습 뒤에 건다.
#     한 번에 2 노드까지만(2026-11-11 사용자): 학습(2 노드) → [checkpoint-N 평가(2 노드)] → final 평가(2 노드)를 차례로 잇는다. --eval-ck 기본 0(final 만).
#     --evals-only JOBID 면 학습은 내지 않고 그 학습 job 뒤에 평가만 다시 건다(none = 학습이 이미 끝남).
#   학습 시간 어림: E13 4.5 h · E16 3.4 h(2 노드, 5,000 step).
# 쓰기(스케줄러 셸): [SLOTS=4 SUBSET=eval] bash slurm/submit-q17-e19.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--ramp R0,R1(기본 1.0,1.0)]
#                   [--n3d-train-chunk N(기본 16 — 추론은 4)] [--run q17-e19*(기본 q17-e19-n3d)] [--init DIR(기본 E13 final)] [--after JOBID[,…]]
#                   [--no-eval] [--nodes-eval N(기본 2)] [--eval-ck N(기본 0 = final 만)] [--evals-only JOBID|none]
#   SLOTS·SUBSET 은 평가(submit-evals.sh)에 그대로 간다. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; init=""; run=q17-e19-n3d; ramp=1.0,1.0; tc=16; ev=1; ne=2; ck=0; eo=""; aft=()
M=/soundai/Model/VAPASR
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --init) init=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;  --ramp) ramp=$2; shift 2 ;;  --n3d-train-chunk) tc=$2; shift 2 ;;  --no-eval) ev=0; shift ;;  --nodes-eval) ne=$2; shift 2 ;;
    --eval-ck) ck=$2; shift 2 ;;  --evals-only) eo=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; aft=(--after "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e19* ]] || { echo "--run 은 q17-e19 로 시작(평가 묶음 q17-e19* — N3D 활동)" >&2; exit 2; }
[[ $ramp =~ ^(0(\.[0-9]+)?|1(\.0+)?),(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "--ramp 는 ρ0,ρ1(각 0–1): $ramp" >&2; exit 2; }
[[ $tc =~ ^[1-9][0-9]?$ && $ne =~ ^[1-8]$ && $steps =~ ^[1-9][0-9]*$ ]] || { echo "--n3d-train-chunk·--nodes-eval·--max-steps: $tc $ne $steps" >&2; exit 2; }
[[ $ck =~ ^[0-9]+$ ]] && (( ck == 0 || (ck < steps && ck % 500 == 0) )) || { echo "--eval-ck 은 0 또는 max-steps 보다 작은 500 의 배수(저장 간격): $ck" >&2; exit 2; }
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e13/final}
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
subs() { grep -oE "Submitted batch job [0-9]+" | awk '{print $4}' | paste -sd: -; }                 # sbatch 출력 → "id:id"
if [[ -n $eo ]]; then                                                                              # 평가만 다시 건다(학습 job 은 이미 있다)
  [[ $eo == none || $eo =~ ^[0-9]+$ ]] || { echo "--evals-only 는 학습 job 번호 또는 none: $eo" >&2; exit 2; }
  dep=$([[ $eo == none ]] && echo "" || echo "$eo"); echo "E19($run) 평가만 — 학습 job ${dep:-끝남}"
else
  out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps --blocks-trans --no-eval ${aft[@]+"${aft[@]}"} \
        --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --freeze-encoder --n3d $n3d --n3d-train-chunk $tc --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --act-pred-ramp $ramp" \
        $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E19 학습 제출 실패" >&2; exit 1; }
  echo "$out" | grep -E "^\[|^1 |^sbatch" | sed 's/^/  /' >&2 || true
  dep=$(echo "$out" | grep -oE "^1 학습 job [0-9]+" | awk '{print $4}'); [[ -n $dep ]] || { echo "!! 학습 job 번호를 못 읽었다" >&2; exit 1; }
  echo "E19($run · 활동 입력 ρ $ramp · N3D 학습 청크 $tc · 인코더 고정 · ${steps} step) 학습 job $dep"
fi
(( ev )) || exit 0
prev=""                                                                                            # 앞 평가 묶음의 job 들 — 다음 묶음은 그 뒤에(한 번에 --nodes-eval 노드까지만)
if (( ck )); then
  o=$(TARGET=checkpoint-$ck bash slurm/submit-evals.sh "$run" ${dep:+--after-job "$dep"} --partition "$part" --nodes "$ne" $dry); echo "$o" | grep -E "Submitted|^sbatch" || true
  prev=$(echo "$o" | subs); echo "checkpoint-$ck 평가 job ${prev//:/ } — 학습 뒤"
fi
if [[ -n $prev ]]; then o=$(bash slurm/submit-evals.sh "$run" --after-any "$prev" --partition "$part" --nodes "$ne" $dry)
else o=$(bash slurm/submit-evals.sh "$run" ${dep:+--after-job "$dep"} --partition "$part" --nodes "$ne" $dry); fi
echo "$o" | grep -E "Submitted|^sbatch" || true
echo "final 평가 job $(echo "$o" | subs | tr ':' ' ') — $([[ -n $prev ]] && echo "checkpoint-$ck 평가 뒤" || echo "학습 뒤")"
