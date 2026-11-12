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
#   겹침 데이터(2026-11-12 사용자 "파티션은 low_p_hpc, 데이터는 겹침 비중을 늘린 판으로 학습할 수 있도록"): --ovl 75 = E17·E18 의 셋(sim_en-ovl3 24 + sim_en-ovl2 18 + sim_en-ovl 6 → 섞기 비중 약 75 %),
#     --ovl 22 = E16 의 셋(sim_en-ovl2 3.5 + sim_en-ovl 1.0 → 약 22 %), --ovl 0(기본) = E13 그대로. --w-ovl3·--w-ovl2·--w-ovl 로 가중을 따로 줄 수도 있다.
#     --ovl 75 면 시작점 기본이 E16 final, 실행 이름 기본이 q17-e19-n3d-ovl 이다 — E17(E16 final → 같은 셋 · 인코더 고정)과 ρ 만 다른 짝이 된다.
#       E17 checkpoint-5000: 표준 cp 27.02 / 31.42, 둘 겹침 놓침 43.4 / 49.1 %(그때까지 평균 ρ 0.45 이하 — 표본의 절반 넘게 흔든 정답). 시작점 E16 final 은 24.92 / 31.70, 43.2 / 49.6 %.
#       학습률 감쇠는 다르다: E17 은 10,000 step 일정의 절반 지점(최대의 50 %), E19 기본 5,000 step 은 0 까지 내려간다(똑같이 맞추려면 --max-steps 10000 --eval-ck 5000).
#   apex 가 아니면 MAX_RESTART 기본 60(low_p_hpc 선점 — E17 은 22 시간에 재시작 41 번). --partition-eval P 로 평가만 다른 파티션에 낼 수 있다(기본은 학습과 같다).
# 쓰기(스케줄러 셸): [SLOTS=4 SUBSET=eval] bash slurm/submit-q17-e19.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--ramp R0,R1(기본 1.0,1.0)]
#                   [--ovl 0|22|75(기본 0)] [--w-ovl3 W] [--w-ovl2 W] [--w-ovl W] [--partition-eval P(기본 학습과 같음)]
#                   [--n3d-train-chunk N(기본 16 — 추론은 4)] [--run q17-e19*(기본 q17-e19-n3d, --ovl 75 면 q17-e19-n3d-ovl)] [--init DIR(기본 E13 final, --ovl 75 면 E16 final)] [--after JOBID[,…]]
#                   [--no-eval] [--nodes-eval N(기본 2)] [--eval-ck N(기본 0 = final 만)] [--evals-only JOBID|none]
#   SLOTS·SUBSET 은 평가(submit-evals.sh)에 그대로 간다. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; init=""; run=""; ramp=1.0,1.0; tc=16; ev=1; ne=2; ck=0; eo=""; aft=(); ovl=0; w3=""; w2=""; w1=""; pe=""
M=/soundai/Model/VAPASR; D=/soundai/users/tskim/VAPKT-data/data
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --init) init=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;  --ramp) ramp=$2; shift 2 ;;  --n3d-train-chunk) tc=$2; shift 2 ;;  --no-eval) ev=0; shift ;;  --nodes-eval) ne=$2; shift 2 ;;
    --eval-ck) ck=$2; shift 2 ;;  --evals-only) eo=$2; shift 2 ;;  --partition-eval) pe=$2; shift 2 ;;
    --ovl) ovl=$2; shift 2 ;;  --w-ovl3) w3=$2; shift 2 ;;  --w-ovl2) w2=$2; shift 2 ;;  --w-ovl) w1=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; aft=(--after "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
case $ovl in 0) p3=0; p2=0; p1=0 ;;  22) p3=0; p2=3.5; p1=1.0 ;;  75) p3=24; p2=18; p1=6 ;;  *) echo "--ovl 은 0·22·75: $ovl" >&2; exit 2 ;; esac   # 겹침 셋 가중(E13 / E16 / E17·E18)
w3=${w3:-$p3}; w2=${w2:-$p2}; w1=${w1:-$p1}; [[ $w3 =~ ^[0-9.]+$ && $w2 =~ ^[0-9.]+$ && $w1 =~ ^[0-9.]+$ ]] || { echo "--w-ovl3·--w-ovl2·--w-ovl: $w3 $w2 $w1" >&2; exit 2; }
run=${run:-$([[ $ovl == 75 ]] && echo q17-e19-n3d-ovl || echo q17-e19-n3d)}; pe=${pe:-$part}
[[ $part == apex ]] || export MAX_RESTART=${MAX_RESTART:-60}                                         # 선점 파티션(submit-q17-f1 기본 20 — E17 은 22 시간에 41 번)
[[ $run == q17-e19* ]] || { echo "--run 은 q17-e19 로 시작(평가 묶음 q17-e19* — N3D 활동)" >&2; exit 2; }
[[ $ramp =~ ^(0(\.[0-9]+)?|1(\.0+)?),(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "--ramp 는 ρ0,ρ1(각 0–1): $ramp" >&2; exit 2; }
[[ $tc =~ ^[1-9][0-9]?$ && $ne =~ ^[1-8]$ && $steps =~ ^[1-9][0-9]*$ ]] || { echo "--n3d-train-chunk·--nodes-eval·--max-steps: $tc $ne $steps" >&2; exit 2; }
[[ $ck =~ ^[0-9]+$ ]] && (( ck == 0 || (ck < steps && ck % 500 == 0) )) || { echo "--eval-ck 은 0 또는 max-steps 보다 작은 500 의 배수(저장 간격): $ck" >&2; exit 2; }
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-$([[ $ovl == 75 ]] && echo q17-e16-ovl || echo q17-e13)/final}
add=(); nz() { [[ ! $1 =~ ^0*\.?0*$ ]]; }                                                           # 가중이 0 이 아닌 겹침 셋만 더한다
nz "$w3" && add+=(--add-sim "$D/sim-dialogue-v8-ovl3=sim_en-ovl3:$w3"); nz "$w2" && add+=(--add-sim "$D/sim-dialogue-v7-ovl2=sim_en-ovl2:$w2"); nz "$w1" && add+=(--add-sim "$D/sim-dialogue-v6-ovl=sim_en-ovl:$w1")
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
for x in ${add[@]+"${add[@]}"}; do [[ $x == --add-sim ]] && continue; d=${x%%=*}; k=${x#*=}; k=${k%%:*}; [[ -f $d/$k.refined.dialogues.jsonl ]] || { echo "없음: $d/$k.refined.dialogues.jsonl" >&2; exit 1; }; done
subs() { grep -oE "Submitted batch job [0-9]+" | awk '{print $4}' | paste -sd: -; }                 # sbatch 출력 → "id:id"
if [[ -n $eo ]]; then                                                                              # 평가만 다시 건다(학습 job 은 이미 있다)
  [[ $eo == none || $eo =~ ^[0-9]+$ ]] || { echo "--evals-only 는 학습 job 번호 또는 none: $eo" >&2; exit 2; }
  dep=$([[ $eo == none ]] && echo "" || echo "$eo"); echo "E19($run) 평가만 — 학습 job ${dep:-끝남}"
else
  out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps --blocks-trans ${add[@]+"${add[@]}"} --no-eval ${aft[@]+"${aft[@]}"} \
        --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --freeze-encoder --n3d $n3d --n3d-train-chunk $tc --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --act-pred-ramp $ramp" \
        $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E19 학습 제출 실패" >&2; exit 1; }
  echo "$out" | grep -E "^\[|^1 |^sbatch" | sed 's/^/  /' >&2 || true
  dep=$(echo "$out" | grep -oE "^1 학습 job [0-9]+" | awk '{print $4}'); [[ -n $dep ]] || { echo "!! 학습 job 번호를 못 읽었다" >&2; exit 1; }
  echo "E19($run · 시작 ${INIT#$M/} · 활동 입력 ρ $ramp · N3D 학습 청크 $tc · 인코더 고정 · 겹침 셋 ovl3 $w3 + ovl2 $w2 + ovl $w1 · ${steps} step · $part${MAX_RESTART:+ · MAX_RESTART $MAX_RESTART}) 학습 job $dep"
fi
(( ev )) || exit 0
prev=""                                                                                            # 앞 평가 묶음의 job 들 — 다음 묶음은 그 뒤에(한 번에 --nodes-eval 노드까지만)
if (( ck )); then
  o=$(TARGET=checkpoint-$ck bash slurm/submit-evals.sh "$run" ${dep:+--after-job "$dep"} --partition "$pe" --nodes "$ne" $dry); echo "$o" | grep -E "Submitted|^sbatch" || true
  prev=$(echo "$o" | subs); echo "checkpoint-$ck 평가 job ${prev//:/ } — 학습 뒤"
fi
if [[ -n $prev ]]; then o=$(bash slurm/submit-evals.sh "$run" --after-any "$prev" --partition "$pe" --nodes "$ne" $dry)
else o=$(bash slurm/submit-evals.sh "$run" ${dep:+--after-job "$dep"} --partition "$pe" --nodes "$ne" $dry); fi
echo "$o" | grep -E "Submitted|^sbatch" || true
echo "final 평가 job $(echo "$o" | subs | tr ':' ' ') — $([[ -n $prev ]] && echo "checkpoint-$ck 평가 뒤" || echo "학습 뒤")"
