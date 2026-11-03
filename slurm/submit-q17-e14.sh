#!/usr/bin/env bash
# E14 학습 제출(2026-11-03, 사용자 "N3D 가 놓친 말을 받아쓰도록 게이트를 고칠 방법을 찾아줘" → 비귀속 칸 + "1 번(활동 입력을 N3D 확률값으로…)도 함께"
#   — wiki output-e13-gate-trans-20261102 8 절, vapasr/data/cst_dialogue 모듈 설명 x_slot).
#   E13 final 에서 E13 레시피(아래 ①·② — E11 레시피 + 활동 게이트 + 번역)에 셋을 더한다.
#     ③ 비귀속 칸 --x-slot: 라운드마다 슬롯 칸 뒤 X 칸(<SPK_5>). 학습 게이트는 정답 활동에서 N3D 놓침을 흉내 낸다(--miss-short·--miss-drop·--miss-late) — 닫힌 슬롯의 글자는
#        X 칸이 내고, 흉내 낸 놓침 프레임은 전문가 활동 입력에서도 지운다(act_miss). 칸 열림은 디코더와 같다(활동 창 ∪ 열린 말).
#     ④ 확률 활동 --n3d-soft: 전문가 활동을 0/1 대신 확률 그대로(학습·추론, 게이트는 > 0.5). X 칸 단어는 평가에서 확률(임계 아래 포함)이 가장 큰 슬롯에 붙인다(attribute_x).
#   E13 레시피:
#     ① 활동 게이트 --slot-gate N(기본 12 = 태그 활동 칸 수): 라운드마다 정답 활동이 최근 N 청크 안에 있는(또는 출력이 있는) 슬롯의 태그만 넣는다.
#        디코드는 예측 활동 창 ∪ 열린 발화(<ONSET> 뒤 <EOT> 전 — model.config.slot_gate). --gate-fa P(기본 0.1): 슬롯마다 그 확률로 말 없는 구간 하나를 더 연다.
#     ② 번역(submit-q17-f1.sh --blocks-trans): 대화 번역 타깃 창의 50 %·교차 언어 합성 80 % 를 번역 모드로 — 슬롯 칸마다 [<TRANS>] 번역 칸, replay 단일 화자 번역 80 %.
#        --no-trans 면 게이트만(받아쓰기만 — --transcribe-only).
#   step 5,000(E13 과 같다). 평가 q17-e14: AMI 창 δ10 8 몫 → 회의 통째 δ10 8 몫(디코드가 게이트·X 칸을 건다), 번역은 묶음 q17-e14-trans 로 따로.
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e14.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--save-every N] [--gate N(기본 12)] [--gate-fa P(기본 0.1)]
#                   [--miss-short P(기본 0.3)] [--miss-drop P(기본 0.1)] [--miss-late P(기본 0.3)] [--no-soft] [--no-trans] [--global-slots P(기본 0.5)]
#                   [--init DIR(기본 E13 final)] [--n3d PATH] [--run q17-e14*] [--after JOBID[,…]] [--no-eval].
#   low_p_hpc: --partition low_p_hpc — submit-q17-f1.sh 가 선점 파티션 설정을 자동으로 건다(저장 250 step, MAX_RESTART 20, requeue). 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; save=(); gs=0.5; gate=12; gfa=0.1; tr=1; dep=(); ev=(); init=""; run=q17-e14; msh=0.3; mdr=0.1; mla=0.3; soft=1
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --save-every) save=(--save-every "$2"); shift 2 ;;  --global-slots) gs=$2; shift 2 ;;
    --gate) gate=$2; shift 2 ;;  --gate-fa) gfa=$2; shift 2 ;;  --no-trans) tr=0; shift ;;  --run) run=$2; shift 2 ;;
    --miss-short) msh=$2; shift 2 ;;  --miss-drop) mdr=$2; shift 2 ;;  --miss-late) mla=$2; shift 2 ;;  --no-soft) soft=0; shift ;;
    --no-eval) ev=(--no-eval); shift ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e14* ]] || { echo "--run 은 q17-e14 로 시작(평가 묶음 q17-e14*)" >&2; exit 2; }
[[ $gate =~ ^[1-9][0-9]*$ ]] || { echo "--gate 는 1 이상 정수(태그 활동 칸 수 12 이하): $gate" >&2; exit 2; }
for x in "$gfa" "$gs" "$msh" "$mdr" "$mla"; do [[ $x =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "확률은 [0, 1]: $x" >&2; exit 2; }; done
M=/soundai/Model/VAPASR; E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e13/final}
SOFT=""; (( soft )) && SOFT=--n3d-soft
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
MODE=(--blocks-trans); (( tr )) || MODE=(--transcribe-only)
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps ${save[@]+"${save[@]}"} "${MODE[@]}" ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots $gs --freeze-encoder --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate $gate --gate-fa $gfa --x-slot --miss-short $msh --miss-drop $mdr --miss-late $mla $SOFT" \
      --eval-set $run $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E14 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E14($run · 게이트 $gate · 헛 구간 $gfa · X 칸 놓침 흉내 짧은 $msh·통째 $mdr·늦게 $mla · $( (( soft )) && echo 확률 활동 || echo 0/1 활동) · $( (( tr )) && echo 번역 || echo 받아쓰기만)) $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
