#!/usr/bin/env bash
# E13 학습 제출(2026-11-02, 사용자 "번역을 다시 넣어서 학습할 수 있도록 코드를 작성해줘. … <SPK_A~D> 까지 활성화할 필요는 없을거 같아. 이 고정 슬롯도 해제해보자"
#   — wiki output-e13-gate-trans-20261102, vapasr/data/cst_dialogue 모듈 설명 E13).
#   E11 final 에서 E11 레시피(submit-q17-e11.sh — 슬롯 블록, 고정 Nemotron 3 Diarization 활동, 헤드 손실 그대로, 인코더 고정, 선언 수 범위, global slots 0.5)에 둘을 더한다.
#     ① 활동 게이트 --slot-gate N(기본 12 = 태그 활동 칸 수): 라운드마다 정답 활동이 최근 N 청크 안에 있는(또는 출력이 있는) 슬롯의 태그만 넣는다.
#        디코드는 예측 활동 창 ∪ 열린 발화(<ONSET> 뒤 <EOT> 전 — model.config.slot_gate). --gate-fa P(기본 0.1): 슬롯마다 그 확률로 말 없는 구간 하나를 더 연다.
#     ② 번역(submit-q17-f1.sh --blocks-trans): 대화 번역 타깃 창의 50 %·교차 언어 합성 80 % 를 번역 모드로 — 슬롯 칸마다 [<TRANS>] 번역 칸, replay 단일 화자 번역 80 %.
#        --no-trans 면 게이트만(받아쓰기만 — --transcribe-only).
#   step 5,000(E11 과 같다 — 형식이 바뀌고 번역이 다시 든다). 평가 q17-e13: AMI 창 δ10 8 몫 → 회의 통째 δ10 8 몫(받아쓰기 — 디코드가 게이트를 건다).
#   번역 평가는 학습 뒤 experiments/cst_stage4_eval.py(슬롯 블록 모델이면 번역 칸 디코드)로 따로 잰다.
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e13.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--save-every N] [--gate N(기본 12)] [--gate-fa P(기본 0.1)]
#                   [--no-trans] [--global-slots P(기본 0.5)] [--init DIR(기본 E11 final)] [--n3d PATH] [--run q17-e13*] [--after JOBID[,…]] [--no-eval].
#   low_p_hpc: --partition low_p_hpc — submit-q17-f1.sh 가 선점 파티션 설정을 자동으로 건다(저장 250 step, MAX_RESTART 20, requeue). 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; save=(); gs=0.5; gate=12; gfa=0.1; tr=1; dep=(); ev=(); init=""; run=q17-e13
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --save-every) save=(--save-every "$2"); shift 2 ;;  --global-slots) gs=$2; shift 2 ;;
    --gate) gate=$2; shift 2 ;;  --gate-fa) gfa=$2; shift 2 ;;  --no-trans) tr=0; shift ;;  --run) run=$2; shift 2 ;;
    --no-eval) ev=(--no-eval); shift ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e13* ]] || { echo "--run 은 q17-e13 으로 시작(평가 묶음 q17-e13*)" >&2; exit 2; }
[[ $gate =~ ^[1-9][0-9]*$ ]] || { echo "--gate 는 1 이상 정수(태그 활동 칸 수 12 이하): $gate" >&2; exit 2; }
for x in "$gfa" "$gs"; do [[ $x =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "확률은 [0, 1]: $x" >&2; exit 2; }; done
M=/soundai/Model/VAPASR; E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e11/final}
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
MODE=(--blocks-trans); (( tr )) || MODE=(--transcribe-only)
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps ${save[@]+"${save[@]}"} "${MODE[@]}" ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots $gs --freeze-encoder --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate $gate --gate-fa $gfa" \
      --eval-set $run $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E13 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E13($run · 게이트 $gate · 헛 구간 $gfa · $( (( tr )) && echo 번역 || echo 받아쓰기만)) $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
