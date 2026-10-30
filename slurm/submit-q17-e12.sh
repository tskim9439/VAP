#!/usr/bin/env bash
# E12 활동 추종 학습 제출(2026-10-30, 사용자 "LLM 활동 추종 학습 구현해놓자" — wiki output-n3d-branch-20261029 6.8 절).
#   진단(E11 회의 통째): LLM 이 활동을 어긴 단어는 한 사람만 말하는 구간(90 %)에서 한 차례를 통째로 잘못된 슬롯에 쓴다(차례 앞뒤 고르게).
#     고른 슬롯은 절반이 직전 화자 슬롯, 1/3 이 30 s 넘게 말이 없던 슬롯이다 — LLM 이 슬롯을 활동이 아니라 자기 화자 추적·순서 습관으로 고른다.
#   E11 final 에서 E11 레시피(submit-q17-e11.sh — 받아쓰기만, 슬롯 블록, 고정 Nemotron 3 Diarization 활동, 헤드 손실 그대로)에 셋을 더한다.
#     --slot-shuffle P(기본 0.5): 선언 슬롯 안 화자 자리를 무작위로 — "창 안 먼저 말한 순서 = 슬롯" 습관을 지운다.
#     --slot-remap P(기본 0.3): 창 안 무음 틈에서 슬롯 대응을 바꾼다 — 같은 목소리라도 활동이 가리키는 슬롯을 따르게.
#     --act-noise(기본 아래 N3D 수준): 흔든 정답 입력을 실제 전문가 활동만큼만 틀리게(E11 은 E1c 헤드 오류를 흉내 낸 기본값 — 흔든 표본 70 %, 말 적은 화자 40 % 통째 누락).
#   step 2,500(기본 — 평가 체크포인트 60·80 % = 1,500·2,000 이 저장 간격으로 나눠떨어진다). 평가 q17-e12: AMI 창 δ10 8 몫 → 회의 통째 δ10 8 몫(행에 활동 추종 셈 — on·다른 슬롯·따르면 cp).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e12.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 2500)] [--save-every N] [--shuffle P] [--remap P]
#                   [--act-noise SPEC | e11(E11 과 같은 기본 잡음)] [--global-slots P(기본 0.5)] [--init DIR] [--n3d PATH] [--run q17-e12*] [--after JOBID[,…]] [--no-eval].
#   low_p_hpc: --partition low_p_hpc — submit-q17-f1.sh 가 선점 파티션 설정을 자동으로 건다(저장 250 step, MAX_RESTART 20, requeue). 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=2500; save=(); gs=0.5; sh=0.5; rm=0.3; dep=(); ev=(); init=""; run=q17-e12
an="clean=0.6,drop=0.02,short=0.15,short_frac=0.05,jitter=2,ovl=0.15,fa=0.002"
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --save-every) save=(--save-every "$2"); shift 2 ;;  --global-slots) gs=$2; shift 2 ;;
    --shuffle) sh=$2; shift 2 ;;  --remap) rm=$2; shift 2 ;;  --act-noise) an=$2; shift 2 ;;  --run) run=$2; shift 2 ;;
    --no-eval) ev=(--no-eval); shift ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e12* ]] || { echo "--run 은 q17-e12 로 시작(평가 묶음 q17-e12*)" >&2; exit 2; }
for x in "$sh" "$rm" "$gs"; do [[ $x =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || { echo "확률은 [0, 1]: $x" >&2; exit 2; }; done
AN=""; [[ $an == e11 ]] || AN=" --act-noise $an"
M=/soundai/Model/VAPASR; E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e11/final}
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps ${save[@]+"${save[@]}"} --transcribe-only ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots $gs --freeze-encoder --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-shuffle $sh --slot-remap $rm$AN" \
      --eval-set $run $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E12 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E12($run · 섞기 $sh · 바꾸기 $rm · 잡음 ${an}) $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
