#!/usr/bin/env bash
# E8 헤드 출력 지연 R 제출(2026-10-27, 사용자 "E8 구현해놓자" — wiki output-diarization-integration-20261025 8 절):
#   E6 과 같은 레시피(submit-q17-f1.sh — 5 단계 ck3000 시작, F1 데이터, 5,000 step, 2 노드, --slot-input --diar-head <E1c final>)에 --act-shift R 만 더한다.
#   헤드 위치 p 의 출력이 프레임 p − R 의 활동을 맞히고(인과 헤드가 R 프레임 뒤를 보고 판단), 그 출력을 같은 위치의 오디오 토큰에 넣는다 — 받아쓰기 지연 δ ≥ R 이면
#   지연이 늘지 않는다. R 기본 4(= 학습 지연 δ4–10 의 가장 짧은 값, 320 ms). E6 대비 차이는 이것 하나(같은 시작점·데이터·step·헤드 시작점).
#   평가: q17-e8(q17-e6 과 같은 v3 묶음 — 헤드 예측 활동, ck3000·ck4000·final·avg3, AMI 4 몫씩). AOSC(회의 통째)와는 아직 함께 쓰지 않는다.
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e8.sh [--dry-run] [--partition P(기본 apex)] [--shift R(기본 4)] [--after JOBID[,JOBID…](모두 끝난 뒤 — afterany)]
#                   [--afterok JOBID] [--no-eval]. 평가 job 이름은 submit-evals.sh 기본(SA_EVAL_FullDuplex).
#   아무것도 지우지 않는다. 산출: /soundai/Model/VAPASR/stage3-q17-e8(--shift 가 4 가 아니면 stage3-q17-e8-r<R>).
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; R=4; dep=(); ev=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --shift) R=$2; shift 2 ;;  --no-eval) ev=(--no-eval); shift ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    --afterok) [[ ${2:-} =~ ^[0-9]+$ ]] || { echo "$1 은 job id" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $R =~ ^[1-4]$ ]] || { echo "--shift 1–4(가장 짧은 학습 지연 δ4 를 넘지 않게): $R" >&2; exit 2; }
M=/soundai/Model/VAPASR; E1C=$M/diar-q17-diar-e1c/final/diar.pt
[[ -f $E1C ]] || { echo "없음: $E1C" >&2; exit 1; }
run=q17-e8; [[ $R == 4 ]] || run=q17-e8-r$R
[[ $run == q17-e8 ]] || { echo "평가 묶음 q17-e8 은 run q17-e8 만 잰다 — R ≠ 4 면 --no-eval 로 내고 평가는 따로" >&2; [[ ${#ev[@]} -gt 0 ]] || exit 2; }
out=$(bash slurm/submit-q17-f1.sh --run $run --partition $part ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} --train-extra "--slot-input --diar-head $E1C --act-shift $R" --eval-set q17-e8 $dry 2>&1) \
  || { echo "$out" | tail -5 >&2; echo "!! E8 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E8 $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
