#!/usr/bin/env bash
# E7b LLM 회의 단위 슬롯 학습 제출(2026-10-28, 사용자 "… 3. E7b(학습). E6 final 에서 --global-slots 로 LLM 이 회의 단위 슬롯을 따르게 학습합니다 … 진행해보자"
# — wiki output-diarization-integration-20261025 7 절 결과 3: E6 LLM + E7a 헤드 + AOSC 회의 통째 cpWER 의 귀속이 46 pt — LLM 이 헤드의 회의 단위 슬롯을 따르지 않는다).
#   E6 final 에서 F1 레시피(submit-q17-f1.sh — 데이터·가중·lr 4e-5)로 짧게 미세조정한다. 덧붙임:
#     --slot-input --diar-head <E1c diar.pt>(E6 과 같은 cfg — 헤드는 E6 final 의 것을 그대로 이어 학습) --global-slots P(학습 창을 확률 P 로 대화 전체 도착 순서 슬롯으로
#     — 같은 언어 대화·대화 화자 ≤ 4, 헤드 손실은 창 도착 순서·슬롯 활동 입력은 정답 자리로 맞춘 헤드 예측: diar_align 자동) --freeze-encoder(E7a 헤드가 E6 인코더 위에서
#     학습됐다 — 평가에서 E7a 헤드를 끼운다).
#   step 2,500(저장 500 — 평가 ck1500·ck2000 은 쓰지 않는다), 2 노드. 평가 q17-e7b: final 을 회의 통째 AOSC + E7a 헤드(8 몫) + AMI 창(4 몫 — 창 귀속이 나빠지지 않았는지).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e7b.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 2500)] [--global-slots P(기본 0.5)]
#                   [--after JOBID[,JOBID…](모두 끝난 뒤 — afterany)] [--no-eval]. 아무것도 지우지 않는다. 산출 /soundai/Model/VAPASR/stage3-q17-e7b.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=2500; gs=0.5; dep=(); ev=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --global-slots) gs=$2; shift 2 ;;  --no-eval) ev=(--no-eval); shift ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR; INIT=$M/stage3-q17-e6/final; E1C=$M/diar-q17-diar-e1c/final/diar.pt; E7A=$M/diar-q17-diar-e7a/final/diar.pt
for f in $INIT/config.json $E1C $E7A; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
out=$(bash slurm/submit-q17-f1.sh --run q17-e7b --init $INIT --partition $part --max-steps $steps ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --global-slots $gs --freeze-encoder" --eval-set q17-e7b $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E7b 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E7b $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
