#!/usr/bin/env bash
# E10 한 스트림 안 슬롯별 소리 토큰 학습 제출(2026-10-29, 사용자 "E10 구현해놓자" — wiki output-diarization-integration-20261025 10 절, 그림 https://claude.ai/artifact/GB7BVFF3kktDWN1oVzdmoV).
#   E9 final 에서 E9 와 같은 레시피(submit-q17-f1.sh --transcribe-only — F1 데이터·가중치·lr 4e-5, 받아쓰기만)에 슬롯 추출기를 더한다:
#     --slot-input --diar-head <E1c diar.pt>(E9 의 헤드를 이어 학습) --slot-blocks --declare-range --global-slots P --freeze-encoder(E7a 헤드를 회의 통째 평가에 끼운다)
#     --slot-audio(vapasr/hf/slot_audio — 인코더 마지막 층 + 화자 단서 임베딩(헤드 embed 로 시작) + 슬롯 활동·단독 활동 누적 프로필 → 인과 conv 블록 4 개 →
#       슬롯 소리 토큰을 E9 태그 자리에 더한다, 출력 0 초기화) --mix-drop 0.5(학습 표본마다 섞인 소리 토큰을 빈 임베딩으로 — 평가는 켜고 끈 둘 다).
#   step 5,000(E9 와 같다), 2 노드, 저장 500. 학습 창은 E9 와 같은 20–40 s(--window 로 바꿀 수 있다 — E9b 의 20,90 등).
#   평가 q17-e10: GPU 마다 AMI 창 δ10 → 회의 통째 AOSC·E7a δ10 → 섞인 소리 끈 창 → 섞인 소리 끈 회의 통째(행에 LLM DER·청크 ms).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e10.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--global-slots P(기본 0.5)] [--mix-drop P(기본 0.5)]
#                   [--window lo,hi] [--init DIR(기본 E9 final)] [--after JOBID[,JOBID…](모두 끝난 뒤 — afterany)] [--no-eval].
#   아무것도 지우지 않는다. 산출 /soundai/Model/VAPASR/stage3-q17-e10.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; gs=0.5; md=0.5; dep=(); ev=(); win=(); init=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --global-slots) gs=$2; shift 2 ;;  --mix-drop) md=$2; shift 2 ;;
    --no-eval) ev=(--no-eval); shift ;;  --window) win=(--window "$2"); shift 2 ;;  --init) init=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR; INIT=${init:-$M/stage3-q17-e9/final}; E1C=$M/diar-q17-diar-e1c/final/diar.pt; E7A=$M/diar-q17-diar-e7a/final/diar.pt
for f in $INIT/config.json $E1C $E7A; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
out=$(bash slurm/submit-q17-f1.sh --run q17-e10 --init $INIT --partition $part --max-steps $steps --transcribe-only ${win[@]+"${win[@]}"} ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --slot-audio --mix-drop $md --global-slots $gs --freeze-encoder" --eval-set q17-e10 $dry 2>&1) \
  || { echo "$out" | tail -5 >&2; echo "!! E10 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E10 $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
