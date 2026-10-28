#!/usr/bin/env bash
# E9 슬롯 블록 형식 학습 제출(2026-10-28, 사용자 "응 그렇게 E9 구현해놓자" · "<SPK_*> 의 수를 추후에 조절할 수 있도록 해줘. 즉 최소 샘플의 화자 수, 최대 4명 화자로 해서
# 랜덤으로 선택되게끔" — wiki output-diarization-integration-20261025 9 절).
#   E6 final 에서 F1 레시피를 받아쓰기만으로(submit-q17-f1.sh --transcribe-only — 데이터·가중치·lr 4e-5 는 같고 번역·교대/머무름 가중·EOT-문맥 dropout 없음,
#   replay 번역도 끔) 학습한다. 덧붙임:
#     --slot-input --diar-head <E1c diar.pt>(E6 과 같은 cfg — 헤드는 E6 final 의 것을 이어 학습)
#     --slot-blocks(청크마다 오디오 뒤 선언 슬롯 태그를 모두 입력, 각 태그 뒤 그 슬롯 출력 <BEGIN>·전사·<EOT>, 멈춤 = 다음 입력 위치 label <NEXT_AUDIO>,
#       <BEGIN> = <ONSET> 151721 가중 2, 태그 임베딩에 그 슬롯 최근 12 청크 활동) --declare-range(선언 슬롯 수를 [창 화자 수, 4] 에서 고르게)
#     --global-slots P(대화 전체 도착 순서 슬롯 — E7b 와 같다) --freeze-encoder(E7a 헤드가 E6 인코더 위에서 학습됐다 — 회의 통째 평가에서 끼운다).
#   step 5,000(E6 과 같다), 2 노드, 저장 500. 평가 q17-e9: AMI 창 δ10 8 몫 + 회의 통째 AOSC·E7a 헤드 δ10 8 몫(행에 LLM DER·청크 ms).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e9.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--global-slots P(기본 0.5)]
#                   [--after JOBID[,JOBID…](모두 끝난 뒤 — afterany)] [--no-eval]. 아무것도 지우지 않는다. 산출 /soundai/Model/VAPASR/stage3-q17-e9.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; gs=0.5; dep=(); ev=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --global-slots) gs=$2; shift 2 ;;  --no-eval) ev=(--no-eval); shift ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR; INIT=$M/stage3-q17-e6/final; E1C=$M/diar-q17-diar-e1c/final/diar.pt; E7A=$M/diar-q17-diar-e7a/final/diar.pt
for f in $INIT/config.json $E1C $E7A; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
out=$(bash slurm/submit-q17-f1.sh --run q17-e9 --init $INIT --partition $part --max-steps $steps --transcribe-only ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots $gs --freeze-encoder" --eval-set q17-e9 $dry 2>&1) \
  || { echo "$out" | tail -5 >&2; echo "!! E9 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E9 $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
