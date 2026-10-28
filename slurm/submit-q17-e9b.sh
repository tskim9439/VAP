#!/usr/bin/env bash
# E9b 슬롯 블록 형식 + 긴 학습 창 제출(2026-10-28, 사용자 "학습 창을 90 s 까지 늘려서 E9b 구현해놓자" — wiki output-diarization-integration-20261025 9 절 E9b).
#   E9(slurm/submit-q17-e9.sh)는 학습 창 20–40 s(청크당 위치가 E6 의 2.3 배 — 40 s 면 약 3,200 위치)라 평가 창 60–90 s 에서 <EOT> 반복으로 무너졌다.
#   E9b 는 E9 와 같고 학습 창만 --window 20,90(평가 창과 같은 범위) + --shrink-speakers(화자 상한 4 를 넘는 긴 회의 창은 버리지 않고 다섯 번째 화자 직전까지 줄임)다.
# 아래는 E9 와 같다.
#   E6 final 에서 F1 레시피를 받아쓰기만으로(submit-q17-f1.sh --transcribe-only — 데이터·가중치·lr 4e-5 는 같고 번역·교대/머무름 가중·EOT-문맥 dropout 없음,
#   replay 번역도 끔) 학습한다. 덧붙임:
#     --slot-input --diar-head <E1c diar.pt>(E6 과 같은 cfg — 헤드는 E6 final 의 것을 이어 학습)
#     --slot-blocks(청크마다 오디오 뒤 선언 슬롯 태그를 모두 입력, 각 태그 뒤 그 슬롯 출력 <BEGIN>·전사·<EOT>, 멈춤 = 다음 입력 위치 label <NEXT_AUDIO>,
#       <BEGIN> = <ONSET> 151721 가중 2, 태그 임베딩에 그 슬롯 최근 12 청크 활동) --declare-range(선언 슬롯 수를 [창 화자 수, 4] 에서 고르게)
#     --global-slots P(대화 전체 도착 순서 슬롯 — E7b 와 같다) --freeze-encoder(E7a 헤드가 E6 인코더 위에서 학습됐다 — 회의 통째 평가에서 끼운다).
#   step 5,000(E9 와 같다), 2 노드, 저장 500. 평가 q17-e9b: AMI 창 δ10 8 몫 + 회의 통째 AOSC·E7a 헤드 δ10 8 몫(행에 LLM DER·청크 ms).
#   GPU 당 토큰 예산(16,384)은 같아서 90 s 창(4 슬롯 약 7,000 위치)은 배치에 둘 남짓이다 — step 당 오디오 양은 E9 와 비슷하고 어텐션 비용만 는다.
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e9b.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--global-slots P(기본 0.5)]
#                   [--window lo,hi(기본 20,90)] [--init DIR(기본 E6 final — E9 와 같은 시작점)] [--after JOBID[,JOBID…](모두 끝난 뒤 — afterany)] [--no-eval].
#   아무것도 지우지 않는다. 산출 /soundai/Model/VAPASR/stage3-q17-e9b.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; gs=0.5; dep=(); ev=(); win=20,90; init=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --global-slots) gs=$2; shift 2 ;;  --no-eval) ev=(--no-eval); shift ;;
    --window) win=$2; shift 2 ;;  --init) init=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR; INIT=${init:-$M/stage3-q17-e6/final}; E1C=$M/diar-q17-diar-e1c/final/diar.pt; E7A=$M/diar-q17-diar-e7a/final/diar.pt
[[ $win =~ ^[0-9.]+,[0-9.]+$ ]] || { echo "--window lo,hi: $win" >&2; exit 2; }
for f in $INIT/config.json $E1C $E7A; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
out=$(bash slurm/submit-q17-f1.sh --run q17-e9b --init $INIT --partition $part --max-steps $steps --transcribe-only --window $win ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --shrink-speakers --global-slots $gs --freeze-encoder" --eval-set q17-e9b $dry 2>&1) \
  || { echo "$out" | tail -5 >&2; echo "!! E9b 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E9b $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
