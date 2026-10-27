#!/usr/bin/env bash
# E1c·E5·E6 한꺼번에 제출(2026-10-27, 사용자 "E5~7 학습 제출할 수 있도록 SLURM 명령어 알려줘. 총 6개 노드를 사용할 수 있다는 점을 참고해서 명령어를 작성해줘."
# — wiki output-diarization-integration-20261025 6 절, task-data-expansion-20261026). E7(AOSC 회의 통째·실시간)은 아직 코드가 없어 여기 없다.
#   1) E1c 인과 화자 분리 가지 diar-q17-diar-e1c(1 노드, submit-diar-train.sh): E1 final 가중치에서 lookahead 0(앞보기는 위치 conv 만, 어텐션 엄격 인과)으로,
#      E6 과 같은 시작 모델(5 단계 ck3000 — stage 3 는 인코더도 학습하므로 E1 의 mt4 final 인코더와 다르다)의 인코더 위에서, 원거리 회의(ami·icsi·notsofar·chime6tr-far)·
#      CHiME-6 train·NIA23 회의·Switchboard·회의형 합성 v4 를 더해 10,000 step(lr 1e-4, warmup 500 — E1 은 20,000 step 2.5 시간).
#   2) E5 한 스트림 + 슬롯 활동 입력 q17-e5(2 노드): F1 레시피(submit-q17-f1.sh — 5,000 step·새 데이터) + --slot-input(정답 활동을 act_noise 로 흔들어 입력)
#      → 평가 q17-e5(AMI 정답 활동 8 몫 + 활동 없이 4 몫).
#   3) E6 인과 헤드 + 슬롯 활동 입력 q17-e6(2 노드, E1c 성공 뒤 afterok): F1 레시피 + --slot-input --diar-head <E1c final>(ρ 0.3→0.9 로 헤드 예측/흔든 정답)
#      → 평가 q17-e6(F1 과 같은 묶음을 헤드 예측 활동으로 + AMI 정답 활동 상한).
#   노드(6 노드 안): 처음 E1c 1 + E5 2 = 3 → E1c 가 끝나면(약 1.3 시간) E6 2 → 학습이 끝날 때마다 평가 노드 1. 학습 레시피·평가 체크포인트는 F1 과 같다(비교 기준 F1).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e56.sh [--dry-run] [--partition P(기본 apex)] [--only e1c|e5|e6] [--e1c-job JOBID(E1c 를 이미 냈으면 E6 을 그 뒤로)].
#   아무것도 지우지 않는다. 산출: /soundai/Model/VAPASR/diar-q17-diar-e1c, stage3-q17-e5, stage3-q17-e6.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; only=""; e1c=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --only) only=$2; shift 2 ;;  --e1c-job) e1c=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ -z $only || $only =~ ^(e1c|e5|e6)$ ]] || { echo "--only e1c|e5|e6" >&2; exit 2; }
[[ -z $e1c || $e1c =~ ^[0-9]+$ ]] || { echo "--e1c-job 은 job id" >&2; exit 2; }
D=/soundai/users/tskim/VAPKT-data/data; M=/soundai/Model/VAPASR
INIT=$M/stage3-q17-st3-p11-revive/checkpoint-3000; E1=$M/diar-q17-diar-e1/final/diar.pt; E1C=$M/diar-q17-diar-e1c/final/diar.pt
for f in $INIT/config.json $E1; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
want() { [[ -z $only || $only == "$1" ]]; }
jid() { grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1; }

if want e1c && [[ -z $e1c ]]; then
  [[ -e $M/diar-q17-diar-e1c/DONE ]] && { echo "E1c 는 이미 끝났다: $M/diar-q17-diar-e1c/DONE — E6 만 내려면 --only e6" >&2; exit 1; }
  CORP=aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar,ami-far,icsi-far,notsofar-far,chime6tr,chime6tr-far,nia23mt,swbd
  SIMS=$D/sim-dialogue-v2,$D/sim-dialogue-v3-multi,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1,$D/sim-dialogue-v4-meet
  W=ami:3,icsi:2,notsofar:0.5,ami-far:3,icsi-far:2,notsofar-far:0.5,chime6tr:1,chime6tr-far:1.5,nia23mt:1,swbd:0.5,sim-dialogue-v3-multi:3,sim-dialogue-v4-meet:3,sim-dialogue-v2:1,sim-dialogue-xl-v1:0.5,sim-dialogue-xl-tel-v1:0.5
  out=$(bash slurm/submit-diar-train.sh --run q17-diar-e1c --init $INIT --partition $part --nodes 1 --max-steps 10000 --corpora $CORP --sim-data $SIMS --weights $W \
        --extra "--lookahead 0 --init-branch $E1 --lr 1e-4 --warmup 500" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E1c 제출 실패" >&2; exit 1; }
  echo "$out" | grep -E "^sbatch|화자 분리" >&2 || true
  e1c=$(echo "$out" | jid || true); [[ -n $e1c ]] || { [[ -n $dry ]] && e1c=0; } || true
  [[ -n $e1c ]] || { echo "!! E1c job 번호를 못 읽었다" >&2; exit 1; }
  echo "E1c 학습 job $e1c"
fi
if want e5; then
  out=$(bash slurm/submit-q17-f1.sh --run q17-e5 --partition $part --train-extra "--slot-input" --eval-set q17-e5 $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E5 제출 실패" >&2; exit 1; }
  echo "$out" | grep -E "^\[|^[12] " | sed 's/^/  /' >&2 || true
  echo "E5 $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
fi
if want e6; then
  [[ -n $e1c ]] || [[ -f $E1C ]] || { echo "E6 은 E1c 뒤 — --e1c-job JOBID 를 주거나 E1c 를 함께 낸다($E1C 없음)" >&2; exit 2; }
  out=$(bash slurm/submit-q17-f1.sh --run q17-e6 --partition $part ${e1c:+--afterok $e1c} --train-extra "--slot-input --diar-head $E1C" --eval-set q17-e6 $dry 2>&1) \
    || { echo "$out" | tail -5 >&2; echo "!! E6 제출 실패" >&2; exit 1; }
  echo "$out" | grep -E "^\[|^[12] " | sed 's/^/  /' >&2 || true
  echo "E6 $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
fi
