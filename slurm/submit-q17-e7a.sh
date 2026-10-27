#!/usr/bin/env bash
# E7a AOSC 화자 분리 헤드 학습 제출(2026-10-27, 사용자 "E7 을 먼저 구현해놓자" — wiki output-diarization-integration-20261025 7 절):
#   E6 final 의 인코더(동결 — stage 3 는 인코더도 학습했으므로 E6 것) 위에서 E6 헤드(experiments/diar_extract.py → diar-q17-e6-head)를
#   AOSC 스트리밍 흉내로 미세조정한다(diar_train --aosc: 긴 클립 90–120 s · 학습 청크 47 · 캐시 188 · FIFO 188 · 갱신 94 · 캐시 화자 순서 섞기 0.5,
#   캐시는 클립마다 반은 정답·반은 모델 예측(--aosc-self 0.5 — NeMo 의 스트리밍 학습은 모델 예측) — Streaming Sortformer 가 오프라인 모델을 스트리밍으로
#   미세조정한 것과 같은 단계). 데이터·가중은 E1c 와 같다(원거리·새 코퍼스·회의형 합성). 1 노드.
#   평가(이 job 성공 뒤): q17-e7-amifull — E6 final 에 이 헤드를 끼워 AMI 회의 통째 AOSC(8 몫). 회의 단위 DER 은 e7_session_eval.sh(호스트 GPU).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e7a.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--no-eval]. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; ev=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --no-eval) ev=0; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR; D=/soundai/users/tskim/VAPKT-data/data; INIT=$M/stage3-q17-e6/final; HEAD=$M/diar-q17-e6-head/diar.pt
for f in $INIT/config.json $HEAD; do
  [[ -f $f ]] || { echo "없음: $f — E6 헤드는 PYTHONPATH=. python experiments/diar_extract.py --model $INIT --out $M/diar-q17-e6-head" >&2; exit 1; }
done
CORP=aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar,ami-far,icsi-far,notsofar-far,chime6tr,chime6tr-far,nia23mt,swbd
SIMS=$D/sim-dialogue-v2,$D/sim-dialogue-v3-multi,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1,$D/sim-dialogue-v4-meet
W=ami:3,icsi:2,notsofar:0.5,ami-far:3,icsi-far:2,notsofar-far:0.5,chime6tr:1,chime6tr-far:1.5,nia23mt:1,swbd:0.5,sim-dialogue-v3-multi:3,sim-dialogue-v4-meet:3,sim-dialogue-v2:1,sim-dialogue-xl-v1:0.5,sim-dialogue-xl-tel-v1:0.5
out=$(bash slurm/submit-diar-train.sh --run q17-diar-e7a --init $INIT --partition $part --nodes 1 --max-steps $steps --batch 2 --corpora $CORP --sim-data $SIMS --weights $W \
      --extra "--aosc --aosc-self 0.5 --window 90,120 --lookahead 0 --init-branch $HEAD --lr 1e-4 --warmup 300 --save-every 1000" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E7a 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^sbatch|화자 분리" >&2 || true
dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true); [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true
[[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
echo "1 E7a 학습 job $dep"
(( ev )) || exit 0
out=$(bash slurm/submit-evals.sh q17-e7-amifull --after-job "$dep" --partition "$part" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^목록|^sbatch" >&2 || true
echo "2 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
