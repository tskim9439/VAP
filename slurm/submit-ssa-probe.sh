#!/usr/bin/env bash
# P1 화자 조건 인코더 단독 검증 제출(2026-11-09, 사용자 "먼저 LLM 없이 인코더만으로 화자를 가를 수 있는지 검증해보자").
#   학습 job 하나(노드 1 · 8 GPU, experiments/ssa_probe.py train) + 평가 job 하나(학습 성공 뒤, 같은 조각·같은 활동으로 판 4 개):
#     parakeet-n3d   공개 Multitalker Parakeet(1.12 s 청크, 우측 문맥 13) + 저장 N3D 활동 — 조각 평가 방식의 기준
#     probe-n3d      학습한 화자 조건 Nemotron 3.5(0.32 s 청크, 우리 인코더 설정) + N3D 활동
#     probe-oracle   같은 것 + 정답 활동(화자 분리 오류를 뺀 상한)
#     probe-n3d-rc13 같은 것을 1.12 s 청크로(Parakeet 과 같은 앞보기)
#   쓰기(스케줄러 셸): bash slurm/submit-ssa-probe.sh [--partition P(기본 hpc)] [--steps N(기본 6000)] [--run 이름(기본 p1)] [--dry-run] [--eval-only]
#   산출: 학습 /soundai/Model/VAPASR/ssa-probe-<이름>/(step-N.pt·final.pt), 평가 VAPKT-data/results/ssa-probe-<이름>-<판>. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
part=hpc; steps=6000; run=p1; dry=0; evo=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition) part=$2; shift 2 ;;  --steps) steps=$2; shift 2 ;;  --run) run=$2; shift 2 ;;  --dry-run) dry=1; shift ;;  --eval-only) evo=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
OUT=/soundai/Model/VAPASR/ssa-probe-$run; CK=$OUT/final.pt
MTP=/soundai/users/tskim/VAPKT-data/vapkt-baselines/multitalker-parakeet-v1/multitalker-parakeet-streaming-0.6b-v1.nemo
EV="$run-parakeet-n3d|--model|parakeet:$MTP|--act|n3d|--rc|13;$run-probe-n3d|--model|probe:$CK|--act|n3d|--rc|3;$run-probe-oracle|--model|probe:$CK|--act|oracle|--rc|3;$run-probe-n3d-rc13|--model|probe:$CK|--act|n3d|--rc|13"
tr_cmd="sbatch --partition=$part --export=ALL,MODE=train,OUT=$OUT,ARGS=--steps%20$steps slurm/ssa-probe.sbatch"
if (( dry )); then echo "$tr_cmd"; echo "평가: $EV"; exit 0; fi
dep=""
if (( ! evo )); then
  mkdir -p "$OUT"; o=$(sbatch --partition=$part --export="ALL,MODE=train,OUT=$OUT,ARGS=--steps $steps" slurm/ssa-probe.sbatch); echo "$o"
  dep=$(echo "$o" | grep -oE "[0-9]+$"); echo "학습 job $dep → $OUT"
fi
o=$(sbatch --job-name=SA_EVAL_FullDuplex --partition=$part ${dep:+--dependency=afterok:$dep} --export="ALL,MODE=eval,EVALS=$EV" slurm/ssa-probe.sbatch); echo "$o"
echo "평가 job $(echo "$o" | grep -oE "[0-9]+$") → VAPKT-data/results/ssa-probe-$run-*"
