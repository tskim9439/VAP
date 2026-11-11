#!/usr/bin/env bash
# 보조 헤드 진단 제출(2026-11-11, 사용자 "보조 헤드 진단부터 구현해줘" — experiments/ssa_head_probe.py).
#   학습 job 하나(노드 1 · 8 GPU): 인코더는 고정하고 한 번만 돌린다. 그 출력 위에 화자 조건 헤드(+ 원 모델 디코더·joint)만 학습한다.
#   평가 job 하나(학습 성공 뒤, P1 과 같은 AMI eval 30 s 조각 평가): head-n3d(N3D 활동)·head-oracle(정답 활동). Parakeet·P1 판은 P1 것을 쓴다.
#   쓰기(스케줄러 셸): bash slurm/submit-ssa-head.sh [--partition P(기본 apex)] [--run 이름(기본 h1)] [--steps N(기본 6000)] [--layers N(기본 4)] [--ctx L,R(기본 64,2)]
#                     [--save-every N(기본 500)] [--freeze-dec] [--dry-run] [--eval-only]
#   low_p_hpc 처럼 선점되는 파티션에서는 --save-every 250 을 권한다(선점되면 마지막 저장부터 이어 간다 — 250 step ≈ 5–6 분, 시작 준비 약 7 분).
#   산출: 학습 /soundai/Model/VAPASR/ssa-head-<이름>/(step-N.pt·final.pt), 평가 VAPKT-data/results/ssa-probe-<이름>-head-{n3d,oracle}. 아무것도 지우지 않는다.
#   보고(서버 호스트): VARIANTS="head-n3d head-oracle" bash raw/sources/experiments/2026-11-09-ssa-probe/ssa_probe_report.sh <이름>
set -euo pipefail
cd "$(dirname "$0")/.."
part=apex; steps=6000; run=h1; dry=0; evo=0; layers=4; ctx=64,2; fz=""; save=500
while [[ $# -gt 0 ]]; do
  case $1 in
    --partition) part=$2; shift 2 ;;  --steps) steps=$2; shift 2 ;;  --run) run=$2; shift 2 ;;  --layers) layers=$2; shift 2 ;;  --ctx) ctx=$2; shift 2 ;;
    --freeze-dec) fz=" --freeze-dec"; shift ;;  --dry-run) dry=1; shift ;;  --eval-only) evo=1; shift ;;  --save-every) save=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $steps =~ ^[1-9][0-9]*$ && $layers =~ ^[1-9]$ && $ctx =~ ^[0-9]+,[0-9]+$ && $run =~ ^[a-z0-9-]+$ && $save =~ ^[1-9][0-9]*$ ]] || { echo "--steps·--layers·--ctx·--run·--save-every: $steps $layers $ctx $run $save" >&2; exit 2; }
OUT=/soundai/Model/VAPASR/ssa-head-$run; CK=$OUT/final.pt
EV="$run-head-n3d|--model|head:$CK|--act|n3d|--rc|3;$run-head-oracle|--model|head:$CK|--act|oracle|--rc|3"
TA="--steps $steps --head-layers $layers --head-ctx $ctx --save-every $save$fz"; TAE=${TA//,/%2C}
tr_cmd="sbatch --partition=$part --export=ALL,MODE=train,OUT=$OUT,ARGS=$TAE slurm/ssa-head-probe.sbatch"
if (( dry )); then echo "$tr_cmd"; echo "평가: $EV"; exit 0; fi
dep=""
if (( ! evo )); then
  mkdir -p "$OUT"; o=$(sbatch --partition=$part --export="ALL,MODE=train,OUT=$OUT,ARGS=$TAE" slurm/ssa-head-probe.sbatch); echo "$o"
  dep=$(echo "$o" | grep -oE "[0-9]+$"); echo "학습 job $dep → $OUT"
fi
o=$(sbatch --job-name=SA_EVAL_FullDuplex --partition=$part ${dep:+--dependency=afterok:$dep} --export="ALL,MODE=eval,EVALS=$EV" slurm/ssa-head-probe.sbatch); echo "$o"
echo "평가 job $(echo "$o" | grep -oE "[0-9]+$") → VAPKT-data/results/ssa-probe-$run-head-*"
