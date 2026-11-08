#!/usr/bin/env bash
# E16 겹침 데이터 학습 제출(2026-11-08, 사용자 "화자 겹침 데이터를 더 만들어서 학습되도록 하는 게 낫지 않을까?" → v7 판 D "응 시작해." —
#   wiki output-ssa-papers-review-20261106). E13 final 에서 E13 레시피(활동 게이트 12 · 헛 구간 0.1 · 슬롯 블록 번역 · N3D 활동 · 셋 가중 E13 그대로)에
#   겹침 셋 둘을 더하고 인코더는 고정한다(--freeze-encoder). 데이터만으로 겹친 말 받아쓰기가 얼마나 느는지 본다
#   (E15a·E15b 는 인코더 학습이 섞여 혼자 말이 나빠졌고, 겹침 셋 비중도 3.6 % 뿐이었다).
#     sim_en-ovl2(v7 판 D, 2 만 세션 — 말소리 중 겹침 43.9 %, 겹침 구간 90 % 4.64 s, 완전 겹침 0.3) 가중 3.5 + sim_en-ovl(v6) 가중 1.0 → 섞기 비중 약 22 %.
#   평가: q17-e16* 묶음(submit-evals — q17-e15* 와 같은 묶음, 활동 N3D, 창 없이 회의 통째 + 대화 번역)을 1 노드 job --nodes-eval 개로 학습 뒤에 건다.
# 쓰기(스케줄러 셸): [SLOTS=4 SUBSET=eval] bash slurm/submit-q17-e16.sh [--dry-run] [--partition P(기본 hpc)] [--max-steps N(기본 5000)]
#                   [--w-ovl2 W(기본 3.5)] [--w-ovl W(기본 1.0, 0 = v6 없이)] [--run q17-e16*(기본 q17-e16-ovl)] [--init DIR(기본 E13 final)] [--no-eval] [--nodes-eval N(기본 3)]
#   SLOTS·SUBSET 은 평가(submit-evals.sh)에 그대로 간다. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=hpc; steps=5000; init=""; run=q17-e16-ovl; w2=3.5; w1=1.0; ev=1; ne=3
M=/soundai/Model/VAPASR; D=/soundai/users/tskim/VAPKT-data/data
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --init) init=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;  --w-ovl2) w2=$2; shift 2 ;;  --w-ovl) w1=$2; shift 2 ;;  --no-eval) ev=0; shift ;;  --nodes-eval) ne=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e16* ]] || { echo "--run 은 q17-e16 으로 시작(평가 묶음 q17-e16* — N3D 활동)" >&2; exit 2; }
[[ $w2 =~ ^[0-9.]+$ && $w1 =~ ^[0-9.]+$ && $ne =~ ^[1-8]$ ]] || { echo "--w-ovl2·--w-ovl·--nodes-eval: $w2 $w1 $ne" >&2; exit 2; }
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e13/final}
for f in $INIT/config.json $E1C $n3d $D/sim-dialogue-v7-ovl2/sim_en-ovl2.refined.dialogues.jsonl; do [[ -f $f ]] || { echo "없음: $f(v7 빌드가 끝났는지 — sim_v7_build.sh)" >&2; exit 1; }; done
add=(--add-sim "$D/sim-dialogue-v7-ovl2=sim_en-ovl2:$w2"); [[ $w1 != 0 && $w1 != 0.0 ]] && add+=(--add-sim "$D/sim-dialogue-v6-ovl=sim_en-ovl:$w1")
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps --blocks-trans "${add[@]}" --no-eval \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --freeze-encoder" \
      $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E16 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^1 |^sbatch" | sed 's/^/  /' >&2 || true
dep=$(echo "$out" | grep -oE "^1 학습 job [0-9]+" | awk '{print $4}'); [[ -n $dep ]] || { echo "!! 학습 job 번호를 못 읽었다" >&2; exit 1; }
echo "E16($run · 인코더 고정 · sim_en-ovl2 ${w2} + sim_en-ovl ${w1}) 학습 job $dep"
(( ev )) || exit 0
bash slurm/submit-evals.sh "$run" --after-job "$dep" --partition "$part" --nodes "$ne" $dry
