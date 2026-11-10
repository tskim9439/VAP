#!/usr/bin/env bash
# E17 — 본 모델을 P2 레시피로(2026-11-10, 사용자 "우리 모델도 학습 스텝 더 늘려서 학습될 수 있도록 해줘" → "P2 와 같은 레시피로 진행했으면 좋겠는데").
#   P2(화자 조건 인코더 검증)처럼 겹침 비중을 크게(약 75 %) 하고, 다른 대화를 통째로 덧섞은 셋을 쓰고, 학습량을 크게(30,000 step) 한다.
#   E16 final(E13 레시피 · 인코더 고정 · 겹침 22 %, 5,000 step)에서 이어 간다. 레시피(활동 게이트·슬롯 블록 번역·N3D 활동·인코더 고정)는 E16 그대로다.
#     sim_en-ovl3(v8 — v7 2 인 + sim_en 2 인을 통째로 겹친 4 인 대화 18,156 개, 520 h, experiments/sim_overlay_dialogues.py) 가중 24
#     + sim_en-ovl2(v7) 18 + sim_en-ovl(v6) 6 → 섞기 비중 약 75 %(나머지 셋 가중 합 16). 30,000 step × 25 % ≈ E13 의 5,000 step 보다 다른 셋을 더 본다.
#   평가: q17-e17* 묶음(submit-evals — q17-e15* 와 같은 묶음, 활동 N3D, 창 없이 회의 통째 + 대화 번역)을 1 노드 job --nodes-eval 개로 학습 뒤에 건다.
#   MAX_RESTART 기본 60(low_p_hpc 선점이 잦다 — E16 은 5,000 step 에 재시작 12 번).
#   2026-11-10(사용자 "2 번으로 진행하자 … 과적합이 되지 않고 효과가 나타난다면 1번"): 기본 10,000 step(30,000 step 이면 v8 을 약 75 번 다시 본다 → 약 25 번).
#     같은 --run 으로 다시 내면 out-dir 의 마지막 완성 체크포인트에서 이어 간다(재개 지문 config.stage3 에 max_steps 가 없다; cosine 은 새 max_steps 로 다시 그려진다).
#     과적합 확인: --eval-ck N(기본 max-steps 의 절반, 0 = 끔)이면 checkpoint-N 평가(TARGET=checkpoint-N, 산출 -ck<N>)도 학습 job 뒤에 같이 건다.
# 쓰기(스케줄러 셸): [SLOTS=4 SUBSET=eval] bash slurm/submit-q17-e17.sh [--dry-run] [--partition P(기본 hpc)] [--max-steps N(기본 10000)] [--eval-ck N(기본 절반)]
#                   [--w-ovl3 W(기본 24)] [--w-ovl2 W(기본 18)] [--w-ovl W(기본 6)] [--run q17-e17*(기본 q17-e17-p2r)] [--init DIR(기본 E16 final)] [--no-eval] [--nodes-eval N(기본 3)]
#   SLOTS·SUBSET 은 평가(submit-evals.sh)에 그대로 간다. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=hpc; steps=10000; init=""; run=q17-e17-p2r; w3=24; w2=18; w1=6; ev=1; ne=3; ck=""
export MAX_RESTART=${MAX_RESTART:-60}
M=/soundai/Model/VAPASR; D=/soundai/users/tskim/VAPKT-data/data
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --init) init=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;  --w-ovl3) w3=$2; shift 2 ;;  --w-ovl2) w2=$2; shift 2 ;;  --w-ovl) w1=$2; shift 2 ;;  --no-eval) ev=0; shift ;;  --nodes-eval) ne=$2; shift 2 ;;  --eval-ck) ck=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e17* ]] || { echo "--run 은 q17-e17 로 시작(평가 묶음 q17-e17* — N3D 활동)" >&2; exit 2; }
ck=${ck:-$((steps / 2))}; [[ $ck =~ ^[0-9]+$ ]] && (( ck == 0 || (ck < steps && ck % 250 == 0) )) || { echo "--eval-ck 은 0 또는 max-steps 보다 작은 250 의 배수(저장 간격): $ck" >&2; exit 2; }
[[ $w3 =~ ^[0-9.]+$ && $w2 =~ ^[0-9.]+$ && $w1 =~ ^[0-9.]+$ && $ne =~ ^[1-8]$ ]] || { echo "--w-ovl3·--w-ovl2·--w-ovl·--nodes-eval: $w3 $w2 $w1 $ne" >&2; exit 2; }
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e16-ovl/final}
for f in $INIT/config.json $E1C $n3d $D/sim-dialogue-v8-ovl3/sim_en-ovl3.refined.dialogues.jsonl $D/sim-dialogue-v7-ovl2/sim_en-ovl2.refined.dialogues.jsonl; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
add=(--add-sim "$D/sim-dialogue-v8-ovl3=sim_en-ovl3:$w3" --add-sim "$D/sim-dialogue-v7-ovl2=sim_en-ovl2:$w2"); [[ $w1 != 0 && $w1 != 0.0 ]] && add+=(--add-sim "$D/sim-dialogue-v6-ovl=sim_en-ovl:$w1")
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps --blocks-trans "${add[@]}" --no-eval \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --freeze-encoder" \
      $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E17 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^1 |^sbatch" | sed 's/^/  /' >&2 || true
dep=$(echo "$out" | grep -oE "^1 학습 job [0-9]+" | awk '{print $4}'); [[ -n $dep ]] || { echo "!! 학습 job 번호를 못 읽었다" >&2; exit 1; }
echo "E17($run · P2 레시피 · 인코더 고정 · sim_en-ovl3 ${w3} + sim_en-ovl2 ${w2} + sim_en-ovl ${w1} · ${steps} step · MAX_RESTART $MAX_RESTART) 학습 job $dep"
(( ev )) || exit 0
bash slurm/submit-evals.sh "$run" --after-job "$dep" --partition "$part" --nodes "$ne" $dry
(( ck )) || exit 0
TARGET=checkpoint-$ck bash slurm/submit-evals.sh "$run" --after-job "$dep" --partition "$part" --nodes "$ne" $dry   # 과적합 확인(같은 묶음, 산출 -ck$ck)
