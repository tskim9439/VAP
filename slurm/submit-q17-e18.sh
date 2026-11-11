#!/usr/bin/env bash
# E18 — 인코더와 LLM 을 함께 학습(2026-11-11, 사용자 "N3D 는 고정 시키고 Encoder 와 LLM 모두 학습시키고, 겹침을 늘려서 학습한 실험은 없지??" → "응 E18 준비해줘. low_p_hpc 로 제출할거야").
#   E17(인코더 고정 · 겹침 셋 75 %)과 데이터·시작점(E16 final)·step·학습률 일정을 같게 두고 **인코더 고정만 푼다**. N3D 는 그대로 고정이다.
#     인코더는 전체(서브샘플링 + 24 층)를 학습률 1e-5 로 학습한다(--lr-encoder; E15a 는 위 12 층만 1e-5). --enc-layers LO:HI 를 주면 그 층만(서브샘플링·다른 층 고정).
#     E17 이 인코더 고정 쪽 대조다: 같은 step(checkpoint-5000)에서 E17 은 표준 cp 27.02 / 31.42, 둘 겹침 놓침 43.4 / 49.1 %.
#   추론 구조는 그대로다(인코더 한 번 + LLM 한 스트림). 인코더 가중은 체크포인트에 함께 저장된다.
#   평가: q17-e18* 묶음(submit-evals — q17-e17* 와 같은 묶음, 활동 N3D, 회의 통째 + 대화 번역)을 1 노드 job --nodes-eval 개로 학습 뒤에 건다. --eval-ck N 이면 checkpoint-N 도.
#   MAX_RESTART 기본 60. low_p_hpc 는 선점이 잦다 — E17 은 22 시간에 재시작 41 번, 학습까지 간 것 5 번이었다(재시작마다 첫 step 까지 약 45 분).
# 쓰기(스케줄러 셸): [SLOTS=4 SUBSET=eval] bash slurm/submit-q17-e18.sh [--dry-run] [--partition P(기본 hpc)] [--max-steps N(기본 10000)] [--eval-ck N(기본 절반)]
#                   [--lr-encoder LR(기본 1e-5)] [--enc-layers LO:HI(기본 전체)] [--w-ovl3 W(기본 24)] [--w-ovl2 W(기본 18)] [--w-ovl W(기본 6)]
#                   [--run q17-e18*(기본 q17-e18-enc)] [--init DIR(기본 E16 final)] [--no-eval] [--nodes-eval N(기본 3)]
#   SLOTS·SUBSET 은 평가(submit-evals.sh)에 그대로 간다. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=hpc; steps=10000; init=""; run=q17-e18-enc; w3=24; w2=18; w1=6; ev=1; ne=3; ck=""; lre=1e-5; lay=""
export MAX_RESTART=${MAX_RESTART:-60}
M=/soundai/Model/VAPASR; D=/soundai/users/tskim/VAPKT-data/data
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --init) init=$2; shift 2 ;;
    --run) run=$2; shift 2 ;;  --w-ovl3) w3=$2; shift 2 ;;  --w-ovl2) w2=$2; shift 2 ;;  --w-ovl) w1=$2; shift 2 ;;  --no-eval) ev=0; shift ;;  --nodes-eval) ne=$2; shift 2 ;;  --eval-ck) ck=$2; shift 2 ;;  --lr-encoder) lre=$2; shift 2 ;;  --enc-layers) lay=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e18* ]] || { echo "--run 은 q17-e18 로 시작(평가 묶음 q17-e18* — N3D 활동)" >&2; exit 2; }
[[ $lre =~ ^[0-9.e-]+$ && ( -z $lay || $lay =~ ^[0-9]+:[0-9]+$ ) ]] || { echo "--lr-encoder·--enc-layers: $lre $lay" >&2; exit 2; }
ck=${ck:-$((steps / 2))}; [[ $ck =~ ^[0-9]+$ ]] && (( ck == 0 || (ck < steps && ck % 250 == 0) )) || { echo "--eval-ck 은 0 또는 max-steps 보다 작은 250 의 배수(저장 간격): $ck" >&2; exit 2; }
[[ $w3 =~ ^[0-9.]+$ && $w2 =~ ^[0-9.]+$ && $w1 =~ ^[0-9.]+$ && $ne =~ ^[1-8]$ ]] || { echo "--w-ovl3·--w-ovl2·--w-ovl·--nodes-eval: $w3 $w2 $w1 $ne" >&2; exit 2; }
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e16-ovl/final}
for f in $INIT/config.json $E1C $n3d $D/sim-dialogue-v8-ovl3/sim_en-ovl3.refined.dialogues.jsonl $D/sim-dialogue-v7-ovl2/sim_en-ovl2.refined.dialogues.jsonl; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
add=(--add-sim "$D/sim-dialogue-v8-ovl3=sim_en-ovl3:$w3" --add-sim "$D/sim-dialogue-v7-ovl2=sim_en-ovl2:$w2"); [[ $w1 != 0 && $w1 != 0.0 ]] && add+=(--add-sim "$D/sim-dialogue-v6-ovl=sim_en-ovl:$w1")
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps --blocks-trans "${add[@]}" --no-eval \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --lr-encoder $lre${lay:+ --train-encoder-layers $lay}" \
      $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E18 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^1 |^sbatch" | sed 's/^/  /' >&2 || true
dep=$(echo "$out" | grep -oE "^1 학습 job [0-9]+" | awk '{print $4}'); [[ -n $dep ]] || { echo "!! 학습 job 번호를 못 읽었다" >&2; exit 1; }
echo "E18($run · 인코더 ${lay:-전체} 학습 lr $lre + LLM · sim_en-ovl3 ${w3} + sim_en-ovl2 ${w2} + sim_en-ovl ${w1} · ${steps} step · MAX_RESTART $MAX_RESTART) 학습 job $dep"
(( ev )) || exit 0
bash slurm/submit-evals.sh "$run" --after-job "$dep" --partition "$part" --nodes "$ne" $dry
(( ck )) || exit 0
TARGET=checkpoint-$ck bash slurm/submit-evals.sh "$run" --after-job "$dep" --partition "$part" --nodes "$ne" $dry   # 과적합 확인(같은 묶음, 산출 -ck$ck)
