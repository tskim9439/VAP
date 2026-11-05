#!/usr/bin/env bash
# E15b 학습 제출(2026-11-06, 사용자 "꼭 종단간 모델이 아니고 N3D 를 활용해서 Parakeet + N3D 보다 우세할 수 있으면 상관 없을거 같은데" → "2번부터 진행해."
#   — wiki output-agn-reduction-20261104 7·8 절). Parakeet + N3D 와의 격차는 대부분 겹친 말 받아쓰기(둘 겹침 놓침 43 대 23 %)다. Parakeet 은 화자 분리 활동을
#   인코더에 주입해 화자마다 조건 지은 특징을 만든다 — E15b 는 LLM 은 한 스트림 그대로 두고 인코더 위층만 슬롯마다 갈라 돌린다(vapasr/hf/slot_enc).
#   E13 final 에서 E13 레시피(활동 게이트 12 · 헛 구간 0.1 · 슬롯 블록 번역 · 셋 가중 E13 그대로 — E15a 절제의 교훈)에
#     ① 슬롯 조건 인코더(--slot-enc-layer L): 층 L.. 을 (섞인 1 + 슬롯 4) 벌로, 슬롯 벌에 그 슬롯 활동(태그 활동 입력과 같은 — N3D ρ 0.9 + 흔든 정답)을 주입,
#        슬롯 벌 출력 → adapter_slotenc → 그 슬롯 태그 자리. 주입·어댑터 마지막 층 0 초기화(시작 = E13). 위층 L–24 도 학습(--train-encoder-layers L:24, lr 1e-5).
#     ② 겹침 많은 합성 회의 sim_en-ovl(v6 — 말소리 프레임 중 겹침 22.8 %, 재료는 큰 단일 화자 셋 쪽)을 더한다(--add-sim, 기본 가중 0.6 = sim_en-meet 과 같게).
#   인코더는 E13 것(원래 Nemotron — 고정 N3D 로 활동을 받으므로 G2b 인코더가 필요 없다). step 5,000.
#   평가 q17-e15b: AMI 창·회의 통째 δ10(활동 = config 의 N3D — 묶음 q17-e15* 의 ACT=n3d, q17-e15b 는 기본 n3d) + 대화 번역.
# 쓰기(스케줄러 셸): [SLOTS=4 SUBSET=eval] bash slurm/submit-q17-e15b.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--save-every N] [--layer L(기본 16)]
#                   [--hidden H(기본 1024)] [--sim-weight W(기본 0.6, 0 = 새 합성 셋 없이)] [--init DIR(기본 E13 final)] [--n3d PATH] [--run q17-e15b*] [--after JOBID[,…]] [--no-eval]
#   SLOTS·SUBSET 은 학습 뒤 평가(submit-evals.sh)에 그대로 간다(빠른 평가). 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; save=(); dep=(); ev=(); init=""; run=q17-e15b; layer=16; hidden=1024; simw=0.6
M=/soundai/Model/VAPASR; D=/soundai/users/tskim/VAPKT-data/data
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --save-every) save=(--save-every "$2"); shift 2 ;;
    --layer) layer=$2; shift 2 ;;  --hidden) hidden=$2; shift 2 ;;  --sim-weight) simw=$2; shift 2 ;;  --run) run=$2; shift 2 ;;
    --no-eval) ev=(--no-eval); shift ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e15b* ]] || { echo "--run 은 q17-e15b 로 시작(평가 묶음 q17-e15* — N3D 활동)" >&2; exit 2; }
[[ $layer =~ ^[0-9]+$ && $hidden =~ ^[1-9][0-9]*$ && $simw =~ ^[0-9.]+$ ]] || { echo "--layer·--hidden·--sim-weight: $layer $hidden $simw" >&2; exit 2; }
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e13/final}; OVL=$D/sim-dialogue-v6-ovl
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
add=(); [[ $simw != 0 && $simw != 0.0 ]] && add=(--add-sim "$OVL=sim_en-ovl:$simw")
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps ${save[@]+"${save[@]}"} --blocks-trans ${add[@]+"${add[@]}"} \
      ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --slot-enc-layer $layer --slot-enc-hidden $hidden --train-encoder-layers $layer:24" \
      --eval-set $run $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E15b 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E15b($run · 슬롯 조건 인코더 층 $layer.. 은닉 $hidden · 위층 $layer:24 학습 · sim_en-ovl ${simw}) $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
