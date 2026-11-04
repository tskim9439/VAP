#!/usr/bin/env bash
# E15a 학습 제출(2026-11-04, 사용자 "그럼 agn 을 없애려면 어떻게 개선할 지 고민해봐. 종단간 모델구조를 사용할거야." → "E15a 부터 구현 진행해줘"
#   — wiki output-agn-reduction-20261104). 받아쓰기 오류(agn)를 겨냥한다: 놓친 단어의 절반은 혼자 말할 때, 절반은 보통 발화끼리 겹친 구간이다.
#   E13 final 에서 E13 레시피(① 활동 게이트 12 · 헛 구간 0.1, ② 슬롯 블록 번역 — submit-q17-e13.sh)에 둘을 더한다.
#     ③ 인코더 위층 공동 학습: 시작 인코더를 G2b graft.pt 의 enc(아래 12 층을 화자 분리와 함께 학습 — 학생은 8 층을 탭한다)로 바꾸고(--encoder-state),
#        13–24 층(--train-encoder-layers 12:24)만 LLM 손실로 학습한다(학습률 1e-5 — submit-stage3-train.sh 기본). 아래 층·서브샘플링은 고정·eval 이라 학생 활동은 G2b 그대로다.
#     ④ 겹침 많은 셋 가중(--weights-set): 실제 회의 근접(dlg_mt)·원거리(dlg_far) 1 → 2, 영어 회의형 합성(sim_en-meet) 0.6 → 1.2.
#   학습 활동은 E13 과 같다(고정 N3D ρ 0.9 + 흔든 정답 — 학습에서만 교사로 쓴다). 평가는 종단간이다: 활동을 G2b 학생으로(묶음 q17-e15* — GRAFT_EXPERT).
#   step 5,000. 평가 q17-e15a: AMI 창 δ10 8 몫·회의 통째 δ10 8 몫(G2b 학생 활동, 모델 인코더) + 대화 번역(dlg) — 한 노드.
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e15a.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--save-every N] [--layers LO:HI(기본 12:24)]
#                   [--weights-set 셋:가중,…|none(기본 dlg_mt:2,dlg_far:2,sim_en-meet:1.2)] [--graft PATH(기본 G2b final graft.pt)] [--init DIR(기본 E13 final)] [--n3d PATH]
#                   [--run q17-e15*] [--after JOBID[,…]] [--no-eval].
#   low_p_hpc·hpc: submit-q17-f1.sh 가 선점 파티션 설정을 자동으로 건다(저장 250 step, MAX_RESTART 20, requeue). 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; save=(); dep=(); ev=(); init=""; run=q17-e15a; layers=12:24; wset=dlg_mt:2,dlg_far:2,sim_en-meet:1.2
M=/soundai/Model/VAPASR; graft=$M/graft-q17-graft-g2b/final/graft.pt
n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --save-every) save=(--save-every "$2"); shift 2 ;;
    --layers) layers=$2; shift 2 ;;  --weights-set) wset=$2; shift 2 ;;  --graft) graft=$2; shift 2 ;;  --run) run=$2; shift 2 ;;
    --no-eval) ev=(--no-eval); shift ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $run == q17-e15* ]] || { echo "--run 은 q17-e15 로 시작(평가 묶음 q17-e15*)" >&2; exit 2; }
[[ $layers =~ ^[0-9]+:[0-9]+$ ]] || { echo "--layers LO:HI: $layers" >&2; exit 2; }
[[ $wset == none ]] && wset=""
E1C=$M/diar-q17-diar-e1c/final/diar.pt; INIT=${init:-$M/stage3-q17-e13/final}
for f in $INIT/config.json $E1C $n3d $graft; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps ${save[@]+"${save[@]}"} --blocks-trans ${wset:+--weights-set $wset} \
      ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range --global-slots 0.5 --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4 --slot-gate 12 --gate-fa 0.1 --encoder-state $graft --train-encoder-layers $layers" \
      --eval-set $run $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E15a 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E15a($run · 인코더 층 $layers 학습 ← $(basename $(dirname $(dirname $graft))) · 가중 ${wset:-E13 그대로}) $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
