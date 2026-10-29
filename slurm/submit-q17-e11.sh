#!/usr/bin/env bash
# E11 고정 화자 분리 전문가 학습 제출(2026-10-29, 사용자 "Nemotron 3 Diarization 을 고정 가지로 넣되 하나로 통합할 방법도 고안해놔" — wiki output-n3d-branch-20261029).
#   E9 final 에서 E9 와 같은 레시피(submit-q17-f1.sh --transcribe-only)에 슬롯 활동 입력의 출처만 바꾼다: 헤드(E1c) 예측 → Nemotron 3 Diarization(고정, vapasr/hf/n3d).
#     --n3d <.nemo>(가중치는 체크포인트 밖 — config 에 경로) --n3d-train-chunk 16(학습 스트리밍 청크 1.28 s) --n3d-chunk 4(추론 — 인코더 블록 경계, 지연 더하지 않음).
#     학습은 표본마다 확률 ρ 로 전문가 활동(정답 슬롯 자리로 맞춤), 아니면 흔든 정답(E6 와 같은 ρ 경사). 헤드는 그대로 두고 손실도 그대로 건다(E10 의 탭 임베딩 출처).
#   step 5,000(E9·E10 과 같다 — E9 → E10 은 주입 꼴, E9 → E11 은 활동 출처). --from-e10: E10 final 에서(슬롯 추출기 그대로, --slot-audio --mix-drop 0.5) — 두 변화를 함께.
#   평가 q17-e11: AMI 창 δ10 8 몫 → 회의 통째 δ10 8 몫(전문가가 회의 전체 스트리밍 — AOSC·E7a 헤드 없음).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e11.sh [--dry-run] [--partition P(기본 apex)] [--max-steps N(기본 5000)] [--global-slots P(기본 0.5)] [--from-e10]
#                   [--init DIR] [--n3d PATH] [--after JOBID[,JOBID…](모두 끝난 뒤 — afterany)] [--no-eval].
#   .nemo 가 서버에 있어야 한다(기본 VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo — 사용자 업로드, xcomet 과 같은 곳). 아무것도 지우지 않는다. 산출 /soundai/Model/VAPASR/stage3-q17-e11.
#   low_p_hpc: --partition low_p_hpc — submit-q17-f1.sh 가 선점 파티션 설정을 자동으로 건다(저장 250 step, MAX_RESTART 20, requeue).
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; part=apex; steps=5000; gs=0.5; dep=(); ev=(); init=""; e10=0; n3d=/soundai/users/tskim/VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo; run=q17-e11
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --partition) part=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --global-slots) gs=$2; shift 2 ;;
    --no-eval) ev=(--no-eval); shift ;;  --init) init=$2; shift 2 ;;  --n3d) n3d=$2; shift 2 ;;  --from-e10) e10=1; run=q17-e11b; shift ;;
    --after) [[ ${2:-} =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "$1 은 job id(여럿은 쉼표)" >&2; exit 2; }; dep=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
M=/soundai/Model/VAPASR; E1C=$M/diar-q17-diar-e1c/final/diar.pt
INIT=${init:-$( (( e10 )) && echo $M/stage3-q17-e10/final || echo $M/stage3-q17-e9/final)}
for f in $INIT/config.json $E1C $n3d; do [[ -f $f ]] || { echo "없음: $f" >&2; exit 1; }; done
SA=""; (( e10 )) && SA=" --slot-audio --mix-drop 0.5"
out=$(bash slurm/submit-q17-f1.sh --run $run --init $INIT --partition $part --max-steps $steps --transcribe-only ${dep[@]+"${dep[@]}"} ${ev[@]+"${ev[@]}"} \
      --train-extra "--slot-input --diar-head $E1C --slot-blocks --declare-range$SA --global-slots $gs --freeze-encoder --n3d $n3d --n3d-train-chunk 16 --n3d-chunk 4" \
      --eval-set $run $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! E11 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^\[|^[12] |^sbatch" | sed 's/^/  /' >&2 || true
echo "E11($run) $(echo "$out" | grep -E '^[12] ' | tr '\n' ' ')"
