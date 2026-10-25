#!/usr/bin/env bash
# E2 대상 화자 모드 1.7B 학습 제출(2026-10-25, 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해" — wiki output-diarization-integration-20261025):
# 1.7B 회의 + 4 슬롯 final 에서 --target-spk(창마다 말하는 화자 하나만 한 슬롯 정답, 그 화자 80 ms 활동을 인코더 pre-encode 에 주입 — SSA arXiv:2506.22646)로 학습하고,
# 끝나면 SLURM 평가 노드로 AMI 정답 활동 평가(v3_eval_task.sh amitgt — 창 화자마다 정답 활동 주입·따로 디코드, 8 몫)를 잇는다. 상한 시험이다(실제 화자 분리는 E3).
#   학습 run q17-e2-tgt: init stage3-q17-v3-p1-tel-xlw-mt4/final, 대화 10 셋(회의 AMI·ICSI·NOTSOFAR 포함, split v4) + 합성(3–4 인 v3-multi·교차 언어·전화, 번역 없이 받아쓰기만),
#     replay 없음, 화자 슬롯 헤드 끔, 주입 FFN 은닉 512(마지막 Linear 0 초기화 → 시작 = init 모델), lr 4e-5, 4,000 step, 2 노드.
#   비교: E0(1.7B 회의 + 4 슬롯 avg3 를 정답 구간으로 다시 귀속, SDM eval δ10 cp 39.3), 같은 모델의 창 평가(53.5), 문헌 오라클 DiCoW 16.3(세션·meeteval).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-e2-tgt.sh [--dry-run] [--after-job JOBID] [--partition P(기본 low_p_hpc)] [--nodes N(기본 2)] [--max-steps N(기본 4000)]
#   [--exclude N,…|none] [--no-eval]. 아무것도 지우지 않는다.
#   파티션(2026-10-25 사용자 "low_p_hpc 로 제출할 수 있도록 해줘"): 학습·평가 노드 모두 P. 선점이 잦아 MAX_RESTART 기본 20(apex 면 5) — 학습은 선점 신호에
#   저장하고 requeue 뒤 마지막 checkpoint 에서, 평가 노드(--requeue)는 끝난 창(원출력 행)을 건너뛰고 이어 한다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; exclude=hpc-50,hpc-51,hpc-101,hpc-102; after=""; part=low_p_hpc; nodes=2; steps=4000; ev=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --exclude) exclude=$2; shift 2 ;;  --after-job) after=$2; shift 2 ;;  --partition) part=$2; shift 2 ;;
    --nodes) nodes=$2; shift 2 ;;  --max-steps) steps=$2; shift 2 ;;  --no-eval) ev=0; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $exclude == none ]] && exclude=""
[[ -z $after || $after =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
export MAX_RESTART=${MAX_RESTART:-$([[ $part == apex ]] && echo 5 || echo 20)}
D=/soundai/users/tskim/VAPKT-data/data; M=/soundai/Model/VAPASR
EXC=(); [[ -n $exclude ]] && EXC=(--exclude "$exclude")
ARGS=(--run q17-e2-tgt --init $M/stage3-q17-v3-p1-tel-xlw-mt4/final ${after:+--afterok $after} --partition $part --nodes $nodes
      --backchannel 1.0,3 --spk-act-weight 0 --revive-dir $D/phase2/revive-asr-v2 --phone-aug 0.3
      --sim-data $D/sim-dialogue-v3-multi,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1
      --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.5,sim_ko:0.8,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:0.5,sim_en-ko:0.5
      --set-phone sim_en-de-tel:0 --lr 4e-5 --max-steps $steps --split v4 --max-speakers 4
      --dlg-corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar --extra "--target-spk --inject-hidden 512")
echo "[$(date '+%F %T')] E2 q17-e2-tgt(대상 화자 모드, 정답 활동 주입${after:+, job $after 성공 뒤}, $part $nodes 노드, $steps step, MAX_RESTART $MAX_RESTART)" >&2
out=$(bash slurm/submit-stage3-train.sh "${ARGS[@]}" ${EXC[@]+"${EXC[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^sbatch|Submitted|만들 예정|init 없음|거부" >&2 || true
dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true); [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true
[[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
echo "1 학습 job $dep"
(( ev )) || exit 0
echo "[$(date '+%F %T')] 2 평가 q17-e2-tgt (job $dep 성공 뒤)" >&2
out=$(bash slurm/submit-evals.sh q17-e2-tgt --after-job "$dep" --partition "$part" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^목록|^sbatch" >&2 || true
echo "2 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
