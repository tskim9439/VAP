#!/usr/bin/env bash
# Qwen3-ASR 1.7B 체인(2026-10-22, 사용자 "Qwen asr 1.7B 를 이용해서도 학습시키고 싶어"): 0.6B 와 같은 단계·레시피를 1.7B(/soundai/Model/Qwen3-ASR-1.7B)로.
# 단계마다 앞 단계가 성공해야 시작한다(sbatch afterok). 한 번 실행으로 모두 제출하고 job 번호를 출력한다(제출은 사용자 — 스케줄러 셸에서).
#   1 q17-s1m3-d10         Stage 1 영·한·독 δ4–10(semcommit, q17-s1m-r3 에서 1 epoch) — 2026-10-08 run 의 checkpoint-1000 에서 이어 한다   약 3 h
#   2 q17-s2-ast           단일 화자 번역(ast-v1, 번역 모드 0.5) 2 epoch — init q17-s1m3-d10/final                                    약 10 h
#   3 q17-s3-ast           번역 모드 0.8 + 영어 음량 증강 2 epoch — init q17-s2-ast/final                                              약 10 h
#   4 q17-s4-ast           + CV4 영어 낭독·번역 2 epoch — init q17-s3-ast/final                                                        약 10 h
#   5 q17-st3-p11-revive   대화 Stage 3(0.6B p11 명령 그대로: 같은 언어 대화·합성·replay·revive v1, 4,000 step) — init q17-s4-ast/final   약 2.7 h
#   6 q17-v3-p1-tel-xlw    V3 R1x(전화 합성 셋 + 교대 가중은 교차 언어 셋만, 2,500 step) — init q17-st3-p11-revive/checkpoint-3000       약 2.2 h
#   (시간은 apex 2 노드, 0.6B 단계 시간 × 1.7B 속도비 약 2.4 로 어림 — 합 약 38 h)
#   0.6B 와 다른 점: Stage 1 은 0.6B(q06-s0 에서 영·한·독 3 epoch)와 달리 1.7B 영·한 3 epoch(q17-s1m-r3) 뒤 영·한·독 1 epoch 이다(2026-10-08 계획).
#   5 단계 체크포인트는 0.6B 처럼 checkpoint-3000 을 6 단계 init 으로 쓴다(0.6B 는 평가로 ck3000 을 골랐다 — 1.7B 는 평가 없이 같은 위치).
# 쓰기(스케줄러 셸, /soundai/users/tskim/VAPKT):
#   bash slurm/submit-q17-chain.sh --dry-run            # sbatch 없이 단계별 명령만(앞 단계 job 은 0 으로 둔다)
#   bash slurm/submit-q17-chain.sh                      # 1–6 단계 제출
#   bash slurm/submit-q17-chain.sh --from 5 [--after-job JOBID]   # 5 단계부터(앞 단계가 이미 끝났거나 JOBID 가 끝난 뒤)
#   --exclude N,…: 모든 단계 sbatch 에서 더 뺄 노드(기본 hpc-50,hpc-51,hpc-101,hpc-102 — V3 R1·R2 때와 같다, 'none' 이면 더 빼지 않는다).
# 단계마다 MAX_RESTART(기본 5). 앞 단계가 실패하면 뒤 단계는 DependencyNeverSatisfied 로 남는다 — 취소는 사용자가 한다. 경로 변수는 이 스크립트 안에서만 쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."
from=1; dep=""; dry=""; exclude=hpc-50,hpc-51,hpc-101,hpc-102
while [[ $# -gt 0 ]]; do
  case $1 in
    --from) from=$2; shift 2 ;;  --after-job) dep=$2; shift 2 ;;  --dry-run) dry=--dry-run; shift ;;  --exclude) exclude=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $from =~ ^[1-6]$ ]] || { echo "--from 은 1–6" >&2; exit 2; }
[[ -z $dep || $dep =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
[[ $exclude == none ]] && exclude=""
export MAX_RESTART=${MAX_RESTART:-5}
D=/soundai/users/tskim/VAPKT-data/data; M=/soundai/Model/VAPASR
EXC=(); [[ -n $exclude ]] && EXC=(--exclude "$exclude")
P11=(--partition apex --nodes 2 --backchannel 1.0,3 --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,rep_ko:1.2,rep_en:1.2,rep_de:1.2
     --lr 4e-5 --max-steps 4000 --spk-act-weight 0.5 --sim-data $D/sim-dialogue-v2 --revive-dir $D/phase2/revive-asr-v1)
R1X=(--partition apex --nodes 2 --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $D/phase2/revive-asr-v2 --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16
     --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 --sim-trans-frac 0.8 --trans-lead-space --phone-aug 0.3
     --sim-data $D/sim-dialogue-v2,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1
     --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2
     --set-phone sim_en-de-tel:0 --change-tag-weight 3 --hold-tag-weight 1.5 --eot-drop-synth 1.0 --eot-drop-sets xl --lr 4e-5 --max-steps 2500 --decision-weight-sets xl)
stage() {                                                                                       # 단계 번호, 설명, 명령… → 제출하고 dep 에 job id
  local n=$1 what=$2; shift 2; [[ $n -lt $from ]] && return 0
  local a=(); [[ -n $dep ]] && a=(--afterok "$dep")
  echo "[$(date '+%F %T')] 단계 $n $what${dep:+ (job $dep 성공 뒤)}" >&2
  local out; out=$("$@" ${EXC[@]+"${EXC[@]}"} ${a[@]+"${a[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 단계 $n 제출 실패 — 뒤 단계는 내지 않는다" >&2; exit 1; }
  echo "$out" | grep -E "^sbatch|Submitted|이어서|만들 예정|init 없음|거부" >&2
  dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)
  [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true                                          # dry-run: 대화 단계 스크립트는 job 줄을 내지 않는다 → 0
  [[ -n $dep ]] || { echo "!! 단계 $n job 번호를 못 읽었다" >&2; exit 1; }
  echo "단계 $n job $dep"
}
stage 1 "Stage 1 q17-s1m3-d10(이어 하기)" bash slurm/submit-q17-s1m3-d10.sh
stage 2 "Stage 2 q17-s2-ast" bash slurm/submit-q17-s2-ast.sh
stage 3 "Stage 3 q17-s3-ast" bash slurm/submit-q17-s3-ast.sh
stage 4 "Stage 4 q17-s4-ast" bash slurm/submit-q17-s4-ast.sh
stage 5 "대화 Stage 3 q17-st3-p11-revive" bash slurm/submit-stage3-train.sh --run q17-st3-p11-revive --init $M/semcommit-q17-s4-ast/final "${P11[@]}"
stage 6 "V3 R1x q17-v3-p1-tel-xlw" bash slurm/submit-stage3-train.sh --run q17-v3-p1-tel-xlw --init $M/stage3-q17-st3-p11-revive/checkpoint-3000 "${R1X[@]}"
echo "[$(date '+%F %T')] 체인 제출 끝${dry:+(dry-run — 아무것도 제출하지 않았다)}" >&2
