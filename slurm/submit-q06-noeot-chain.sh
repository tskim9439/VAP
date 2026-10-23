#!/usr/bin/env bash
# 0.6B <EOT> 없는 대화 학습 시험(2026-10-23, 사용자 — 턴 예측은 주 목표가 아니고 화자 귀속이 중요, 모델이 "<EOT> 를 낸 뒤에 교대한다" 를 배우지 않을까 걱정
#   → "진행하도록 해"). 대화 Stage 3(p11)·Stage 4(V3 R1x)를 0.6B 원래 명령 그대로 --no-eot 만 더해 다시 학습하고, 끝나면 SLURM 평가 노드로 R1x 와 같은 평가를 잰다.
#   1 st3-p11-revive-noeot   init q06-s4-ast/final, p11 명령(4,000 step) + --no-eot                                   0.6B 약 1.1 h + 시작
#   2 v3-p1-tel-xlw-noeot    init 1 의 checkpoint-3000(0.6B 처럼 평가 없이 같은 위치), R1x 명령(2,500 step) + --no-eot       약 1.0 h + 시작
#   3 평가 noeot-v3(slurm/submit-evals.sh): 2 의 ck1500·ck2000·final·avg3 × 9 작업 + final·avg3 AMI, 1 의 ck3000 같은 언어 평가
#   비교 대상: R1x(v3-p1-tel-xlw), 같은 기준(dev 귀속으로 고르고 TAXI·같은 언어 가드·AMI·번역 BLEU). <EOT> 를 뺀 쪽이 이기면 1.7B 체인의 대화 두 단계
#   (job 94813·94814 — 사용자 취소)를 --no-eot 로 다시 낸다(submit-q17-chain.sh --from 5 와 같은 명령 + --no-eot).
#   P11·R1X 인자는 slurm/submit-q17-chain.sh 와 같다(0.6B 원래 p11·R1x 제출 명령에서 init·run·의존성·제외 노드만 다르다).
# 쓰기(스케줄러 셸, /soundai/users/tskim/VAPKT):
#   bash slurm/submit-q06-noeot-chain.sh --dry-run          # sbatch 없이 단계별 명령만(앞 단계 job 은 0)
#   bash slurm/submit-q06-noeot-chain.sh                    # 학습 두 단계 + 평가 노드 제출
#   --exclude N,…: 더 뺄 노드(기본 hpc-50,hpc-51,hpc-101,hpc-102 — 1.7B 체인과 같다, 'none' 이면 더 빼지 않는다). MAX_RESTART 기본 5. 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; exclude=hpc-50,hpc-51,hpc-101,hpc-102
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --exclude) exclude=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
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
dep=""
stage() {                                                                                       # 단계 이름, 명령… → 제출하고 dep 에 job id
  local what=$1; shift; local a=(); [[ -n $dep ]] && a=(--afterok "$dep")
  echo "[$(date '+%F %T')] $what${dep:+ (job $dep 성공 뒤)}" >&2
  local out; out=$("$@" ${EXC[@]+"${EXC[@]}"} ${a[@]+"${a[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 제출 실패 — 뒤 단계는 내지 않는다" >&2; exit 1; }
  echo "$out" | grep -E "^sbatch|Submitted|만들 예정|init 없음|거부" >&2 || true
  dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true)
  [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true
  [[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
  echo "$what job $dep"
}
stage "1 대화 Stage 3 st3-p11-revive-noeot" bash slurm/submit-stage3-train.sh --run st3-p11-revive-noeot --init $M/semcommit-q06-s4-ast/final "${P11[@]}" --no-eot
stage "2 V3 R1x st4 v3-p1-tel-xlw-noeot" bash slurm/submit-stage3-train.sh --run v3-p1-tel-xlw-noeot --init $M/stage3-st3-p11-revive-noeot/checkpoint-3000 "${R1X[@]}" --no-eot
echo "[$(date '+%F %T')] 3 평가 noeot-v3 (job $dep 성공 뒤)" >&2
out=$(bash slurm/submit-evals.sh noeot-v3 --after-job "$dep" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^목록|^sbatch" >&2 || true
echo "3 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
echo "[$(date '+%F %T')] 제출 끝${dry:+(dry-run — 아무것도 제출하지 않았다)}" >&2
