#!/usr/bin/env bash
# F1 데이터 확장 시험(2026-10-26, 사용자 "모두 진행하는 데, 모두 진행하고 나서 E5 E6 학습 진행하자" — wiki task-data-expansion-20261026):
# 1.7B 회의 + 4 슬롯(q17-v3-p1-tel-xlw-mt4, submit-q17-chain.sh --dlg mt4 의 6 단계)과 같은 init(5 단계 ck3000)·레시피·step·학습률에 새 데이터만 더한다
# — 같은 계산량에서 데이터 효과를 잰다. 더하는 것:
#   원거리 회의 녹음(dlg_far: ami-far·icsi-far·notsofar-far·chime6-far — p2_build_far, 회의 번역 타깃은 원본 id 로 쓴다), CHiME-6 근접(chime6 → dlg_mt),
#   Switchboard(swbd → dlg_en), 한국어 회의 NIA23(nia23mt → dlg_mtko), 회의형 합성 v4(sim_<l>-meet), 화자별 원거리 증강(--far-aug 0.3, 근접 대화·회의 셋).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-f1.sh [--dry-run] [--allow-missing] [--partition P(기본 apex)] [--nodes N(기본 2)] [--max-steps N(기본 2500)]
#   [--run NAME(기본 q17-f1)] [--exclude N,…|none] [--no-eval]. 새 셋이 하나라도 없으면 멈춘다(--allow-missing 이면 빼고 낸다). 아무것도 지우지 않는다.
#   평가: 학습 뒤 submit-evals.sh q17-f1(같은 파티션 — q17-mt4-v3 와 같은 묶음: final·ck1500·ck2000·avg3, AMI 4 몫).
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; allow=0; part=apex; nodes=2; steps=2500; run=q17-f1; exclude=hpc-50,hpc-51,hpc-101,hpc-102; ev=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --allow-missing) allow=1; shift ;;  --partition) part=$2; shift 2 ;;  --nodes) nodes=$2; shift 2 ;;
    --max-steps) steps=$2; shift 2 ;;  --run) run=$2; shift 2 ;;  --exclude) exclude=$2; shift 2 ;;  --no-eval) ev=0; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $exclude == none ]] && exclude=""
export MAX_RESTART=${MAX_RESTART:-$([[ $part == apex ]] && echo 5 || echo 20)}
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; M=/soundai/Model/VAPASR
CORP=aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar; miss=()
for c in swbd chime6 nia23mt ami-far icsi-far notsofar-far chime6-far; do
  if [[ -f $P2/$c.refined.dialogues.jsonl ]]; then CORP+=",$c"; else miss+=("$c"); fi
done
SIMS=$D/sim-dialogue-v2,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1; MEET=""
for l in en ko de; do
  if [[ -f $D/sim-dialogue-v4-meet/sim_$l-meet.refined.dialogues.jsonl ]]; then MEET+=",sim_$l-meet:$([[ $l == en ]] && echo 0.6 || echo 0.3)"; else miss+=("sim_$l-meet"); fi
done
[[ -n $MEET ]] && SIMS+=",$D/sim-dialogue-v4-meet"
if (( ${#miss[@]} )); then echo "!! 아직 없는 셋: ${miss[*]}" >&2; (( allow )) || { echo "   모두 준비된 뒤 내거나 --allow-missing 으로 빼고 낸다" >&2; exit 1; }; fi
W=dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.0,dlg_far:1.0,dlg_mtko:0.5,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:1.0,sim_en-ko:1.0$MEET,rep_ko:1.2,rep_en:1.2,rep_de:1.2
EXC=(); [[ -n $exclude ]] && EXC=(--exclude "$exclude")
ARGS=(--run $run --init $M/stage3-q17-st3-p11-revive/checkpoint-3000 --partition $part --nodes $nodes --backchannel 1.0,3 --spk-act-weight 0.5
      --revive-dir $P2/revive-asr-v2 --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16,$D/stage4-dlg-units-meet-v1/targets-fix-gen-v3-bf16
      --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 --sim-trans-frac 0.8 --trans-lead-space --phone-aug 0.3
      --sim-data $SIMS --weights $W --set-phone sim_en-de-tel:0 --change-tag-weight 3 --hold-tag-weight 1.5 --eot-drop-synth 1.0 --eot-drop-sets xl
      --lr 4e-5 --max-steps $steps --decision-weight-sets xl --split v4 --dlg-corpora $CORP --max-speakers 4 --declare-all-p 0.5
      --extra "--far-aug 0.3 --far-sets dlg_ko,dlg_en,dlg_de,dlg_mt,dlg_mtko")
echo "[$(date '+%F %T')] F1 $run($part $nodes 노드, $steps step, MAX_RESTART $MAX_RESTART) · 대화 $CORP · 합성 $SIMS" >&2
out=$(bash slurm/submit-stage3-train.sh "${ARGS[@]}" ${EXC[@]+"${EXC[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^sbatch|Submitted|만들 예정|init 없음|거부" >&2 || true
dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true); [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true
[[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
echo "1 학습 job $dep"
(( ev )) || exit 0
out=$(bash slurm/submit-evals.sh q17-f1 --after-job "$dep" --partition "$part" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^목록|^sbatch" >&2 || true
echo "2 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
