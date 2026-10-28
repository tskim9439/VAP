#!/usr/bin/env bash
# F1 데이터 확장 시험(2026-10-26, 사용자 "모두 진행하는 데, 모두 진행하고 나서 E5 E6 학습 진행하자" — wiki task-data-expansion-20261026):
# 1.7B 회의 + 4 슬롯(q17-v3-p1-tel-xlw-mt4, submit-q17-chain.sh --dlg mt4 의 6 단계)과 같은 init(5 단계 ck3000)·레시피·학습률에 새 데이터를 더한다. 더하는 것:
#   원거리 회의 녹음(dlg_far: ami-far·icsi-far·notsofar-far·chime6tr-far — p2_build_far, 회의 번역 타깃은 원본 id 로 쓴다), CHiME-6 train 근접(chime6tr → dlg_mt;
#   dev·eval 세션은 평가에만 — 키를 chime6 와 다르게, wiki decision-chime6-train-sessions-20261026),
#   Switchboard(swbd → 따로 dlg_sw 0.7 — --dlg-group-map), 한국어 회의 NIA23(nia23mt → dlg_mtko), 회의형 합성 v4(sim_<l>-meet), 화자별 원거리 증강(--far-aug 0.3, 근접 대화·회의 셋).
#   step(2026-10-26 사용자 "데이터도 늘었는데, step 수를 늘려야 하는거 아닐까?" → "5,000 step + Switchboard 분리"): 2,500 step 이면 새 그룹이 몫을 나눠 기존 셋 노출이 줄고
#   (한국어 대화 −18 %, 회의 근접 −42 %), swbd 가 dlg_en 에서 otoSpeech 를 묽혔다(−85 %). 5,000 step 이면 실제 대화 셋(epoch 기준)을 mt4 와 같은 1.7 회 돌고 기존 셋이 모두
#   mt4 이상 노출된다. 평가 체크포인트는 같은 상대 위치(60·80 %) ck3000·ck4000·final(avg3 도 그 셋).
# 쓰기(스케줄러 셸): bash slurm/submit-q17-f1.sh [--dry-run] [--allow-missing] [--partition P(기본 apex)] [--nodes N(기본 2)] [--max-steps N(기본 2500)]
#   [--run NAME(기본 q17-f1)] [--exclude N,…|none] [--no-eval] [--save-every N] [--after JOBID(그 job 이 끝난 뒤 — 성패 무관, afterany)]
#   [--afterok JOBID(그 job 이 성공한 뒤 — 예 E6 은 E1c)] [--train-extra "stage3_train.py 인자"(F1 레시피에 덧붙임 — E5 --slot-input, E6 + --diar-head …)]
#   [--eval-set 묶음(기본 q17-f1 — E5 q17-e5, E6 q17-e6; submit-evals.sh)] [--init DIR(2026-10-28, 기본 5 단계 ck3000 — E7b 는 E6 final 에서)]
#   [--transcribe-only(2026-10-28 E9 — 받아쓰기만: 대화·교차 언어 합성 번역 타깃·번역 행 앞 공백·교대/머무름 가중·EOT-문맥 dropout 을 빼고 --sim-trans-frac 0,
#    replay 도 --ast-frac 0. 데이터·가중치·나머지 레시피는 같다)].
#   새 셋이 하나라도 없으면 멈춘다(--allow-missing 이면 빼고 낸다). 아무것도 지우지 않는다.
#   low_p_hpc(2026-10-26 사용자 "low_p_hpc 로 제출할 수 있도록 해줘"): 선점 유예(≈70 s)가 1.7B 저장(≈3.5 분)보다 짧아 선점 때 저장은 끊긴다 —
#   재개는 마지막 완성 checkpoint(last_complete_checkpoint)에서. 그래서 apex 가 아니면 --save-every 기본 250(apex 500, 평가 avg3 = ck1500·ck2000·final 그대로),
#   MAX_RESTART 기본 20(apex 5). 평가 노드도 같은 파티션(--requeue, 끝난 창은 건너뛴다).
#   평가: 학습 뒤 submit-evals.sh q17-f1(같은 파티션 — q17-mt4-v3 와 같은 묶음: final·ck1500·ck2000·avg3, AMI 4 몫).
set -euo pipefail
cd "$(dirname "$0")/.."
dry=""; allow=0; part=apex; nodes=2; steps=5000; run=q17-f1; exclude=hpc-50,hpc-51,hpc-101,hpc-102; ev=1; save=""; after=""; afterok=""; textra=""; evset=q17-f1; init=""; tonly=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) dry=--dry-run; shift ;;  --allow-missing) allow=1; shift ;;  --partition) part=$2; shift 2 ;;  --nodes) nodes=$2; shift 2 ;;
    --max-steps) steps=$2; shift 2 ;;  --run) run=$2; shift 2 ;;  --exclude) exclude=$2; shift 2 ;;  --no-eval) ev=0; shift ;;  --save-every) save=$2; shift 2 ;;  --after) after=$2; shift 2 ;;  --afterok) afterok=$2; shift 2 ;;  --init) init=$2; shift 2 ;;  --train-extra) textra=$2; shift 2 ;;  --eval-set) evset=$2; shift 2 ;;
    --transcribe-only) tonly=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $exclude == none ]] && exclude=""
[[ -z $after || $after =~ ^[0-9]+([,:][0-9]+)*$ ]] || { echo "--after 는 job id(여럿은 쉼표 — 모두 끝난 뒤): $after" >&2; exit 2; }
[[ -z $afterok || $afterok =~ ^[0-9]+$ ]] || { echo "--afterok 는 job id: $afterok" >&2; exit 2; }
[[ -z $after || -z $afterok ]] || { echo "--after 와 --afterok 는 함께 쓰지 않는다" >&2; exit 2; }
export MAX_RESTART=${MAX_RESTART:-$([[ $part == apex ]] && echo 5 || echo 20)}
[[ -n $save ]] || save=$([[ $part == apex ]] && echo 500 || echo 250)
CKA=$((steps * 3 / 5)); CKB=$((steps * 4 / 5))                                                 # 평가 체크포인트(60·80 %) — mt4 2,500 step 의 1500·2000 과 같은 위치
(( CKA % save == 0 && CKB % save == 0 )) || { echo "--save-every $save: 평가가 checkpoint-$CKA·$CKB 를 쓴다 — 둘의 약수로" >&2; exit 2; }
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; M=/soundai/Model/VAPASR; init=${init:-$M/stage3-q17-st3-p11-revive/checkpoint-3000}
CORP=aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar; miss=()
for c in swbd chime6tr nia23mt ami-far icsi-far notsofar-far chime6tr-far; do
  if [[ -f $P2/$c.refined.dialogues.jsonl ]]; then CORP+=",$c"; else miss+=("$c"); fi
done
SIMS=$D/sim-dialogue-v2,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1; MEET=""
for l in en ko de; do
  if [[ -f $D/sim-dialogue-v4-meet/sim_$l-meet.refined.dialogues.jsonl ]]; then MEET+=",sim_$l-meet:$([[ $l == en ]] && echo 0.6 || echo 0.3)"; else miss+=("sim_$l-meet"); fi
done
[[ -n $MEET ]] && SIMS+=",$D/sim-dialogue-v4-meet"
if (( ${#miss[@]} )); then echo "!! 아직 없는 셋: ${miss[*]}" >&2; (( allow )) || { echo "   모두 준비된 뒤 내거나 --allow-missing 으로 빼고 낸다" >&2; exit 1; }; fi
FARW=""; [[ $CORP == *-far* ]] && FARW=",dlg_far:1.0"; MTKO=""; [[ ,$CORP, == *,nia23mt,* ]] && MTKO=",dlg_mtko:0.5"   # --allow-missing: 코퍼스가 없는 그룹은 가중치에서 뺀다
SW=""; GMAP=""; [[ ,$CORP, == *,swbd,* ]] && { SW=",dlg_sw:0.7"; GMAP=" --dlg-group-map swbd:dlg_sw"; }                  # Switchboard 는 따로(otoSpeech 를 묽히지 않게)
W=dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.0$FARW$MTKO$SW,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:1.0,sim_en-ko:1.0$MEET,rep_ko:1.2,rep_en:1.2,rep_de:1.2
EXC=(); [[ -n $exclude ]] && EXC=(--exclude "$exclude"); [[ -n $after ]] && EXC+=(--after "$after")   # submit-stage3-train.sh --after = afterany
[[ -n $afterok ]] && EXC+=(--afterok "$afterok")                                                # 성공 뒤(afterok)
if (( tonly )); then                                                                            # E9: 받아쓰기만(번역·C1·C5 없음 — replay 번역도 끔)
  TR=(--sim-trans-frac 0); XTR=" --ast-frac 0"
else
  TR=(--dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16,$D/stage4-dlg-units-meet-v1/targets-fix-gen-v3-bf16 --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16
      --dlg-ast-frac 0.5 --sim-trans-frac 0.8 --trans-lead-space --change-tag-weight 3 --hold-tag-weight 1.5 --eot-drop-synth 1.0 --eot-drop-sets xl --decision-weight-sets xl); XTR=""
fi
ARGS=(--run $run --init $init --partition $part --nodes $nodes --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $P2/revive-asr-v2 "${TR[@]}" --phone-aug 0.3
      --sim-data $SIMS --weights $W --set-phone sim_en-de-tel:0
      --lr 4e-5 --max-steps $steps --save-every $save --split v4 --dlg-corpora $CORP --max-speakers 4 --declare-all-p 0.5
      --extra "--far-aug 0.3 --far-sets dlg_ko,dlg_en,dlg_de,dlg_mt,dlg_mtko$GMAP$XTR${textra:+ $textra}")
echo "[$(date '+%F %T')] F1 $run($part $nodes 노드, $steps step, 저장 $save step 마다, 평가 ck$CKA·ck$CKB·final, MAX_RESTART $MAX_RESTART${after:+, job ${after//[,:]/·} 끝난 뒤}${afterok:+, job $afterok 성공 뒤}${textra:+, 덧붙임 $textra}$( (( tonly )) && echo ", 받아쓰기만"), 평가 $evset) · 대화 $CORP · 합성 $SIMS" >&2
out=$(bash slurm/submit-stage3-train.sh "${ARGS[@]}" ${EXC[@]+"${EXC[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^sbatch|Submitted|만들 예정|init 없음|거부" >&2 || true
dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true); [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true
[[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
echo "1 학습 job $dep"
(( ev )) || exit 0
out=$(bash slurm/submit-evals.sh "$evset" --after-job "$dep" --partition "$part" --cks "$CKA,$CKB" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^목록|^sbatch" >&2 || true
echo "2 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
