#!/usr/bin/env bash
# Qwen3-ASR 1.7B 체인(2026-10-22, 사용자 "Qwen asr 1.7B 를 이용해서도 학습시키고 싶어"): 0.6B 와 같은 단계·레시피를 1.7B(/soundai/Model/Qwen3-ASR-1.7B)로.
# 체인 1–6 = 프로젝트 단계 Stage 1 ASR · Stage 2 AST 1–3 차(run 이름 s2·s3·s4-ast 는 0.6B 번역 학습 차수) · Stage 3 2 인 대화 · Stage 4 대화 번역(V3).
# 단계마다 앞 단계가 성공해야 시작한다(sbatch afterok). 한 번 실행으로 모두 제출하고 job 번호를 출력한다(제출은 사용자 — 스케줄러 셸에서).
#   1 q17-s1m3-d10-s0      Stage 1 영·한·독 δ4–10 3 epoch(semcommit) — init q17-s0/final(q06-s1m3-d10 과 같은 레시피)                 약 13 h
#   2 q17-s2-ast           단일 화자 번역(ast-v1, 번역 모드 0.5) 2 epoch — init q17-s1m3-d10-s0/final                                 약 8–9 h
#   3 q17-s3-ast           번역 모드 0.8 + 영어 음량 증강 2 epoch — init q17-s2-ast/final                                              약 8–9 h
#   4 q17-s4-ast           + CV4 영어 낭독·번역 2 epoch — init q17-s3-ast/final                                                        약 5–9 h
#   5 q17-st3-p11-revive   대화 Stage 3(0.6B p11 명령 그대로: 같은 언어 대화·합성·replay·revive v1, 4,000 step) — init q17-s4-ast/final   약 1.3–2.5 h
#   6 q17-v3-p1-tel-xlw    V3 R1x(전화 합성 셋 + 교대 가중은 교차 언어 셋만, 2,500 step) — init q17-st3-p11-revive/checkpoint-3000       약 1.2–2.2 h
#   시간(2026-10-23 다시 어림, apex 2 노드 순수 step 시간): 1 단계는 1.7B 실측(job 94809 step 1–3,630 중앙 2.86 s/step — 같은 step 구간의 0.6B 1.22 s 의
#   2.3 배). 2–6 단계는 0.6B 단계별 step 시간(작업 로그 s_per_step 중앙)에 1.7B 배율 1.2–2.3 을 곱한 범위다(데이터 읽기 병목이면 작고 GPU 계산이면 크다).
#   합 약 37–46 h + 단계마다 시작·재시작. 2026-10-22 의 "1.7B 와 0.6B 속도가 같다(약 30 h)" 는 10-08 한 구간(둘 다 1.1–1.7 s/step)만 보고 낸 틀린 어림이었다.
#   --s1 resume: Stage 1 을 2026-10-08 run q17-s1m3-d10(q17-s1m-r3 에서 1 epoch, checkpoint-1000 에서 멈춤)을 이어 하는 판으로 — 남은 4,391 step 약 2.9 h.
#     0.6B 와 Stage 1 이 다르다(영·한 3 epoch δ1–8 뒤 영·한·독 1 epoch — 독일어·δ4–10 은 0.6B 의 1/3). 2 단계 init 도 그 final 로 바꿔 넘긴다.
#   5 단계 체크포인트는 0.6B 처럼 checkpoint-3000 을 6 단계 init 으로 쓴다(0.6B 는 평가로 ck3000 을 골랐다 — 1.7B 는 평가 없이 같은 위치).
# 쓰기(스케줄러 셸, /soundai/users/tskim/VAPKT):
#   bash slurm/submit-q17-chain.sh --dry-run            # sbatch 없이 단계별 명령만(앞 단계 job 은 0 으로 둔다)
#   bash slurm/submit-q17-chain.sh                      # 1–6 단계 제출
#   bash slurm/submit-q17-chain.sh --from 5 [--after-job JOBID]   # 5 단계부터(앞 단계가 이미 끝났거나 JOBID 가 끝난 뒤)
#   bash slurm/submit-q17-chain.sh --to 5                          # 1–5 단계만(6 단계 레시피는 나중에 --from 6 --after-job <5 단계 job>)
#   --exclude N,…: 모든 단계 sbatch 에서 더 뺄 노드(기본 hpc-50,hpc-51,hpc-101,hpc-102 — V3 R1·R2 때와 같다, 'none' 이면 더 빼지 않는다).
#   --s1 mirror|resume: Stage 1 판(기본 mirror = 0.6B 와 같은 q17-s1m3-d10-s0). --from 2 이후로 낼 때도 앞서 쓴 판을 같이 준다(2 단계 init 이 따라간다).
#   --dlg r1x|noeot|noeot-mt4(2026-10-23): 대화 5·6 단계 레시피. r1x(기본) = 지금 명령. noeot = 두 단계 모두 --no-eot(run 이름 -noeot).
#     noeot-mt4 = noeot + 6 단계에 회의(AMI·ICSI·NOTSOFAR dlg_mt 1.0·회의 번역 타깃)·화자 슬롯 4(--max-speakers 4 --declare-all-p 0.5)·split v4(run -noeot-mt4).
#     0.6B 시험(submit-q06-noeot-chain.sh·submit-q06-mt4.sh)과 같은 변경이다. 바꿀 때: 기다리는 5·6 단계 job 과 그 평가 job 을 사용자가 취소하고
#     bash slurm/submit-q17-chain.sh --from 5 --after-job <4 단계 job> --dlg <레시피> --eval
#   --eval: 6 단계 뒤 평가 묶음(slurm/submit-evals.sh q17-v3·q17-noeot-v3·q17-noeot-mt4-v3)을 6 단계 job 성공 뒤로 함께 낸다.
# 단계마다 MAX_RESTART(기본 5). 앞 단계가 실패하면 뒤 단계는 DependencyNeverSatisfied 로 남는다 — 취소는 사용자가 한다. 경로 변수는 이 스크립트 안에서만 쓴다.
set -euo pipefail
cd "$(dirname "$0")/.."
from=1; to=6; dep=""; dry=""; exclude=hpc-50,hpc-51,hpc-101,hpc-102; s1=mirror; dlg=r1x; ev=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --from) from=$2; shift 2 ;;  --to) to=$2; shift 2 ;;  --after-job) dep=$2; shift 2 ;;  --dry-run) dry=--dry-run; shift ;;  --exclude) exclude=$2; shift 2 ;;
    --s1) s1=$2; shift 2 ;;  --dlg) dlg=$2; shift 2 ;;  --eval) ev=1; shift ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
[[ $from =~ ^[1-6]$ && $to =~ ^[1-6]$ && $from -le $to ]] || { echo "--from·--to 는 1–6(from ≤ to)" >&2; exit 2; }
[[ -z $dep || $dep =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
case $s1 in
  mirror) S1SH=slurm/submit-q17-s1m3-d10-s0.sh; S1RUN=q17-s1m3-d10-s0; S1WHAT="Stage 1 q17-s1m3-d10-s0(q17-s0 에서 3 epoch — 0.6B 와 같은 레시피)" ;;
  resume) S1SH=slurm/submit-q17-s1m3-d10.sh; S1RUN=q17-s1m3-d10; S1WHAT="Stage 1 q17-s1m3-d10(q17-s1m-r3 에서 1 epoch — checkpoint-1000 이어 하기)" ;;
  *) echo "--s1 은 mirror|resume" >&2; exit 2 ;;
esac
case $dlg in r1x) SUF=""; EVSET=q17-v3 ;; noeot) SUF=-noeot; EVSET=q17-noeot-v3 ;; noeot-mt4) SUF=-noeot; EVSET=q17-noeot-mt4-v3 ;; *) echo "--dlg 는 r1x|noeot|noeot-mt4" >&2; exit 2 ;; esac
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
  local n=$1 what=$2; shift 2; [[ $n -lt $from || $n -gt $to ]] && return 0
  local a=(); [[ -n $dep ]] && a=(--afterok "$dep")
  echo "[$(date '+%F %T')] 단계 $n $what${dep:+ (job $dep 성공 뒤)}" >&2
  local out; out=$("$@" ${EXC[@]+"${EXC[@]}"} ${a[@]+"${a[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 단계 $n 제출 실패 — 뒤 단계는 내지 않는다" >&2; exit 1; }
  echo "$out" | grep -E "^sbatch|Submitted|이어서|만들 예정|init 없음|거부" >&2 || true
  dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true)
  [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true                                          # dry-run: 대화 단계 스크립트는 job 줄을 내지 않는다 → 0
  [[ -n $dep ]] || { echo "!! 단계 $n job 번호를 못 읽었다" >&2; exit 1; }
  echo "단계 $n job $dep"
}
stage 1 "$S1WHAT" bash $S1SH
stage 2 "Stage 2(AST 1 차) q17-s2-ast(init $S1RUN/final)" bash slurm/submit-q17-s2-ast.sh --init $M/semcommit-$S1RUN/final
stage 3 "Stage 2(AST 2 차) q17-s3-ast" bash slurm/submit-q17-s3-ast.sh
stage 4 "Stage 2(AST 3 차) q17-s4-ast" bash slurm/submit-q17-s4-ast.sh
NE=(); [[ $dlg != r1x ]] && NE=(--no-eot)                                                      # --dlg: <EOT> 없음(5·6 단계), 회의·화자 슬롯 4(6 단계)
MT=(); S6RUN=q17-v3-p1-tel-xlw$SUF
if [[ $dlg == noeot-mt4 ]]; then
  S6RUN=q17-v3-p1-tel-xlw-noeot-mt4
  MT=(--weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.0,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2
      --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16,$D/stage4-dlg-units-meet-v1/targets-fix-gen-v3-bf16 --split v4
      --dlg-corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar --max-speakers 4 --declare-all-p 0.5)   # 제출 래퍼는 같은 옵션의 뒤 값을 쓴다
fi
stage 5 "Stage 3(2 인 대화) q17-st3-p11-revive$SUF" bash slurm/submit-stage3-train.sh --run q17-st3-p11-revive$SUF --init $M/semcommit-q17-s4-ast/final "${P11[@]}" ${NE[@]+"${NE[@]}"}
stage 6 "Stage 4(대화 번역, V3 R1x) $S6RUN" bash slurm/submit-stage3-train.sh --run $S6RUN --init $M/stage3-q17-st3-p11-revive$SUF/checkpoint-3000 "${R1X[@]}" ${NE[@]+"${NE[@]}"} ${MT[@]+"${MT[@]}"}
if (( ev )) && [[ $to == 6 ]]; then                                                             # 6 단계 뒤 평가 묶음
  echo "[$(date '+%F %T')] 평가 $EVSET (job $dep 성공 뒤)" >&2
  out=$(bash slurm/submit-evals.sh $EVSET --after-job "$dep" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
  echo "$out" | grep -E "^목록|^sbatch" >&2 || true; echo "평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
fi
echo "[$(date '+%F %T')] 체인 제출 끝${dry:+(dry-run — 아무것도 제출하지 않았다)}" >&2
