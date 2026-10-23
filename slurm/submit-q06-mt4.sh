#!/usr/bin/env bash
# 0.6B 회의 데이터 + 화자 슬롯 4 시험(2026-10-23, 사용자 "회의 데이터 넣고 화자 슬롯도 4개를 사용해 학습시키고 싶어"): V3 R1x 명령에 아래만 더해
# 대화 Stage 4 를 다시 학습하고, 끝나면 SLURM 평가 노드로 R1x 와 같은 평가(AMI 는 4 슬롯 선언)를 잰다.
#   더하는 것: AMI·ICSI·NOTSOFAR 학습 분할(dlg_mt 1.0 — V2 와 같은 가중) · 회의 번역 타깃(stage4-dlg-units-meet-v1) ·
#              화자 슬롯 4(--max-speakers 4 --declare-all-p 0.5 — V2 와 같다) · split v4(기존 7 코퍼스 held-out 은 v2 와 같고 AMI 공식 dev·eval·NOTSOFAR dev·eval 을 뺀다).
#   그대로 둔 것: 창 20–40 s 격자, 합성 셋(2 인·교차 언어·전화), C1·C5, 2,500 step, lr 4e-5, init p11 ck3000 — V2 의 다른 변경(인코더 탭·20–90 s 창·3–4 인 합성)은 넣지 않는다.
#   1 v3-p1-tel-xlw-mt4   init st3-p11-revive/checkpoint-3000, R1x 명령 + 위 변경        0.6B 약 1.0–1.3 h + 시작
#   2 평가 mt4-v3(slurm/submit-evals.sh): ck1500·ck2000·final·avg3 × 9 작업 + final·avg3 AMI
#   비교: R1x(v3-p1-tel-xlw)·V2(v2-p1) — dev 귀속·TAXI·같은 언어 가드·AMI(IHM·SDM)·번역 BLEU. <EOT> 는 R1x 처럼 둔다(no-EOT 시험과 섞지 않는다).
# 쓰기(스케줄러 셸): bash slurm/submit-q06-mt4.sh [--dry-run] [--exclude N,…|none]   (기본 제외 hpc-50·51·101·102, MAX_RESTART 5). 아무것도 지우지 않는다.
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
ARGS=(--run v3-p1-tel-xlw-mt4 --init $M/stage3-st3-p11-revive/checkpoint-3000
      --partition apex --nodes 2 --backchannel 1.0,3 --spk-act-weight 0.5 --revive-dir $D/phase2/revive-asr-v2
      --dlg-trans $D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16,$D/stage4-dlg-units-meet-v1/targets-fix-gen-v3-bf16
      --dlg-trans-input $D/stage4-dlg-units-revive-v1/targets-fix-gen-v3-bf16 --dlg-ast-frac 0.5 --sim-trans-frac 0.8 --trans-lead-space --phone-aug 0.3
      --sim-data $D/sim-dialogue-v2,$D/sim-dialogue-xl-v1,$D/sim-dialogue-xl-tel-v1
      --weights dlg_ko:2,dlg_en:0.7,dlg_de:0.7,dlg_mt:1.0,sim_ko:0.5,sim_en:0.8,sim_de:0.8,sim_en-de:0.5,sim_en-de-tel:1.0,sim_en-ko:1.0,rep_ko:1.2,rep_en:1.2,rep_de:1.2
      --set-phone sim_en-de-tel:0 --change-tag-weight 3 --hold-tag-weight 1.5 --eot-drop-synth 1.0 --eot-drop-sets xl --lr 4e-5 --max-steps 2500 --decision-weight-sets xl
      --split v4 --dlg-corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar --max-speakers 4 --declare-all-p 0.5)
echo "[$(date '+%F %T')] 1 v3-p1-tel-xlw-mt4(회의 + 화자 슬롯 4)" >&2
out=$(bash slurm/submit-stage3-train.sh "${ARGS[@]}" ${EXC[@]+"${EXC[@]}"} $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 학습 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^sbatch|Submitted|init 없음|거부" >&2 || true
dep=$(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1 || true); [[ -n $dep ]] || { [[ -n $dry ]] && dep=0; } || true
[[ -n $dep ]] || { echo "!! job 번호를 못 읽었다" >&2; exit 1; }
echo "1 학습 job $dep"
echo "[$(date '+%F %T')] 2 평가 mt4-v3 (job $dep 성공 뒤)" >&2
out=$(bash slurm/submit-evals.sh mt4-v3 --after-job "$dep" $dry 2>&1) || { echo "$out" | tail -5 >&2; echo "!! 평가 제출 실패" >&2; exit 1; }
echo "$out" | grep -E "^목록|^sbatch" >&2 || true
echo "2 평가 job $(echo "$out" | grep -o "Submitted batch job [0-9]*" | awk '{print $4}' | tail -1)"
echo "[$(date '+%F %T')] 제출 끝${dry:+(dry-run — 아무것도 제출하지 않았다)}" >&2
