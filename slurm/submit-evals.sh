#!/usr/bin/env bash
# 평가 묶음 SLURM 제출(2026-10-23, 사용자 "평가 작업도 SLURM 통해 수행할 수 있도록 명령어를 알려줘" · "AMI_IHM 이랑 AMI_SDM 에 대해서도 평가를 수행하고 싶어"):
# 묶음의 작업 목록('<GPU>|<명령>')을 만들어 slurm/v3-eval-node.sbatch(apex 1 노드 8 GPU, GPU 마다 위에서부터 차례로)에 낸다. 제출은 사용자(스케줄러 셸).
#   r1x-ami  0.6B V3 R1x final·세 체크포인트 평균(avg3) — AMI 공식 dev·eval(IHM-mix·SDM, 창 전부, δ4·10), 모델마다 4 몫(GPU 4 장). 의존 없음(바로 시작).
#            V1 ck2000 은 2026-10-21 에 같은 명령으로 쟀다(results/stage3-eval-ami-st4-m2a-ck2000).
#   q17-s1   1.7B Stage 1 q17-s1m3-d10-s0/final 단일 발화 ASR — 0.6B 2026-10-09 평가와 같은 명령(run_single_turn_asr_mxc.sh, batch 64, δ4·6·8·10 + final):
#            영·한 single-turn-vpaa-kspon-v1 은 GPU 0·1, 독일어 single-turn-yodas-de-dev-v1 은 GPU 2·3. 기본 의존 job 94809(Stage 1).
#   q17-v3   1.7B V3 R1x q17-v3-p1-tel-xlw ck1500·ck2000·final·avg3 × 9 작업(dlg·xl·taxi40·stage3·regress·xltel·xleval·taxi90·onset — 0.6B R1x 평가
#            job 94789 와 같은 v3_eval_task.sh) + final·avg3 AMI(4 몫씩, 각 GPU 목록 맨 뒤). 평균은 GPU 7 의 첫 작업(avg_checkpoints.py, CPU)이
#            VAPKT-data/models/stage3-q17-v3-p1-tel-xlw-avg3/final 에 만들고 평균 평가들은 그 DONE 을 기다린다. 기본 의존 job 94814(6 단계).
#   noeot-v3 0.6B <EOT> 없는 대화 학습 시험(2026-10-23, slurm/submit-q06-noeot-chain.sh): V3 R1x-noEOT(v3-p1-tel-xlw-noeot)에 q17-v3 와 같은 묶음 +
#            대화 Stage 3 st3-p11-revive-noeot checkpoint-3000 의 같은 언어 평가(stage3). 의존 job 은 --after-job 으로 꼭 준다(그 시험의 Stage 4 학습 job).
#   noeot-mt4-v3  0.6B 회의 + 화자 슬롯 4 + <EOT> 없음 시험(2026-10-23, slurm/submit-q06-mt4.sh): v3-p1-tel-xlw-noeot-mt4 에 q17-v3 와 같은 묶음(AMI 는 4 슬롯 선언). --after-job 필수.
#   q17-noeot-v3 · q17-noeot-mt4-v3  1.7B 대화 레시피를 바꿔 낸 6 단계(submit-q17-chain.sh --dlg noeot|noeot-mt4 --eval)에 q17-v3 와 같은 묶음. --after-job 필수.
#   --after-job JOBID  의존 job 을 바꾼다(재제출 등).   --dry-run  목록·sbatch 명령만 출력(파일·제출 없음).   --partition P  평가 노드 파티션(기본 sbatch 의 apex — 예 low_p_hpc).
# 시간 한도는 24 h 로 낸다(sbatch 기본 12 h 를 명령줄로 덮는다). 목록은 VAPKT-data/tmp/evals-<묶음>-<시각>.txt(새 파일).
# 각 작업은 끝났거나 시작된 산출이 있으면 건너뛰거나 이어 하고(AMI 몫은 이어 한다) 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
set_=${1:-}; [[ $# -gt 0 ]] && shift; dep=""; dry=0; part=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --after-job) dep=$2; shift 2 ;;  --dry-run) dry=1; shift ;;  --partition) part=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
case $set_ in r1x-ami) ;; q17-s1) dep=${dep:-94809} ;; q17-v3) dep=${dep:-94814} ;; noeot-v3|noeot-mt4-v3|q17-noeot-v3|q17-noeot-mt4-v3) [[ -n $dep ]] || { echo "$set_ 는 --after-job(학습 job)이 필요하다" >&2; exit 2; } ;;
  *) echo "묶음: r1x-ami | q17-s1 | q17-v3 | noeot-v3 | noeot-mt4-v3 | q17-noeot-v3 | q17-noeot-mt4-v3" >&2; exit 2 ;; esac
[[ -z $dep || $dep =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
D=/soundai/users/tskim/VAPKT-data; MR=$D/models; X=raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh
L=()
ami4() { local g0=$1 run=$2 root=$3 r; for r in 0 1 2 3; do L+=("$((g0 + r))|${root:+MODEL_ROOT=$root }bash $X ami $run final $r/4"); done; }   # GPU g0..g0+3 에 4 몫
v3set() {   # V3 run $1: ck1500·ck2000·final·avg3 × 9 작업 + final·avg3 AMI 4 몫씩 — 평균은 GPU 7 의 첫 작업, 평균 평가는 그 DONE 을 기다린다
  local RUN=$1 AVG=$1-avg3 M=/soundai/Model/VAPASR/stage3-$1 i=0 t k; local -a K=(dlg xl taxi40 stage3 regress xltel xleval taxi90 onset)
  L+=("7|\${PY:-python} experiments/avg_checkpoints.py --out $MR/stage3-$AVG/final --done $M/checkpoint-1500 $M/checkpoint-2000 $M/final > $D/tmp/avg-$RUN.log 2>&1")
  for t in checkpoint-1500 checkpoint-2000 final; do for k in "${K[@]}"; do L+=("$((i % 8))|bash $X $k $RUN $t"); i=$((i + 1)); done; done
  for k in "${K[@]}"; do L+=("$((i % 8))|MODEL_ROOT=$MR bash $X $k $AVG final"); i=$((i + 1)); done
  ami4 4 $RUN ""; ami4 0 $AVG $MR                                                           # 단일 체크포인트 작업이 적은 GPU 4–7 에 final 몫
}
case $set_ in
  r1x-ami) ami4 0 v3-p1-tel-xlw ""; ami4 4 v3-p1-tel-xlw-avg3 $MR ;;
  q17-s1)
    CK=/soundai/Model/VAPASR/semcommit-q17-s1m3-d10-s0/final; EV=$D/data/evaluation
    L+=("0|bash experiments/run_single_turn_asr_mxc.sh $CK $D/results/single-turn-vpaa-kspon-q17-s1m3-d10-s0 0,1 64 0 4,6,8,10 $EV/single-turn-vpaa-kspon-v1/manifest.jsonl final")
    L+=("2|bash experiments/run_single_turn_asr_mxc.sh $CK $D/results/single-turn-yodas-de-q17-s1m3-d10-s0 2,3 64 0 4,6,8,10 $EV/single-turn-yodas-de-dev-v1/manifest.jsonl final") ;;
  q17-v3) v3set q17-v3-p1-tel-xlw ;;
  noeot-v3) v3set v3-p1-tel-xlw-noeot; L+=("4|bash $X stage3 st3-p11-revive-noeot checkpoint-3000") ;;      # Stage 3 같은 언어 평가(0.6B p11 ck3000 과 비교)
  noeot-mt4-v3) v3set v3-p1-tel-xlw-noeot-mt4 ;;
  q17-noeot-v3) v3set q17-v3-p1-tel-xlw-noeot ;;
  q17-noeot-mt4-v3) v3set q17-v3-p1-tel-xlw-noeot-mt4 ;;
esac
T=$D/tmp/evals-$set_-$(date +%Y%m%d-%H%M%S).txt
cmd="sbatch ${part:+--partition=$part }${dep:+--dependency=afterok:$dep }--time=24:00:00 --export=ALL,TASKS=$T slurm/v3-eval-node.sbatch"
if (( dry )); then printf '%s\n' "${L[@]}"; echo "$cmd"; echo "Submitted batch job 0"; exit 0; fi
[[ -e $T ]] && { echo "이미 있다: $T" >&2; exit 1; }
{ echo "# 평가 묶음 $set_(slurm/submit-evals.sh, $(date '+%F %T'))${dep:+ — job $dep 성공 뒤}"; printf '%s\n' "${L[@]}"; } > "$T"
echo "목록 $T (${#L[@]} 작업)" >&2
$cmd
