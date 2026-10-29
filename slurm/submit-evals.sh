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
#   mt4-v3 · noeot-mt4-v3  0.6B 회의 + 화자 슬롯 4 시험(slurm/submit-q06-mt4.sh, [--no-eot]): v3-p1-tel-xlw[-noeot]-mt4 에 q17-v3 와 같은 묶음(AMI 는 4 슬롯 선언). --after-job 필수.
#   q17-mt4-v3 · q17-noeot-v3 · q17-noeot-mt4-v3  1.7B 대화 레시피를 바꿔 낸 6 단계(submit-q17-chain.sh --dlg mt4|noeot|noeot-mt4 --eval)에 q17-v3 와 같은 묶음. --after-job 필수.
#   v2-amifull · mt4-amifull · q17-mt4-amifull  AMI 회의 통째(2026-10-24, 사용자 "AMI 평가에 한번에 돌릴 수 있도록 창 길이를 늘리고 싶은데, 자르지 않고"):
#            v3_eval_task.sh amifull — 34 회의 × IHM-mix·SDM 을 창으로 자르지 않고 한 스트림으로(세션 단위 cpWER, δ4·10, LLM 문맥 창 30 s — 바꾸려면 서브셸에서
#            CTX=60 를 export 하고 낸다), 8 몫(GPU 0–7, 몫 0 이 --verify). 노드 하나에 모델 하나 약 4 h(디코드 eager 25 ms/step — V3 README 28 절).
#            v2-amifull = V2 v2-p1 final(4 슬롯, 의존 없음). mt4-amifull·q17-mt4-amifull = 회의 + 4 슬롯 run 의 final·avg3(평균은 GPU 7 의 첫 작업 — 이미 있으면
#            DONE 만 맞춘다), --after-job 필수(학습 job). 2 슬롯 모델(R1x 등)은 4 인 회의를 두 슬롯에 담아야 해 세션 cpWER 가 구조적으로 크다 — 묶음에 넣지 않았다.
#   q17-e2-tgt  E2 대상 화자 모드(slurm/submit-q17-e2-tgt.sh, 2026-10-25): q17-e2-tgt final 을 AMI 정답 활동 주입·화자별 디코드(v3_eval_task.sh amitgt)로 8 몫. --after-job 필수.
#   q17-f1  F1 데이터 확장 시험(slurm/submit-q17-f1.sh, 2026-10-26): q17-mt4-v3 와 같은 묶음(체크포인트 둘·final·avg3 × 9 작업 + AMI 4 몫씩 — 5,000 step 이라 --cks 3000,4000). --after-job 필수.
#   q17-e6-amifull  E7 기준선(2026-10-27): E6 final 회의 통째를 조각마다 새 슬롯(예전 encode_long)과 AOSC 스트리밍(헤드는 E6 그대로 — AOSC 학습 없음)으로 8 몫씩. 의존 없음.
#   q17-e7-amifull  E7: E6 final 에 AOSC 로 학습한 헤드(diar-q17-diar-e7a final)를 끼워 회의 통째 AOSC 8 몫. --after-job(E7a 학습 job) 필수.
#   q17-e6  E6 인과 화자 분리 헤드 + 슬롯 활동 입력(2026-10-27, slurm/submit-q17-e56.sh): q17-f1 과 같은 묶음(헤드 예측 활동으로 — 모든 디코드 경로가 model.encode 를 쓴다,
#           --cks 3000,4000) + final 정답 슬롯 활동 입력(amiact, 8 몫 — 헤드 대신 정답일 때의 상한). --after-job 필수.
#   δ10 빠른 평가(2026-10-28, 사용자 "평가하는 데, 너무 오래 걸린다. 일단 AMI IHM 이랑 SDM 두개에 대해 delta 10 으로만 평가해서 비교해서 의사결정을 빠르게 하자"):
#     AMI 작업을 DELTAS=10 으로(산출 이름에 -d10). 창 AMI 는 8 몫(GPU 0–7)으로 나눠 한 노드에서 약 20 분.
#   q17-e8-ami10  E8 final AMI 창 δ10 8 몫. --after-job 필수.   q17-e6-amifull-d10  E6 회의 통째 조각마다 새로(AOSC 없음) δ10 8 몫 — 기준선(의존 없음).
#   q17-e7-amifull-guide  E7(2026-10-28): q17-e7-amifull 과 같되 화자 태그를 활동으로 이끈다(GUIDE=1 — --act-tag-guide, 학습 없음), δ10. 의존 없음.
#   q17-e7b  E7b(2026-10-28, slurm/submit-q17-e7b.sh — LLM 이 회의 단위 슬롯을 따르게 --global-slots 로 미세조정): final 회의 통째 AOSC + E7a 헤드(8 몫) + AMI 창(8 몫,
#            창 귀속이 나빠지지 않았는지) — 모두 δ10. --after-job 필수.
#   q17-e9  E9 슬롯 블록 형식(2026-10-28, slurm/submit-q17-e9.sh): q17-e7b 와 같은 묶음(AMI 창 δ10 8 몫 + 회의 통째 AOSC·E7a 헤드 δ10 8 몫) — 평가가 config.slot_blocks 로
#           슬롯 블록 디코드를 고르고 행에 LLM DER(<BEGIN>·<EOT>)·청크 ms 를 남긴다. --after-job 필수.
#   q17-e9b  E9b(2026-10-28, slurm/submit-q17-e9b.sh — 학습 창 20–90 s): q17-e9 와 같은 묶음. --after-job 필수.
#   q17-e10  E10(2026-10-29, slurm/submit-q17-e10.sh — 슬롯 추출기): GPU 마다 AMI 창 δ10 → 회의 통째 AOSC·E7a δ10 → 섞인 소리 끈(MIXOFF=1) 창 → 섞인 소리 끈 회의 통째.
#            --after-job 필수.
#   q17-e9-n3d  E11 진단(2026-10-29, wiki output-n3d-branch-20261029 — 학습 없음): Nemotron 3 Diarization(고정 전문가, N3D=.nemo — 기본 VAPKT-data/vapkt-baselines/Nemotron-3-Diarization/…)의
#            AMI 창·회의 DER(diar_eval run --n3d, 우리 참조 — results/diar-n3d-c4-win·-session) + E9 final 의 슬롯 활동을 전문가 활동으로 바꾼 AMI 창 δ10·회의 통째 δ10(8 몫씩).
#            의존 없음. 순서: 창 DER → 창 cpWER → 회의 DER → 회의 cpWER.
#   q17-e10-n3d  같은 진단을 E10 final 에(창·회의 통째 × 섞인 소리 켬·끔). --after-job(E10 학습 job) 필수.
#   q17-e11 · q17-e11b  E11(slurm/submit-q17-e11.sh — 전문가 활동으로 학습, b = E10 에서): AMI 창 δ10 8 몫 → 회의 통째 δ10 8 몫(config 의 전문가). --after-job 필수.
#   q17-graft  G1 이식 어댑터(2026-10-29, slurm/submit-n3d-graft.sh — GRAFT_RUN 기본 q17-graft-l8): final/graft.pt 의 창·회의 DER(diar_eval run --graft,
#            스트리밍 청크 4, 우리 참조 — results/diar-graft-<run>-c4-win·-session, 각 8 몫). 교사 기준은 q17-e9-n3d 의 diar-n3d-c4-*. --after-job 필수.
#   q17-e8  E8 헤드 출력 지연 R(2026-10-27, slurm/submit-q17-e8.sh — 헤드 위치 p = 프레임 p − R, 받아쓰기 지연 δ ≥ R): q17-e6 과 같은 v3 묶음(헤드 예측 활동).
#           정답 활동 상한(amiact)은 넣지 않는다 — --act-oracle 이 활동을 R 만큼 밀지 않는다(E6 의 상한과 같은 값이 될 까닭도 없다). --after-job 필수.
#   q17-e5  E5 한 스트림 + 슬롯 활동 입력(2026-10-26): q17-e5 final 을 AMI 정답 슬롯 활동 입력(amiact, 8 몫)과 활동 없이(ami, 4 몫 — 입력이 없을 때 얼마나 무너지나)로. --after-job 필수.
#   q17-e3  E3(2026-10-25): 같은 q17-e2-tgt final 에 정답 대신 E1 화자 분리 가지의 예측 활동(v3_eval_task.sh amie3, 임계 E3_THR — 기본 0.45 = E1 AMI dev 에서 고른 값, 바꾸려면 서브셸에서 export)으로 8 몫.
#           --after-job 은 E2 학습이 아직 돌 때만 준다(끝났으면 빼고 낸다 — 작업이 DONE 을 확인하고 돈다. 끝난 지 오래된 job 은 스케줄러 기록에서 빠져 의존이 거부될 수 있다).
#   --job-name NAME(2026-10-27, 사용자 "Job 이름은 SA_EVAL_FullDuplex 로 설정해주고") 평가 노드 job 이름(기본 SA_EVAL_FullDuplex — 그 전에는 sbatch 의 SA_V3_EVAL;
#              로그는 logs/train/<이름>-<job>.out).
#   --after-job JOBID  의존 job 을 바꾼다(재제출 등).   --dry-run  목록·sbatch 명령만 출력(파일·제출 없음).   --partition P  평가 노드 파티션(기본 sbatch 의 apex — 예 low_p_hpc).
#   --cks A,B  체크포인트 평가·avg3 에 쓸 checkpoint-A·checkpoint-B(기본 1500,2000 — 2,500 step run 의 60·80 %; 2026-10-26 F1 5,000 step 은 3000,4000).
# 시간 한도는 24 h 로 낸다(sbatch 기본 12 h 를 명령줄로 덮는다). 목록은 VAPKT-data/tmp/evals-<묶음>-<시각>.txt(새 파일).
# 각 작업은 끝났거나 시작된 산출이 있으면 건너뛰거나 이어 하고(AMI 몫은 이어 한다) 아무것도 지우지 않는다.
set -euo pipefail
cd "$(dirname "$0")/.."
set_=${1:-}; [[ $# -gt 0 ]] && shift; dep=""; dry=0; part=""; cks=1500,2000; jname=SA_EVAL_FullDuplex
while [[ $# -gt 0 ]]; do
  case $1 in
    --after-job) dep=$2; shift 2 ;;  --job-name) jname=$2; shift 2 ;;  --dry-run) dry=1; shift ;;  --partition) part=$2; shift 2 ;;  --cks) cks=$2; shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
  esac
done
case $set_ in r1x-ami|v2-amifull|q17-e3|q17-e6-amifull|q17-e7-amifull-guide|q17-e6-amifull-d10) ;; q17-s1) dep=${dep:-94809} ;; q17-v3) dep=${dep:-94814} ;;
  noeot-v3|mt4-v3|noeot-mt4-v3|q17-noeot-v3|q17-noeot-mt4-v3|q17-mt4-v3|mt4-amifull|q17-mt4-amifull|q17-e2-tgt|q17-f1|q17-e5|q17-e6|q17-e8|q17-e8-ami10|q17-e7b|q17-e9|q17-e9b|q17-e10|q17-e7-amifull|q17-e10-n3d|q17-e11|q17-e11b|q17-graft) [[ -n $dep ]] || { echo "$set_ 는 --after-job(학습 job)이 필요하다" >&2; exit 2; } ;;
  q17-e9-n3d) ;;
  *) echo "묶음: r1x-ami | q17-s1 | q17-v3 | noeot-v3 | mt4-v3 | noeot-mt4-v3 | q17-noeot-v3 | q17-noeot-mt4-v3 | q17-mt4-v3 | v2-amifull | mt4-amifull | q17-mt4-amifull | q17-e2-tgt | q17-e3 | q17-f1 | q17-e5 | q17-e6 | q17-e8 | q17-e8-ami10 | q17-e7b | q17-e9 | q17-e9b | q17-e10 | q17-e7-amifull-guide | q17-e6-amifull-d10 | q17-e6-amifull | q17-e7-amifull | q17-e9-n3d | q17-e10-n3d | q17-e11 | q17-e11b | q17-graft" >&2; exit 2 ;; esac
[[ -z $dep || $dep =~ ^[0-9]+$ ]] || { echo "--after-job 은 job id" >&2; exit 2; }
[[ $cks =~ ^[1-9][0-9]*,[1-9][0-9]*$ ]] || { echo "--cks A,B(step 둘) — $cks" >&2; exit 2; }; CKA=${cks%,*}; CKB=${cks#*,}
D=/soundai/users/tskim/VAPKT-data; MR=$D/models; X=raw/sources/experiments/2026-10-21-v3-attribution/v3_eval_task.sh
L=()
ami4() { local g0=$1 run=$2 root=$3 r; for r in 0 1 2 3; do L+=("$((g0 + r))|${root:+MODEL_ROOT=$root }bash $X ami $run final $r/4"); done; }   # GPU g0..g0+3 에 4 몫
ami8d10() { local run=$1 r; for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 bash $X ami $run final $r/8"); done; }                # δ10 만, GPU 0–7 에 8 몫(2026-10-28)
v3set() {   # V3 run $1: ck1500·ck2000·final·avg3 × 9 작업 + final·avg3 AMI 4 몫씩 — 평균은 GPU 7 의 첫 작업, 평균 평가는 그 DONE 을 기다린다
  local RUN=$1 AVG=$1-avg3 M=/soundai/Model/VAPASR/stage3-$1 i=0 t k; local -a K=(dlg xl taxi40 stage3 regress xltel xleval taxi90 onset)
  L+=("7|\${PY:-python} experiments/avg_checkpoints.py --out $MR/stage3-$AVG/final --done $M/checkpoint-$CKA $M/checkpoint-$CKB $M/final > $D/tmp/avg-$RUN.log 2>&1")
  for t in checkpoint-$CKA checkpoint-$CKB final; do for k in "${K[@]}"; do L+=("$((i % 8))|bash $X $k $RUN $t"); i=$((i + 1)); done; done
  for k in "${K[@]}"; do L+=("$((i % 8))|MODEL_ROOT=$MR bash $X $k $AVG final"); i=$((i + 1)); done
  ami4 4 $RUN ""; ami4 0 $AVG $MR                                                           # 단일 체크포인트 작업이 적은 GPU 4–7 에 final 몫
}
amifull8() { local run=$1 root=$2 r; for r in 0 1 2 3 4 5 6 7; do L+=("$r|${root:+MODEL_ROOT=$root }bash $X amifull $run final $r/8"); done; }   # 회의 통째 8 몫(GPU 0–7)
fullset() {   # run $1 의 final·avg3 회의 통째 — 평균은 GPU 7 의 첫 작업(v3set 과 같은 명령), 평균 평가는 그 DONE 을 기다린다
  local RUN=$1 M=/soundai/Model/VAPASR/stage3-$1
  L+=("7|\${PY:-python} experiments/avg_checkpoints.py --out $MR/stage3-$RUN-avg3/final --done $M/checkpoint-$CKA $M/checkpoint-$CKB $M/final > $D/tmp/avg-$RUN-amifull.log 2>&1")
  amifull8 $RUN ""; amifull8 $RUN-avg3 $MR
}
case $set_ in
  r1x-ami) ami4 0 v3-p1-tel-xlw ""; ami4 4 v3-p1-tel-xlw-avg3 $MR ;;
  q17-s1)
    CK=/soundai/Model/VAPASR/semcommit-q17-s1m3-d10-s0/final; EV=$D/data/evaluation
    L+=("0|bash experiments/run_single_turn_asr_mxc.sh $CK $D/results/single-turn-vpaa-kspon-q17-s1m3-d10-s0 0,1 64 0 4,6,8,10 $EV/single-turn-vpaa-kspon-v1/manifest.jsonl final")
    L+=("2|bash experiments/run_single_turn_asr_mxc.sh $CK $D/results/single-turn-yodas-de-q17-s1m3-d10-s0 2,3 64 0 4,6,8,10 $EV/single-turn-yodas-de-dev-v1/manifest.jsonl final") ;;
  q17-v3) v3set q17-v3-p1-tel-xlw ;;
  noeot-v3) v3set v3-p1-tel-xlw-noeot; L+=("4|bash $X stage3 st3-p11-revive-noeot checkpoint-3000") ;;      # Stage 3 같은 언어 평가(0.6B p11 ck3000 과 비교)
  mt4-v3) v3set v3-p1-tel-xlw-mt4 ;;
  noeot-mt4-v3) v3set v3-p1-tel-xlw-noeot-mt4 ;;
  q17-mt4-v3) v3set q17-v3-p1-tel-xlw-mt4 ;;
  q17-f1) v3set q17-f1 ;;
  q17-e6) v3set q17-e6; for r in 0 1 2 3 4 5 6 7; do L+=("$r|bash $X amiact q17-e6 final $r/8"); done ;;
  q17-e8) v3set q17-e8 ;;
  q17-e7-amifull-guide) for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 AOSC=1 DIAR=/soundai/Model/VAPASR/diar-q17-diar-e7a/final/diar.pt DIAR_TAG=e7a GUIDE=1 bash $X amifull q17-e6 final $r/8"); done ;;
  q17-e8-ami10) ami8d10 q17-e8 ;;
  q17-e6-amifull-d10) for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 bash $X amifull q17-e6 final $r/8"); done ;;
  q17-e7b) ami8d10 q17-e7b; for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 AOSC=1 DIAR=/soundai/Model/VAPASR/diar-q17-diar-e7a/final/diar.pt DIAR_TAG=e7a bash $X amifull q17-e7b final $r/8"); done ;;   # 창 AMI 를 먼저(약 20 분)
  q17-e9|q17-e9b) ami8d10 $set_; for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 AOSC=1 DIAR=/soundai/Model/VAPASR/diar-q17-diar-e7a/final/diar.pt DIAR_TAG=e7a bash $X amifull $set_ final $r/8"); done ;;
  q17-e10) E7A=/soundai/Model/VAPASR/diar-q17-diar-e7a/final/diar.pt; ami8d10 q17-e10                       # 평가 순서: 창 → 회의 통째 → 섞인 소리 끈 창 → 끈 회의 통째
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 AOSC=1 DIAR=$E7A DIAR_TAG=e7a bash $X amifull q17-e10 final $r/8"); done
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 MIXOFF=1 bash $X ami q17-e10 final $r/8"); done
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 MIXOFF=1 AOSC=1 DIAR=$E7A DIAR_TAG=e7a bash $X amifull q17-e10 final $r/8"); done ;;
  q17-e9-n3d) N3=${N3D:-$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo}; [[ -f $N3 ]] || { echo "Nemotron 3 Diarization 이 없다: $N3(사용자 업로드)" >&2; exit 1; }
    DW=$D/results/diar-n3d-c4-win; DSS=$D/results/diar-n3d-c4-session
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|mkdir -p $DW && \${PY:-python} experiments/diar_eval.py run --n3d $N3 --out $DW --rank $r --world-size 8 >> $DW/run-rank$r.log 2>&1; \${PY:-python} experiments/diar_eval.py summarize --out $DW > $DW/summary.log 2>&1"); done
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 N3D=$N3 bash $X ami q17-e9 final $r/8"); done
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|mkdir -p $DSS && \${PY:-python} experiments/diar_eval.py run --n3d $N3 --session --out $DSS --rank $r --world-size 8 >> $DSS/run-rank$r.log 2>&1; \${PY:-python} experiments/diar_eval.py summarize --out $DSS/n3d > $DSS/summary.log 2>&1"); done
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 N3D=$N3 bash $X amifull q17-e9 final $r/8"); done ;;
  q17-e10-n3d) N3=${N3D:-$D/vapkt-baselines/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo}; [[ -f $N3 ]] || { echo "Nemotron 3 Diarization 이 없다: $N3(사용자 업로드)" >&2; exit 1; }
    for t in ami amifull; do for mo in 0 1; do for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 MIXOFF=$mo N3D=$N3 bash $X $t q17-e10 final $r/8"); done; done; done ;;
  q17-graft) GR=${GRAFT_RUN:-q17-graft-l8}; G=/soundai/Model/VAPASR/graft-$GR/final/graft.pt; DW=$D/results/diar-graft-$GR-c4-win; DSS=$D/results/diar-graft-$GR-c4-session
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|mkdir -p $DW && \${PY:-python} experiments/diar_eval.py run --graft $G --out $DW --rank $r --world-size 8 >> $DW/run-rank$r.log 2>&1; \${PY:-python} experiments/diar_eval.py summarize --out $DW > $DW/summary.log 2>&1"); done
    for r in 0 1 2 3 4 5 6 7; do L+=("$r|mkdir -p $DSS && \${PY:-python} experiments/diar_eval.py run --graft $G --session --out $DSS --rank $r --world-size 8 >> $DSS/run-rank$r.log 2>&1; \${PY:-python} experiments/diar_eval.py summarize --out $DSS/graft > $DSS/summary.log 2>&1"); done ;;
  q17-e11|q17-e11b) ami8d10 $set_; for r in 0 1 2 3 4 5 6 7; do L+=("$r|DELTAS=10 bash $X amifull $set_ final $r/8"); done ;;
  q17-e6-amifull) amifull8 q17-e6 ""; for r in 0 1 2 3 4 5 6 7; do L+=("$r|AOSC=1 bash $X amifull q17-e6 final $r/8"); done ;;
  q17-e7-amifull) for r in 0 1 2 3 4 5 6 7; do L+=("$r|AOSC=1 DIAR=/soundai/Model/VAPASR/diar-q17-diar-e7a/final/diar.pt DIAR_TAG=e7a bash $X amifull q17-e6 final $r/8"); done ;;
  q17-e5) for r in 0 1 2 3 4 5 6 7; do L+=("$r|bash $X amiact q17-e5 final $r/8"); done; ami4 0 q17-e5 "" ;;
  q17-noeot-v3) v3set q17-v3-p1-tel-xlw-noeot ;;
  q17-noeot-mt4-v3) v3set q17-v3-p1-tel-xlw-noeot-mt4 ;;
  v2-amifull) amifull8 v2-p1 "" ;;
  mt4-amifull) fullset v3-p1-tel-xlw-mt4 ;;
  q17-mt4-amifull) fullset q17-v3-p1-tel-xlw-mt4 ;;
  q17-e2-tgt) for r in 0 1 2 3 4 5 6 7; do L+=("$r|bash $X amitgt q17-e2-tgt final $r/8"); done ;;
  q17-e3) for r in 0 1 2 3 4 5 6 7; do L+=("$r|E3_THR=${E3_THR:-0.45} bash $X amie3 q17-e2-tgt final $r/8"); done ;;
esac
T=$D/tmp/evals-$set_-$(date +%Y%m%d-%H%M%S).txt
cmd="sbatch --job-name=$jname ${part:+--partition=$part }${dep:+--dependency=afterok:$dep }--time=24:00:00 --export=ALL,TASKS=$T slurm/v3-eval-node.sbatch"
if (( dry )); then printf '%s\n' "${L[@]}"; echo "$cmd"; echo "Submitted batch job 0"; exit 0; fi
[[ -e $T ]] && { echo "이미 있다: $T" >&2; exit 1; }
{ echo "# 평가 묶음 $set_(slurm/submit-evals.sh, $(date '+%F %T'))${dep:+ — job $dep 성공 뒤}"; printf '%s\n' "${L[@]}"; } > "$T"
echo "목록 $T (${#L[@]} 작업)" >&2
$cmd
