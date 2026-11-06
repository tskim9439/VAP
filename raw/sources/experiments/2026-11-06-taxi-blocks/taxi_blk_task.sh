#!/usr/bin/env bash
# TAXI(실제 독↔영 전화 대화) 평가를 슬롯 블록 모델(E13·E15b)로(2026-11-06, 사용자 "응 진행해" — TAXI 를 최신 모델로 BLEU 까지).
#   experiments/cst_stage4_taxi.py 의 슬롯 블록 번역 디코드(decode_blocks_batch trans) → cstbench 채점(BLEU·chrF·StreamLAAL·빈 턴, 받아쓰기 WER·cp/agn/귀속 몫·턴 결정).
#   bash taxi_blk_task.sh <run> <대상 final|checkpoint-N> <판>   (GPU 는 부르는 쪽이 — v3-eval-node.sbatch)
#   판 = <활동>-<슬롯 언어>: 활동 n3d(config 의 고정 N3D) | g2b(이식 학생 G2b + G2b 인코더 — 종단간, GRAFT_RUN 기본 q17-graft-g2b),
#        슬롯 언어 first(창의 첫 화자 언어를 슬롯 0 — "누가 먼저 말할지 안다") | hash(학습의 교차 언어 규칙).
#   δ10, 창 90 s(taxi90 과 같은 창). 산출 results/taxi-blk-<run>-<태그>-<판>/(hyp·asr 은 TAXI 글자 — mxc 에만, report·asr-report·turn-metrics 는 집계).
#   끝난 판은 건너뛴다(report-d10.json). 아무것도 지우지 않는다.
RUN=${1:?run}; t=${2:?대상}; V=${3:?판}
[[ $V =~ ^(n3d|g2b)-(first|hash)$ ]] || { echo "판은 n3d|g2b-first|hash: $V"; exit 2; }
ACT=${V%-*}; SL=${V#*-}
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
SD0=${VAPASR_STAGE_DIR:-}
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=${SD0:-/tmp/sa_tskim/vapasr-stage}
P=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}; R=/soundai/users/tskim/VAPKT-data/results; D=/soundai/users/tskim/VAPKT-data/data
TAXI=${TAXI_SESSIONS:-$D/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl}
OUT=${MODEL_ROOT:-/soundai/Model/VAPASR}/stage3-$RUN; CK=$OUT/$t; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag
O=${TAXI_OUT:-$R/taxi-blk-$RUN-$tag-$V}
say() { echo "[$(date '+%F %T')] taxi-blk $RUN/$t $V(GPU ${GPU_IDX:-${CUDA_VISIBLE_DEVICES:-?}}): $*"; }
if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 60; done; else until [ -f $CK/trainer_state.json ]; do sleep 60; done; fi
[ -f $O/report-d10.json ] && { say "이미 끝"; exit 0; }; mkdir -p $O
GX=(); if [ $ACT = g2b ]; then G=/soundai/Model/VAPASR/graft-${GRAFT_RUN:-q17-graft-g2b}/final/graft.pt; [ -f $G ] || { say "이식 학생 없음 $G"; exit 2; }; GX=(--graft-expert $G --encoder-state $G); fi
say "시작 → $O"
$P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O --deltas 10 --max-window 90 --batch-size 8 \
  --slot-lang $SL "${GX[@]}" > $O/run.log 2>&1 || { say "디코드 실패 — $O/run.log"; exit 1; }
$P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O --turn-metrics > $O/summary.log 2>&1 || { say "채점 실패 — $O/summary.log"; exit 1; }
say "끝 — $(grep -E 'BLEU' $O/summary.log | tr '\n' ' ' | cut -c1-300)"
