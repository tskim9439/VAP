#!/usr/bin/env bash
# 대화 번역 평가를 몫으로(2026-11-05, 사용자 "평가는 왜 5시간이나 걸리는 거지?" → 번역 92 분이 한 GPU 에 몰려 마지막을 잡았다. 슬롯 평가(100350)에서는
# GPU 2 에 회의 디코드 4 개와 같이 돌다 메모리가 모자랐다). v3_eval_task.sh dlg 와 같은 명령을 cst_stage4_eval --rank r --world-size w 로 나눠 돌고,
# 마지막 몫이 summarize 해 dlg-summary.log 를 쓴다(e15a_report·auto_e15a 가 읽는 이름).
#   bash dlg_shard_task.sh <run> <대상> <r/w>   (GPU 는 부르는 쪽이 — v3-eval-node.sbatch). 산출 results/stage4-eval-<run>-<태그>/dlg<w>(새 폴더 — 몫 없는 dlg 와 재개 지문이 달라 섞지 않는다).
#   끝난 몫은 건너뛰고(.rank<r>-of-<w>.done), 덜 끝난 몫은 이어 한다. 아무것도 지우지 않는다.
RUN=${1:?run}; t=${2:?대상}; SH=${3:?r/w}; r=${SH%/*}; w=${SH#*/}
[[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { echo "몫은 r/w(0 ≤ r < w): $SH"; exit 2; }
SD0=${VAPASR_STAGE_DIR:-}
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=${SD0:-/tmp/sa_tskim/vapasr-stage}
P=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}; R=/soundai/users/tskim/VAPKT-data/results; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16
OUT=${MODEL_ROOT:-/soundai/Model/VAPASR}/stage3-$RUN; CK=$OUT/$t; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; O=$R/stage4-eval-$RUN-$tag; DD=$O/dlg$w
say() { echo "[$(date '+%F %T')] dlg 몫 $RUN/$t(GPU ${GPU_IDX:-${CUDA_VISIBLE_DEVICES:-?}}): $*"; }
if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 60; done; else until [ -f $CK/trainer_state.json ]; do sleep 60; done; fi
[ -f $DD/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $DD; [ -f $DD/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
say "몫 $r/$w 시작"
$P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 --limit-windows 100 \
  --trans-lead-space --batch-size 32 --backchannel 1.0,3 --rank $r --world-size $w --out $DD >> $DD/run-rank$r.log 2>&1
grep -q "^DONE rank=$r " $DD/run-rank$r.log || { say "몫 $r/$w 실패 — $DD/run-rank$r.log"; exit 1; }
touch $DD/.rank$r-of-$w.done; n=$(ls $DD/.rank*-of-$w.done 2>/dev/null | wc -l)
[ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
( flock 9; [ -f $DD/summary.json ] && exit 0
  $P experiments/cst_stage4_eval.py summarize --out $DD > $O/dlg-summary.log 2>&1 ) 9> $DD/.sum.lock
say "번역 합침 → $O/dlg-summary.log"
