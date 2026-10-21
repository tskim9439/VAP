# V3 평가 재실행(mxc 호스트): 공용 GPU 의 CUDA OOM 으로 깨진 단계만 다시 — bash v3_eval_rerun.sh
#   대화 번역(cst_stage4_eval dlg): R1 ck1500·R2 ck1500·R2 final — 같은 --out 에서 이어 한다(설정 같음, 배치만 32 → 16 — 배치는 설정 키가 아니다).
#   Stage 3 회귀·오류 분해: R1 final(모델 로드 중 OOM) — 같은 --out 에서 이어 한다.
# 빈 메모리 MIN_FREE_MB(기본 60000) GPU 를 기다린다. 아무것도 지우지 않는다(이어 쓰기만).
set -u
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; R=/soundai/users/tskim/VAPKT-data/results; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16
gpu() { while true; do g=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2, $4}' | sort -k2,2nr -k3,3n | head -1)
  [ "$(echo $g | cut -d' ' -f2)" -ge ${MIN_FREE_MB:-60000} ] && { export CUDA_VISIBLE_DEVICES=$(echo $g | cut -d' ' -f1); return; }; sleep 120; done; }
for x in v3-p1:checkpoint-1500:ck1500 v3-p1-nodata:checkpoint-1500:ck1500 v3-p1-nodata:final:final; do
  RUN=${x%%:*}; t=$(echo $x | cut -d: -f2); tag=${x##*:}; O=$R/stage4-eval-$RUN-$tag; gpu
  echo "[$(date '+%F %T')] $RUN/$t 대화 번역 이어 하기 → GPU $CUDA_VISIBLE_DEVICES"
  $P experiments/cst_stage4_eval.py run --checkpoint /soundai/Model/VAPASR/stage3-$RUN/$t --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 \
    --limit-windows 100 --trans-lead-space --batch-size 16 --backchannel 1.0,3 --out $O/dlg >> $O/dlg.log 2>&1; $P experiments/cst_stage4_eval.py summarize --out $O/dlg > $O/dlg-summary.log 2>&1
  echo "[$(date '+%F %T')] $RUN/$t 대화 번역 끝 — $(tail -1 $O/dlg.log | cut -c1-120)"
done
RUN=v3-p1; tag=final; E3=$R/stage3-eval-$RUN-$tag; A=$R/stage3-attr-errors-$RUN-$tag; gpu
echo "[$(date '+%F %T')] $RUN/final Stage 3 회귀 이어 하기 → GPU $CUDA_VISIBLE_DEVICES"
$P experiments/cst_stage3_eval.py run --checkpoint /soundai/Model/VAPASR/stage3-$RUN/final --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E3 --deltas 4,10 --limit-windows 200 \
  --batch-size 16 --verify --backchannel 1.0,3 >> $E3/run.log 2>&1; $P experiments/cst_stage3_eval.py summarize --out $E3 > $E3/summary.log 2>&1
$P experiments/cst_attr_errors.py --pred $E3 --data $P2 --split $P2/splits/v2.json --out $A --deltas 4,10 --workers 16 >> $A/run.log 2>&1
echo "[$(date '+%F %T')] $RUN/final Stage 3 끝 — $(tail -1 $E3/summary.log | cut -c1-120)"
echo "V3_RERUN_DONE"
