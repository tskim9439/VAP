# V3 평가 작업 하나: bash v3_eval_task.sh <작업> <run> <대상> — GPU 는 부르는 쪽이 CUDA_VISIBLE_DEVICES 로 준다(slurm/v3-eval-node.sbatch 일꾼, GPU_IDX = 그 번호).
#   작업: dlg | xl | taxi40 | stage3 | regress | xltel | xleval | taxi90 | onset | ami — stage4_eval_queue.sh(1–6 단계)·v3_eval_queue.sh·v3_dev_decode.sh·onset_probe.sh 와
#   같은 명령·산출 폴더다(TRANS_LEAD_SPACE=1 과 같이 --trans-lead-space — V3 run 은 모두 그렇게 학습했다).
#   체크포인트: MODEL_ROOT(기본 /soundai/Model/VAPASR)/stage3-<run>/<대상>. 대상이 final 이면 DONE, 아니면 trainer_state.json 이 생길 때까지 기다린다.
#   건너뛰기: 끝난 작업(요약·보고 파일 있음). xltel·xleval·taxi90·onset 은 산출 폴더가 있으면(호스트 큐가 시작했거나 끝냄) 건너뛴다 — 같은 일을 두 번 하지 않는다.
#   PY 로 python 을 바꾼다(노드 로컬 env). 아무것도 지우지 않는다.
#   ami(2026-10-23, 사용자 "AMI_IHM 이랑 AMI_SDM 에 대해서도 평가를 수행하고 싶어"): AMI 공식 dev 18·eval 16 회의(ami-bench-v1 — 공식 Mix-Headset = IHM-mix,
#     Array1-01 = SDM) 창 전부, δ4·10, 4 슬롯 선언 — V2 v2_eval_extra.sh 12 단계와 같은 명령. 다섯째 인자 r/w 로 창을 w 몫으로 나눠 r 번째만 돈다
#     (cst_stage3_eval --rank·--world-size, 이어 하기 됨). 마지막으로 끝난 몫이 summarize·오류 분해·ami_report.py(공식 분할별)를 한 번 돈다.
#     산출 results/stage3-eval-ami-<run>-<태그>(+ ami-report.txt) · stage3-attr-errors-ami-<run>-<태그>.
set -u
K=${1:?작업}; RUN=${2:?run}; t=${3:?대상}; SH=${4:-0/1}
SD0=${VAPASR_STAGE_DIR:-}                                                                   # 부르는 쪽(평가 노드) 값 — .env 의 /scratch 는 컨테이너 전용이라 덮이면 안 된다
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=${SD0:-/tmp/sa_tskim/vapasr-stage}   # 2026-10-23 전에는 .env 값이 이겨 "stage 실패 → 원본에서 읽음"(NFS 직접)
P=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}; R=/soundai/users/tskim/VAPKT-data/results; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; XE=$D/sim-dialogue-xl-eval-v1; TAXI=$D/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl; DEV=$D/v3-dev
CH=$D/evaluation/single-turn-callhome-de-v1/manifest.jsonl; X=raw/sources/experiments/2026-10-21-v3-attribution
OUT=${MODEL_ROOT:-/soundai/Model/VAPASR}/stage3-$RUN; CK=$OUT/$t; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; O=$R/stage4-eval-$RUN-$tag; E3=$R/stage3-eval-$RUN-$tag
say() { echo "[$(date '+%F %T')] $K $RUN/$t(GPU ${GPU_IDX:-${CUDA_VISIBLE_DEVICES:-?}}): $*"; }
if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 60; done; else until [ -f $CK/trainer_state.json ]; do sleep 60; done; fi
dev() {                                                                                     # 전화 dev·짝 비교 셋(v3_dev_decode.sh 와 같다, bias 없음)
  local o=$R/v3dev-$RUN-$tag-$1-sessions; [ -e $o ] && { say "있다 — 건너뜀 $o"; return 0; }; mkdir -p $o
  $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $DEV/$1/sessions.jsonl --out $o --deltas 10 --max-window 90 --batch-size 8 > $o/run.log 2>&1
  $P experiments/cst_stage4_taxi.py score --sessions $DEV/$1/sessions.jsonl --out $o --turn-metrics > $o/summary.log 2>&1; }
say "시작"; mkdir -p $O
case $K in
  dlg) [ -f $O/dlg/summary.json ] && { say "이미 끝"; exit 0; }
       $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 --limit-windows 100 \
         --trans-lead-space --batch-size 32 --backchannel 1.0,3 --out $O/dlg >> $O/dlg.log 2>&1
       $P experiments/cst_stage4_eval.py summarize --out $O/dlg > $O/dlg-summary.log 2>&1 ;;
  xl) [ -f $O/xl/summary.json ] && { say "이미 끝"; exit 0; }
      $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $XE --trans $XE/targets-sim_en-de --corpora sim_en-de --deltas 4,10 --limit-windows 200 \
        --trans-lead-space --batch-size 32 --out $O/xl >> $O/xl.log 2>&1
      $P experiments/cst_stage4_eval.py summarize --out $O/xl > $O/xl-summary.log 2>&1 ;;
  taxi40) [ -f $O/taxi/report-d10.json ] && { say "이미 끝"; exit 0; }
          $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi --deltas 4,10 > $O/taxi.log 2>&1
          $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi > $O/taxi-summary.log 2>&1 ;;
  stage3) [ -f $E3/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E3 $R/stage3-attr-errors-$RUN-$tag
          $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E3 --deltas 4,10 --limit-windows 200 \
            --batch-size 32 --verify --backchannel 1.0,3 >> $E3/run.log 2>&1
          $P experiments/cst_stage3_eval.py summarize --out $E3 > $E3/summary.log 2>&1
          $P experiments/cst_attr_errors.py --pred $E3 --data $P2 --split $P2/splits/v2.json --out $R/stage3-attr-errors-$RUN-$tag --deltas 4,10 --workers 16 >> $R/stage3-attr-errors-$RUN-$tag/run.log 2>&1 ;;
  regress) G=$R/stage3-regress-$RUN-$tag
           if [ -f $G/libri-other/summary.json ]; then say "단일 화자 이미 끝"; else bash $R/stage3-regress/stage2_regress.sh $CK $RUN-$tag ${GPU_IDX:-0} > $R/stage3-regress/run-$RUN-$tag.log 2>&1; fi
           if [ -f $CH ] && [ ! -f $G/callhome-de/summary.json ]; then mkdir -p $G/callhome-de
             $P experiments/eval_single_turn_asr.py run --manifest $CH --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --out $G/callhome-de --deltas 4 10 --batch-size 64 > $G/callhome-de/run.log 2>&1
             $P experiments/eval_single_turn_asr.py summarize --out $G/callhome-de > $G/callhome-de/summary.log 2>&1; fi ;;
  xltel) dev xl-tel-dev ;;
  xleval) dev xl-eval-v1 ;;
  taxi90) [ -e $O/taxi90 ] && { say "있다 — 건너뜀"; exit 0; }
          $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi90 --deltas 4,10 --max-window 90 --batch-size 8 > $O/taxi90.log 2>&1
          $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi90 --turn-metrics > $O/taxi90-summary.log 2>&1 ;;
  onset) o=$R/taxi-v3onset-$RUN-$tag; [ -e $o ] && { say "있다 — 건너뜀"; exit 0; }; mkdir -p $o
         $P $X/taxi_v3_probe.py --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $o --deltas 10 --max-window 90 --classify-onsets > $o/run.log 2>&1 ;;
  ami) r=${SH%/*}; w=${SH#*/}; [[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { say "몫은 r/w(0 ≤ r < w)"; exit 2; }
       E=$R/stage3-eval-ami-$RUN-$tag; A=$R/stage3-attr-errors-ami-$RUN-$tag; AB=$D/ami-bench-v1
       [ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
       $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas 4,10 --batch-size 16 \
         --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
       grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
       touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
       [ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
       ( flock 9; [ -f $E/summary.json ] && exit 0
         $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
         mkdir -p $A; $P experiments/cst_attr_errors.py --pred $E --data $AB --out $A --deltas 4,10 --workers 16 > $A/run.log 2>&1
         $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1 ) 9> $E/.sum.lock
       say "AMI 합침 → $E/ami-report.txt" ;;
  *) say "모르는 작업"; exit 2 ;;
esac
say "끝"
